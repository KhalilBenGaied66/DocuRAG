"""HTTP API. Run with `docurag serve` (or uvicorn docurag.api.app:app)."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, File, Form, Request, Response, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from docurag import __version__
from docurag.api.auth import current_user, require_admin
from docurag.api.errors import ApiError, install_error_handlers
from docurag.ingestion.parsers import SUFFIX_TYPES
from docurag.logging_setup import request_id_var
from docurag.models import Answer, ChatTurn, DocumentSpec, User

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class JobAccepted(BaseModel):
    job_id: UUID
    status_url: str


def _container(request: Request) -> Any:
    return request.app.state.container


@router.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", tags=["ops"])
async def readyz(request: Request) -> dict[str, str]:
    try:
        async with _container(request).pool.connection() as conn:
            await asyncio.wait_for(conn.execute("SELECT 1"), timeout=2)
    except Exception as exc:
        raise ApiError(503, "not_ready", "database unreachable") from exc
    return {"status": "ready"}


@router.post("/v1/ask", response_model=Answer, response_model_exclude={"contexts"}, tags=["ask"])
async def ask(body: AskRequest, request: Request, user: User = Depends(current_user)) -> Answer:
    return await _container(request).service.answer(
        body.question, user, history=body.history, request_id=request.state.request_id
    )


@router.post("/v1/ask/stream", tags=["ask"])
async def ask_stream(
    body: AskRequest, request: Request, user: User = Depends(current_user)
) -> EventSourceResponse:
    """SSE events: `sources`, `token`…, `done`. The `done` event carries the verified text
    and citations; it is authoritative over the streamed tokens."""
    service = _container(request).service

    async def events():  # noqa: ANN202
        stream = service.stream(
            body.question, user, history=body.history, request_id=request.state.request_id
        )
        async with aclosing(stream):
            async for event in stream:
                if await request.is_disconnected():
                    break  # closing the stream cancels generation: unread tokens are still billed
                data = event.data
                if isinstance(data, BaseModel):
                    data = data.model_dump(mode="json", exclude={"contexts"})
                yield {"event": event.type, "data": json.dumps(data, ensure_ascii=False)}

    return EventSourceResponse(events())


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]", "_", Path(name).name)[:120] or "upload"


async def _save_upload(file: UploadFile, dest: Path) -> None:
    """Stream to disk in 1 MB pieces with a size cap; file I/O stays off the event loop."""
    await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
    out = await asyncio.to_thread(dest.open, "wb")
    size = 0
    try:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                limit_mb = MAX_UPLOAD_BYTES // 2**20
                raise ApiError(413, "too_large", f"files are limited to {limit_mb} MB")
            await asyncio.to_thread(out.write, chunk)
    except BaseException:
        out.close()
        await asyncio.to_thread(dest.unlink, missing_ok=True)
        raise
    out.close()


@router.post("/v1/documents", status_code=202, response_model=JobAccepted, tags=["documents"])
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    acl_groups: str = Form(..., description="comma-separated groups allowed to read it"),
    source_uri: str | None = Form(None, description="stable ID; re-uploading it replaces it"),
    doc_key: str | None = Form(None),
    version: str | None = Form(None),
    title: str | None = Form(None),
    metadata: str = Form("{}"),
    record_id_field: str | None = Form(None),
    record_text_fields: str | None = Form(None),
    user: User = Depends(require_admin),
) -> JobAccepted:
    name = _safe_name(file.filename or "upload")
    if Path(name).suffix.lower() not in SUFFIX_TYPES:
        raise ApiError(415, "unsupported_type", f"accepted types: {sorted(SUFFIX_TYPES)}")
    try:
        meta = json.loads(metadata)
        if not isinstance(meta, dict):
            raise ValueError
    except ValueError as exc:
        raise ApiError(422, "invalid_metadata", "metadata must be a JSON object") from exc
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    if not groups:
        raise ApiError(422, "invalid_acl", "at least one group is required")

    container = _container(request)
    dest = Path(container.settings.storage_dir) / uuid4().hex / name
    await _save_upload(file, dest)
    spec = DocumentSpec(
        source_uri=source_uri or f"upload://{name}",
        path=str(dest.resolve()),
        title=title, doc_key=doc_key, version=version, acl_groups=groups, metadata=meta,
        record_id_field=record_id_field,
        record_text_fields=[f.strip() for f in (record_text_fields or "").split(",") if f.strip()],
    )
    job_id = await container.repo.enqueue_job(spec, created_by=user.id)
    status_url = f"/v1/jobs/{job_id}"
    response.headers["Location"] = status_url
    return JobAccepted(job_id=job_id, status_url=status_url)


@router.get("/v1/jobs/{job_id}", tags=["documents"])
async def get_job(job_id: UUID, request: Request, _: User = Depends(require_admin)) -> dict:
    job = await _container(request).repo.get_job(job_id)
    if job is None:
        raise ApiError(404, "not_found", "no such job")
    return job


@router.get("/v1/documents/{document_id}", tags=["documents"])
async def get_document(
    document_id: UUID, request: Request, _: User = Depends(require_admin)
) -> dict:
    document = await _container(request).repo.get_document(document_id)
    if document is None:
        raise ApiError(404, "not_found", "no such document")
    return document


@router.delete("/v1/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(
    document_id: UUID, request: Request, _: User = Depends(require_admin)
) -> Response:
    if not await _container(request).repo.delete_document(document_id):
        raise ApiError(404, "not_found", "no such document")
    return Response(status_code=204)


def create_app(container_factory: Callable[[], Awaitable[Any]] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        if container_factory is None:
            from docurag.container import build_container

            app.state.container = await build_container()
        else:
            app.state.container = await container_factory()
        yield
        await app.state.container.close()

    app = FastAPI(title="DocuRAG", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next):  # noqa: ANN001, ANN202
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _REQUEST_ID.fullmatch(incoming) else uuid4().hex
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        return response

    install_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()

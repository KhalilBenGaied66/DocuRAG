"""One error format for every endpoint: application/problem+json (RFC 9457)."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

PROBLEM_JSON = "application/problem+json"


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, detail: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail, self.headers = status, code, detail, headers


def problem(
    request: Request, status: int, code: str, detail: str, *,
    headers: dict[str, str] | None = None, extra: dict | None = None,
) -> JSONResponse:
    body = {
        "type": f"urn:docurag:problem:{code}",
        "title": code.replace("_", " "),
        "status": status,
        "detail": detail,
        "request_id": getattr(request.state, "request_id", None),
        **(extra or {}),
    }
    return JSONResponse(body, status_code=status, headers=headers, media_type=PROBLEM_JSON)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return problem(request, exc.status, exc.code, exc.detail, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()
        ]
        return problem(
            request, 422, "invalid_request", "the request body or parameters are invalid",
            extra={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 400 and "parsing the body" in str(exc.detail):
            return problem(request, 400, "invalid_body", "the request body must be JSON encoded in UTF-8")
        return problem(request, exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return problem(request, 500, "internal_error", "unexpected error; quote the request_id")

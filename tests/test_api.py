from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from docurag.api.app import create_app
from docurag.models import Answer
from docurag.service import Event

DEV = {"X-Dev-User": "tech-42", "X-Dev-Groups": "maintenance"}
ADMIN = {"X-Dev-User": "admin-1", "X-Dev-Groups": "maintenance,docurag-admin"}


class StubService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def _answer(self, question: str, request_id: str | None) -> Answer:
        return Answer(request_id=request_id or "x", text="25 N·m [S1]", abstained=False,
                      standalone_question=question, prompt_version="answer-v1",
                      contexts=["must not leak"])

    async def stream(self, question, user, *, history=None, request_id=None, **_):
        self.calls.append((question, user.groups))
        yield Event("sources", [])
        yield Event("token", "25 N·m ")
        yield Event("token", "[S1]")
        yield Event("done", self._answer(question, request_id))

    async def answer(self, question, user, *, history=None, request_id=None, **_):
        self.calls.append((question, user.groups))
        return self._answer(question, request_id)


class StubRepo:
    def __init__(self) -> None:
        self.jobs: dict[UUID, object] = {}

    async def enqueue_job(self, spec, created_by):
        job_id = uuid4()
        self.jobs[job_id] = spec
        return job_id

    async def get_job(self, job_id):
        return {"id": str(job_id), "status": "queued"} if job_id in self.jobs else None

    async def get_document(self, document_id):
        return None

    async def delete_document(self, document_id):
        return False


@pytest.fixture
def client(settings):
    container = SimpleNamespace(settings=settings, service=StubService(), repo=StubRepo(), pool=None)

    async def close() -> None:
        return None

    container.close = close

    async def factory():
        return container

    with TestClient(create_app(container_factory=factory)) as c:
        c.container = container
        yield c


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_missing_identity_is_a_problem_json_401(client):
    response = client.post("/v1/ask", json={"question": "Couple M8 ?"})
    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["type"] == "urn:docurag:problem:unauthenticated" and body["request_id"]


def test_ask_uses_the_caller_groups_and_echoes_request_id(client):
    response = client.post("/v1/ask", json={"question": "Couple M8 ?"},
                           headers={**DEV, "X-Request-ID": "gmao-7f3a"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "gmao-7f3a"
    body = response.json()
    assert body["text"] == "25 N·m [S1]" and body["request_id"] == "gmao-7f3a"
    assert "contexts" not in body
    assert client.container.service.calls == [("Couple M8 ?", ["maintenance"])]


def test_untrusted_request_id_is_replaced(client):
    response = client.post("/v1/ask", json={"question": "Couple ?"},
                           headers={**DEV, "X-Request-ID": "bad id\nwith newline"})
    assert response.headers["x-request-id"] != "bad id\nwith newline"


def test_body_that_is_not_utf8_json_gets_a_clear_error(client):
    response = client.post("/v1/ask", content="{\"question\": \"apr\xe8s\"}".encode("latin-1"),
                           headers={**DEV, "Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["type"] == "urn:docurag:problem:invalid_body"


def test_validation_errors_use_problem_json(client):
    response = client.post("/v1/ask", json={"question": "x"}, headers=DEV)
    assert response.status_code == 422
    assert response.json()["errors"][0]["loc"][-1] == "question"


def test_stream_emits_sse_events_in_order(client):
    with client.stream("POST", "/v1/ask/stream", json={"question": "Couple M8 ?"},
                       headers=DEV) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]
    events = [line.split(": ", 1)[1] for line in lines if line.startswith("event:")]
    data = [line.split(": ", 1)[1] for line in lines if line.startswith("data:")]
    assert events == ["sources", "token", "token", "done"]
    done = json.loads(data[-1])
    assert done["text"] == "25 N·m [S1]" and "contexts" not in done


def test_upload_needs_admin(client):
    response = client.post("/v1/documents", headers=DEV, data={"acl_groups": "maintenance"},
                           files={"file": ("p200.md", b"# Titre\n\ntexte", "text/markdown")})
    assert response.status_code == 403


def test_upload_rejects_unsupported_types(client):
    response = client.post("/v1/documents", headers=ADMIN, data={"acl_groups": "maintenance"},
                           files={"file": ("setup.exe", b"MZ", "application/octet-stream")})
    assert response.status_code == 415


def test_upload_is_queued_not_processed_inline(client, settings):
    response = client.post(
        "/v1/documents", headers=ADMIN,
        data={"acl_groups": "maintenance, methods", "doc_key": "proc-12",
              "metadata": '{"site": "Lyon"}'},
        files={"file": ("../../procédure 12.md", b"# Titre\n\ntexte", "text/markdown")},
    )
    assert response.status_code == 202
    job_id = UUID(response.json()["job_id"])
    assert response.headers["location"] == f"/v1/jobs/{job_id}"
    spec = client.container.repo.jobs[job_id]
    assert spec.acl_groups == ["maintenance", "methods"] and spec.metadata == {"site": "Lyon"}
    assert spec.path.startswith(str(__import__("pathlib").Path(settings.storage_dir).resolve()))
    assert ".." not in spec.source_uri
    assert client.get(f"/v1/jobs/{job_id}", headers=ADMIN).json()["status"] == "queued"


def test_upload_rejects_bad_metadata(client):
    response = client.post("/v1/documents", headers=ADMIN,
                           data={"acl_groups": "maintenance", "metadata": "[1, 2]"},
                           files={"file": ("a.md", b"# t", "text/markdown")})
    assert response.status_code == 422


def test_unknown_document_is_404(client):
    response = client.delete(f"/v1/documents/{uuid4()}", headers=ADMIN)
    assert response.status_code == 404

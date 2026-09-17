from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from maintagent.app import create_app
from maintagent.config import AgentSettings
from maintagent.models import AgentPermissionError, TicketStateError, TicketView

TECH = {"X-Dev-User": "tech-42", "X-Dev-Groups": "maintenance"}
OTHER_TECH = {"X-Dev-User": "tech-99", "X-Dev-Groups": "maintenance"}
SUPERVISOR = {"X-Dev-User": "sup-7", "X-Dev-Groups": "maintenance,maintenance-supervisor"}
TICKET = {"ticket_id": "GMAO-2026-0101", "report": "Pompe P-200-07 : alarme E-41"}


class StubAgent:
    settings = AgentSettings(_env_file=None)

    def __init__(self) -> None:
        self.views: dict[str, TicketView] = {}

    async def submit(self, ticket_id, report, user):
        return self.views.setdefault(ticket_id, TicketView(
            ticket_id=ticket_id, status="awaiting_approval", requested_by=user.id, intent="incident",
        ))

    async def get(self, ticket_id):
        return self.views.get(ticket_id)

    async def decide(self, ticket_id, user, *, approved, feedback=None):
        if self.settings.supervisor_group not in user.groups:
            raise AgentPermissionError("approving needs the supervisor group")
        view = self.views[ticket_id]
        if view.status != "awaiting_approval":
            raise TicketStateError("no decision is pending on this ticket")
        self.views[ticket_id] = view.model_copy(
            update={"status": "work_order_created", "work_order_id": "WO-1"}
        )
        return self.views[ticket_id]

    async def resume(self, ticket_id):
        raise TicketStateError("the ticket has nothing to resume")


@pytest.fixture
def client(settings):
    container = SimpleNamespace(settings=settings, agent=StubAgent(), service=None, repo=None, pool=None)

    async def close() -> None:
        return None

    async def factory():
        return container

    container.close = close
    with TestClient(create_app(container_factory=factory)) as c:
        yield c


def test_submit_and_read_back(client):
    response = client.post("/v1/tickets", json=TICKET, headers=TECH)
    assert response.status_code == 200 and response.json()["status"] == "awaiting_approval"
    assert client.get("/v1/tickets/GMAO-2026-0101", headers=TECH).status_code == 200


def test_tickets_are_private_to_requester_and_supervisors(client):
    client.post("/v1/tickets", json=TICKET, headers=TECH)
    assert client.get("/v1/tickets/GMAO-2026-0101", headers=OTHER_TECH).status_code == 404
    assert client.get("/v1/tickets/GMAO-2026-0101", headers=SUPERVISOR).status_code == 200
    reuse = client.post("/v1/tickets", json=TICKET, headers=OTHER_TECH)
    assert reuse.status_code == 409


def test_decision_rules(client):
    client.post("/v1/tickets", json=TICKET, headers=TECH)
    denied = client.post("/v1/tickets/GMAO-2026-0101/decision", json={"approved": True}, headers=TECH)
    assert denied.status_code == 403
    assert denied.headers["content-type"].startswith("application/problem+json")

    ok = client.post("/v1/tickets/GMAO-2026-0101/decision", json={"approved": True}, headers=SUPERVISOR)
    assert ok.status_code == 200 and ok.json()["work_order_id"] == "WO-1"

    twice = client.post("/v1/tickets/GMAO-2026-0101/decision", json={"approved": True},
                        headers=SUPERVISOR)
    assert twice.status_code == 409


def test_resume_conflict_and_bad_ids(client):
    client.post("/v1/tickets", json=TICKET, headers=TECH)
    assert client.post("/v1/tickets/GMAO-2026-0101/resume", headers=TECH).status_code == 409
    assert client.get("/v1/tickets/bad id!", headers=TECH).status_code == 422
    bad = client.post("/v1/tickets", json={**TICKET, "ticket_id": "../../etc"}, headers=TECH)
    assert bad.status_code == 422


def test_docurag_endpoints_are_served_by_the_same_app(client):
    assert client.get("/healthz").json() == {"status": "ok"}

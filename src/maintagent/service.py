"""Ticket lifecycle on top of the graph: submit, decide, resume, read.

The thread ID is the ticket ID, so a ticket maps to exactly one durable run: submitting it
twice returns the same run, and a crashed run resumes from its last completed step.
Who may approve is enforced here, before the graph is resumed — the graph only ever sees
decisions that passed these checks.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import Command

from docurag.models import User
from maintagent.config import AgentSettings
from maintagent.graph import initial_state
from maintagent.models import (
    AgentPermissionError,
    Decision,
    TicketStateError,
    TicketStatus,
    TicketView,
)

log = logging.getLogger(__name__)


def pending_interrupts(snapshot: Any) -> list[Any]:
    return [i for task in snapshot.tasks for i in (task.interrupts or ())]


def ticket_status(snapshot: Any) -> TicketStatus:
    values = snapshot.values
    if pending_interrupts(snapshot):
        return "awaiting_approval"
    if snapshot.next:
        return "stalled"  # a step failed; the run can be resumed from its checkpoint
    if values.get("work_order_id"):
        return "work_order_created"
    if values.get("answer") is not None:
        return "answered"
    if values.get("decision") and not values["decision"].get("approved"):
        return "rejected"
    return "unknown"


def view_from_snapshot(ticket_id: str, snapshot: Any) -> TicketView:
    v = snapshot.values
    return TicketView(
        ticket_id=ticket_id, status=ticket_status(snapshot), requested_by=v.get("user_id"),
        intent=v.get("intent"), equipment_tag=v.get("equipment_tag"), answer=v.get("answer"),
        proposal=v.get("proposal"), issues=v.get("issues") or [],
        incomplete=bool(v.get("incomplete")), decision=v.get("decision"),
        work_order_id=v.get("work_order_id"), steps=v.get("steps", 0),
        tokens=v.get("tokens", 0), tool_calls=v.get("tool_log") or [],
    )


class MaintAgentService:
    def __init__(self, graph: Any, settings: AgentSettings) -> None:
        self.graph = graph
        self.settings = settings

    def _config(self, ticket_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": ticket_id}, "recursion_limit": self.settings.recursion_limit}

    async def _snapshot(self, ticket_id: str) -> Any:
        return await self.graph.aget_state(self._config(ticket_id))

    async def _run(self, ticket_id: str, payload: Any) -> None:
        try:
            await self.graph.ainvoke(payload, self._config(ticket_id))
        except Exception:
            # The checkpoint keeps every completed step; the ticket shows as 'stalled'.
            log.exception("agent run failed for ticket %s; resumable", ticket_id)

    async def get(self, ticket_id: str) -> TicketView | None:
        snapshot = await self._snapshot(ticket_id)
        return view_from_snapshot(ticket_id, snapshot) if snapshot.values else None

    async def _require(self, ticket_id: str) -> TicketView:
        view = await self.get(ticket_id)
        if view is None:
            raise TicketStateError(f"unknown ticket {ticket_id}")
        return view

    async def submit(self, ticket_id: str, report: str, user: User) -> TicketView:
        existing = await self.get(ticket_id)
        if existing is not None:
            return existing  # same ticket, same run
        await self._run(ticket_id, initial_state(ticket_id, report, user))
        return await self._require(ticket_id)

    async def decide(
        self, ticket_id: str, approver: User, *, approved: bool, feedback: str | None = None
    ) -> TicketView:
        snapshot = await self._snapshot(ticket_id)
        if not snapshot.values:
            raise TicketStateError(f"unknown ticket {ticket_id}")
        if not pending_interrupts(snapshot):
            raise TicketStateError("no decision is pending on this ticket")
        if self.settings.supervisor_group not in approver.groups:
            raise AgentPermissionError("approving needs the supervisor group")
        if approver.id == snapshot.values["user_id"]:
            raise AgentPermissionError("the requester cannot approve their own work order")
        decision = Decision(approved=approved, approver=approver.id, feedback=feedback)
        await self._run(ticket_id, Command(resume=decision.model_dump()))
        return await self._require(ticket_id)

    async def resume(self, ticket_id: str) -> TicketView:
        snapshot = await self._snapshot(ticket_id)
        if not snapshot.values:
            raise TicketStateError(f"unknown ticket {ticket_id}")
        if pending_interrupts(snapshot):
            raise TicketStateError("the ticket is waiting for a supervisor decision")
        if not snapshot.next:
            raise TicketStateError("the ticket has nothing to resume")
        await self._run(ticket_id, None)
        return await self._require(ticket_id)

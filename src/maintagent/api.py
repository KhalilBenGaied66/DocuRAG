"""Ticket endpoints, mounted next to DocuRAG's /v1/ask in the same service."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from docurag.api.auth import current_user
from docurag.api.errors import ApiError
from docurag.models import User
from maintagent.models import (
    TICKET_ID,
    AgentPermissionError,
    TicketStateError,
    TicketView,
)

router = APIRouter(prefix="/v1/tickets", tags=["tickets"])
TicketIdPath = PathParam(pattern=TICKET_ID)


class TicketRequest(BaseModel):
    ticket_id: str = Field(pattern=TICKET_ID, description="ID from the calling system (GMAO)")
    report: str = Field(min_length=5, max_length=4000)


class DecisionRequest(BaseModel):
    approved: bool
    feedback: str | None = Field(default=None, max_length=2000)


def _agent(request: Request) -> Any:
    return request.app.state.container.agent


def _can_see(view: TicketView, user: User, request: Request) -> bool:
    return view.requested_by == user.id or _agent(request).settings.supervisor_group in user.groups


async def _visible(request: Request, ticket_id: str, user: User) -> TicketView:
    view = await _agent(request).get(ticket_id)
    if view is None or not _can_see(view, user, request):
        raise ApiError(404, "not_found", "no such ticket")  # don't reveal other users' tickets
    return view


@router.post("", response_model=TicketView)
async def submit_ticket(
    body: TicketRequest, request: Request, user: User = Depends(current_user)
) -> TicketView:
    """Runs the agent until it answers, or until a proposal awaits a supervisor's decision."""
    view = await _agent(request).submit(body.ticket_id, body.report, user)
    if not _can_see(view, user, request):
        raise ApiError(409, "ticket_conflict", "this ticket ID is already used")
    return view


@router.get("/{ticket_id}", response_model=TicketView)
async def get_ticket(
    request: Request, ticket_id: str = TicketIdPath, user: User = Depends(current_user)
) -> TicketView:
    return await _visible(request, ticket_id, user)


@router.post("/{ticket_id}/decision", response_model=TicketView)
async def decide(
    body: DecisionRequest, request: Request, ticket_id: str = TicketIdPath,
    user: User = Depends(current_user),
) -> TicketView:
    try:
        return await _agent(request).decide(
            ticket_id, user, approved=body.approved, feedback=body.feedback
        )
    except AgentPermissionError as exc:
        raise ApiError(403, "forbidden", str(exc)) from exc
    except TicketStateError as exc:
        raise ApiError(409, "invalid_state", str(exc)) from exc


@router.post("/{ticket_id}/resume", response_model=TicketView)
async def resume(
    request: Request, ticket_id: str = TicketIdPath, user: User = Depends(current_user)
) -> TicketView:
    await _visible(request, ticket_id, user)
    try:
        return await _agent(request).resume(ticket_id)
    except TicketStateError as exc:
        raise ApiError(409, "invalid_state", str(exc)) from exc

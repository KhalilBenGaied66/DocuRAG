"""Typed contracts: what the model must produce, what a supervisor decides, what the API returns."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

# Equipment tags have 3+ digits (P-200-07, P-2041); alarm codes have 2 (E-12, E-41) and must
# never be taken for equipment: "Alarme E-12 sur la P-200-07" is about P-200-07.
EQUIPMENT_TAG = r"^[A-Z]{1,4}-\d{3,5}(?:-\d{1,3})?$"
TAG_IN_TEXT = re.compile(r"\b[A-Z]{1,4}-\d{3,5}(?:-\d{1,3})?\b")
PART_REF = r"^[A-Za-z0-9][A-Za-z0-9.\-]{1,39}$"  # GM-45, 6309-2Z, JT-80x3
TICKET_ID = r"^[A-Za-z0-9._-]{1,64}$"


class Intent(BaseModel):
    """Classe la demande d'un technicien de maintenance."""

    kind: Literal["question", "incident"] = Field(
        description="question : recherche d'une information dans la documentation ; "
        "incident : panne, alarme ou anomalie sur un équipement, à diagnostiquer"
    )
    equipment_tag: str | None = Field(
        default=None, description="Repère de l'équipement s'il est mentionné, ex. P-200-07"
    )


class PartLine(BaseModel):
    ref: str = Field(max_length=40, description="Référence exacte, telle que dans la documentation")
    quantity: int = Field(default=1, ge=1, le=50)


class WorkOrderDraft(BaseModel):
    """Proposition d'ordre de travail, soumise à la validation d'un superviseur."""

    equipment_tag: str = Field(max_length=32, description="Repère de l'équipement, ex. P-200-07")
    title: str = Field(max_length=120, description="Intitulé court de l'intervention")
    diagnosis: str = Field(max_length=2000, description="Diagnostic argumenté, citant les sources doc:…")
    probable_cause: str = Field(max_length=500)
    actions: list[str] = Field(min_length=1, max_length=15, description="Étapes, dans l'ordre")
    parts: list[PartLine] = Field(default_factory=list, max_length=10)
    priority: Literal["P1", "P2", "P3"] = Field(
        description="P1 : arrêt ou risque sécurité ; P2 : dégradation à traiter sous 48 h ; "
        "P3 : planifiable"
    )
    safety_notes: list[str] = Field(default_factory=list, max_length=10)
    sources: list[str] = Field(default_factory=list, description="Identifiants doc:… utilisés")
    confidence: Literal["low", "medium", "high"]


class Decision(BaseModel):
    approved: bool
    approver: str
    feedback: str | None = Field(default=None, max_length=2000)


TicketStatus = Literal[
    "answered", "awaiting_approval", "work_order_created", "rejected", "stalled", "unknown"
]


class TicketView(BaseModel):
    ticket_id: str
    status: TicketStatus
    requested_by: str | None = None
    intent: str | None = None
    equipment_tag: str | None = None
    answer: dict | None = None
    proposal: dict | None = None
    issues: list[str] = Field(default_factory=list)
    incomplete: bool = False
    decision: dict | None = None
    work_order_id: str | None = None
    steps: int = 0
    tokens: int = 0
    tool_calls: list[dict] = Field(default_factory=list)


class TicketStateError(RuntimeError):
    """The ticket is not in a state that allows this action."""


class AgentPermissionError(PermissionError):
    """The caller may not perform this action on this ticket."""

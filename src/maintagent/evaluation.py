"""Scenario evaluation: the agent is judged on its trajectory, not only on its final text.

Each scenario states the expected intent, equipment and tools. Measured: task success,
intent and equipment accuracy, tool recall, steps within bounds, writes gated by approval,
idempotent creation, tokens and latency.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from docurag.evaluation.metrics import mean, percentile
from docurag.models import User
from maintagent.service import MaintAgentService


class Scenario(BaseModel):
    id: str
    report: str
    expected_intent: Literal["question", "incident"]
    expected_equipment: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    expected_parts: list[str] = Field(default_factory=list)
    note: str | None = None


def load_scenarios(path: Path) -> list[Scenario]:
    return [
        Scenario.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def score(scenario: Scenario, view: Any) -> dict[str, Any]:
    called = {c["tool"] for c in view.tool_calls}
    proposed_parts = {p["ref"] for p in (view.proposal or {}).get("parts", [])}
    if scenario.expected_intent == "incident":
        success = view.status == "awaiting_approval" and view.proposal is not None and not view.incomplete
    else:
        success = view.status == "answered" and not (view.answer or {}).get("abstained", True)
    return {
        "id": scenario.id,
        "status": view.status,
        "task_success": success,
        "intent_correct": view.intent == scenario.expected_intent,
        "equipment_correct": (
            view.equipment_tag == scenario.expected_equipment if scenario.expected_equipment else None
        ),
        "tool_recall": (
            len(set(scenario.expected_tools) & called) / len(scenario.expected_tools)
            if scenario.expected_tools else None
        ),
        "within_steps": view.steps <= scenario.max_steps if scenario.max_steps else None,
        "parts_recall": (
            len(set(scenario.expected_parts) & proposed_parts) / len(scenario.expected_parts)
            if scenario.expected_parts else None
        ),
        # By design no order exists before a decision; measured anyway, it must stay 1.0.
        "write_gated": view.work_order_id is None if scenario.expected_intent == "incident" else None,
        "issues": len(view.issues),
        "steps": view.steps,
        "tokens": view.tokens,
    }


async def run_scenarios(
    agent: MaintAgentService,
    scenarios: list[Scenario],
    *,
    requester: User,
    supervisor: User | None = None,
    label: str = "run",
) -> dict[str, Any]:
    run_id = uuid4().hex[:6]
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        ticket_id = f"eval-{run_id}-{scenario.id}"
        started = perf_counter()
        view = await agent.submit(ticket_id, scenario.report, requester)
        row = score(scenario, view) | {"latency_ms": round((perf_counter() - started) * 1000)}
        if supervisor is not None and view.status == "awaiting_approval":
            approved = await agent.decide(ticket_id, supervisor, approved=True)
            again = await agent.submit(ticket_id, scenario.report, requester)
            row["work_order_created"] = approved.status == "work_order_created"
            row["idempotent"] = (
                approved.work_order_id is not None and again.work_order_id == approved.work_order_id
            )
        rows.append(row)

    metrics = ["task_success", "intent_correct", "equipment_correct", "tool_recall",
               "within_steps", "parts_recall", "write_gated", "work_order_created", "idempotent",
               "steps", "tokens"]
    summary = {m: mean(r.get(m) for r in rows) for m in metrics}
    summary = {k: v for k, v in summary.items() if v is not None}
    summary["latency_ms_p50"] = percentile([r["latency_ms"] for r in rows], 50)
    summary["latency_ms_p95"] = percentile([r["latency_ms"] for r in rows], 95)
    return {"label": label, "n": len(rows), "summary": summary, "rows": rows}

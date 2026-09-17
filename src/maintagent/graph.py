"""MaintAgent graph.

    START → classify ─question→ rag_answer → END
                     └incident→ reason ⇄ tools      (bounded: steps, tokens)
                                  └→ propose → approval (interrupt) ─approved→ create_work_order → END
                                                    ├ rejected + feedback → reason
                                                    └ rejected → END

Only the diagnosis is an open loop; routing, proposal and approval are fixed steps. The only
write in the system sits behind a human decision and an idempotency key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from docurag.models import User
from maintagent.config import AgentSettings
from maintagent.models import EQUIPMENT_TAG, TAG_IN_TEXT, Decision, Intent, WorkOrderDraft
from maintagent.prompts import (
    AGENT_PROMPT_VERSION,
    CLASSIFY_SYSTEM,
    PROPOSE_SYSTEM,
    REASON_SYSTEM,
    feedback_message,
    propose_instruction,
)
from maintagent.tools import (
    BusinessLike,
    ErpLike,
    RetrieverLike,
    ToolContext,
    run_tool_call,
    tool_schemas,
)

log = logging.getLogger(__name__)

_INCIDENT_WORDS = re.compile(
    r"alarme|panne|fuite|vibr|bruit|défaut|arrêt|échauff|surchauffe|\bE-\d{2}\b", re.I
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ticket_id: str
    user_id: str
    groups: list[str]
    report: str
    prompt_version: str
    intent: str | None
    equipment_tag: str | None
    steps: int
    tokens: int
    rejections: int
    evidence: Annotated[list[dict], operator.add]
    tool_log: Annotated[list[dict], operator.add]
    proposal: dict | None
    issues: list[str]
    incomplete: bool
    decision: dict | None
    answer: dict | None
    work_order_id: str | None


def initial_state(ticket_id: str, report: str, user: User) -> AgentState:
    return AgentState(
        messages=[HumanMessage(report)], ticket_id=ticket_id, user_id=user.id,
        groups=list(user.groups), report=report, prompt_version=AGENT_PROMPT_VERSION,
        intent=None, equipment_tag=None, steps=0, tokens=0, rejections=0, evidence=[],
        tool_log=[], proposal=None, issues=[], incomplete=False, decision=None, answer=None,
        work_order_id=None,
    )


class RagLike(Protocol):
    async def answer(self, question: str, user: User, **kwargs: Any) -> Any: ...


class GmaoLike(BusinessLike, Protocol):
    async def create_work_order(
        self, *, ticket_id: str, draft: WorkOrderDraft, requested_by: str, approved_by: str,
        sources: list[dict[str, Any]],
    ) -> str: ...


@dataclass(slots=True)
class AgentDeps:
    chat: BaseChatModel  # diagnosis and proposal
    fast_chat: BaseChatModel  # classification
    retriever: RetrieverLike
    rag: RagLike
    business: GmaoLike
    erp: ErpLike
    settings: AgentSettings


def tokens_of(message: BaseMessage | None) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))


async def structured_call(
    model: BaseChatModel, schema: type[BaseModel], messages: list[BaseMessage]
) -> tuple[BaseModel | None, int]:
    """Structured output with one corrective retry. Returns (None, tokens) instead of raising:
    small local models do produce invalid JSON, and a ticket must not fail because of it."""
    runnable = model.with_structured_output(schema, include_raw=True)
    tokens = 0
    for _ in range(2):
        out = await runnable.ainvoke(messages)
        tokens += tokens_of(out.get("raw"))
        if out.get("parsed") is not None:
            return out["parsed"], tokens
        problem = out.get("parsing_error") or "aucune sortie structurée"
        messages = [*messages, HumanMessage(f"Sortie invalide ({str(problem)[:300]}). "
                                            "Réponds de nouveau en respectant exactement le schéma.")]
    return None, tokens


def heuristic_intent(report: str) -> Intent:
    tag = TAG_IN_TEXT.search(report)
    kind = "incident" if tag and _INCIDENT_WORDS.search(report) else "question"
    return Intent(kind=kind, equipment_tag=tag.group(0) if tag else None)


def _last_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return ""


def fallback_draft(state: AgentState, history: list[BaseMessage]) -> WorkOrderDraft:
    return WorkOrderDraft(
        equipment_tag=state["equipment_tag"] or "INCONNU",
        title="Diagnostic à compléter",
        diagnosis=(_last_text(history) or "Le modèle n'a pas produit de diagnostic exploitable.")[:2000],
        probable_cause="Indéterminée",
        actions=["Diagnostic manuel par un technicien qualifié"],
        priority="P2",
        confidence="low",
    )


async def check_draft(
    draft: WorkOrderDraft, state: AgentState, deps: AgentDeps
) -> tuple[WorkOrderDraft, list[str]]:
    """Deterministic checks on what the model proposed, before any human sees it."""
    issues: list[str] = []
    known = {e["source_id"] for e in state["evidence"]}
    unknown = [s for s in draft.sources if s not in known]
    if unknown:
        issues.append(f"Sources inconnues retirées : {', '.join(unknown)}")
        draft = draft.model_copy(update={"sources": [s for s in draft.sources if s in known]})
    if not draft.sources:
        issues.append("Aucune source documentaire citée")

    reported = state["equipment_tag"]
    if reported and draft.equipment_tag != reported:
        issues.append(f"Repère proposé {draft.equipment_tag} différent du repère signalé {reported}")
    if await deps.business.equipment(draft.equipment_tag) is None:
        issues.append(f"Équipement {draft.equipment_tag} inconnu du référentiel")

    if draft.parts:
        try:
            stock = await deps.erp.stock([p.ref for p in draft.parts])
        except Exception:
            log.warning("ERP unavailable while checking ticket %s", state["ticket_id"])
            issues.append("Stock non vérifié : ERP indisponible")
        else:
            for part in draft.parts:
                item = stock.get(part.ref)
                if item is None:
                    issues.append(f"Pièce {part.ref} inconnue de l'ERP")
                elif item["available"] < part.quantity:
                    issues.append(
                        f"Pièce {part.ref} : {item['available']} en stock pour {part.quantity} "
                        f"demandée(s), délai {item['lead_time_days']} j"
                    )
    return draft, issues


class Nodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.d = deps
        self.tools = tool_schemas()

    async def classify(self, state: AgentState) -> dict:
        messages = [SystemMessage(CLASSIFY_SYSTEM), HumanMessage(state["report"])]
        intent, tokens = await structured_call(self.d.fast_chat, Intent, messages)
        if intent is None:
            log.warning("ticket %s: classification fell back to rules", state["ticket_id"])
            intent = heuristic_intent(state["report"])
        tag = intent.equipment_tag
        if not tag or not re.fullmatch(EQUIPMENT_TAG, tag):
            found = TAG_IN_TEXT.search(state["report"])
            tag = found.group(0) if found else None
        return {"intent": intent.kind, "equipment_tag": tag, "tokens": state["tokens"] + tokens}

    def route_intent(self, state: AgentState) -> str:
        return "rag_answer" if state["intent"] == "question" else "reason"

    async def rag_answer(self, state: AgentState) -> dict:
        user = User(id=state["user_id"], groups=state["groups"])
        answer = await self.d.rag.answer(state["report"], user, request_id=state["ticket_id"])
        return {
            "answer": answer.model_dump(mode="json", exclude={"contexts"}),
            "messages": [AIMessage(answer.text)],
            "tokens": state["tokens"] + answer.usage.input_tokens + answer.usage.output_tokens,
        }

    def _exhausted(self, state: AgentState) -> bool:
        s = self.d.settings
        return state["steps"] >= s.max_steps or state["tokens"] >= s.token_budget

    async def reason(self, state: AgentState) -> dict:
        system = REASON_SYSTEM.format(max_steps=self.d.settings.max_steps)
        if state["equipment_tag"]:
            system += f"\n\nÉquipement signalé : {state['equipment_tag']}."
        response = await self.d.chat.bind_tools(self.tools).ainvoke(
            [SystemMessage(system), *state["messages"]]
        )
        return {
            "messages": [response],
            "steps": state["steps"] + 1,
            "tokens": state["tokens"] + tokens_of(response),
        }

    def after_reason(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and not self._exhausted(state):
            return "tools"
        return "propose"

    async def run_tools(self, state: AgentState) -> dict:
        s = self.d.settings
        calls = list(state["messages"][-1].tool_calls)
        ctx = ToolContext(
            user_id=state["user_id"], groups=state["groups"], ticket_id=state["ticket_id"],
            retriever=self.d.retriever, business=self.d.business, erp=self.d.erp,
        )
        allowed = calls[: s.max_parallel_tool_calls]
        outcomes = await asyncio.gather(
            *(run_tool_call(call, ctx, timeout_s=s.tool_timeout_s) for call in allowed)
        )

        messages: list[ToolMessage] = []
        new_evidence: list[dict] = []
        log_rows: list[dict] = []
        known = {e["source_id"] for e in state["evidence"]}
        for i, (call, (result, ms)) in enumerate(zip(allowed, outcomes, strict=True)):
            messages.append(ToolMessage(
                content=json.dumps(result.content, ensure_ascii=False, default=str),
                tool_call_id=call.get("id") or f"call-{state['steps']}-{i}",
                name=call["name"], status="error" if result.error else "success",
            ))
            for item in result.evidence:
                if item["source_id"] not in known:
                    known.add(item["source_id"])
                    new_evidence.append(item)
            log_rows.append({"step": state["steps"], "tool": call["name"], "args": call.get("args"),
                             "ok": not result.error, "ms": ms})
        # Every tool call must get an answer, or the next model call is rejected by the API.
        for i, call in enumerate(calls[len(allowed):], start=len(allowed)):
            messages.append(ToolMessage(
                content=json.dumps({"error": "too_many_parallel_calls", "max": s.max_parallel_tool_calls}),
                tool_call_id=call.get("id") or f"call-{state['steps']}-{i}",
                name=call["name"], status="error",
            ))
            log_rows.append({"step": state["steps"], "tool": call["name"], "args": call.get("args"),
                             "ok": False, "ms": 0.0})
        return {"messages": messages, "evidence": new_evidence, "tool_log": log_rows}

    async def propose(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        dangling = isinstance(last, AIMessage) and bool(last.tool_calls)
        incomplete = dangling or state["tokens"] >= self.d.settings.token_budget
        # Tool calls that were never executed can't be sent back to the model.
        history = list(state["messages"][:-1] if dangling else state["messages"])
        evidence_ids = [e["source_id"] for e in state["evidence"]]
        messages = [
            SystemMessage(PROPOSE_SYSTEM), *history,
            HumanMessage(propose_instruction(evidence_ids, incomplete)),
        ]
        draft, tokens = await structured_call(self.d.chat, WorkOrderDraft, messages)
        fallback = draft is None
        if fallback:
            draft = fallback_draft(state, history)
            incomplete = True
        draft, issues = await check_draft(draft, state, self.d)
        if fallback:
            issues.insert(0, "Le modèle n'a pas produit de proposition valide : proposition minimale")
        return {
            "proposal": draft.model_dump(), "issues": issues, "incomplete": incomplete,
            "tokens": state["tokens"] + tokens,
        }

    def approval(
        self, state: AgentState
    ) -> Command[Literal["create_work_order", "reason", "__end__"]]:
        # Code before interrupt() runs again when the graph resumes: keep it side-effect free.
        raw = interrupt({
            "ticket_id": state["ticket_id"], "proposal": state["proposal"],
            "issues": state["issues"], "incomplete": state["incomplete"],
        })
        decision = Decision.model_validate(raw)
        update: dict[str, Any] = {"decision": decision.model_dump()}
        if decision.approved:
            return Command(goto="create_work_order", update=update)
        if decision.feedback and state["rejections"] < self.d.settings.max_rejections:
            return Command(goto="reason", update={
                **update, "rejections": state["rejections"] + 1, "steps": 0, "proposal": None,
                "issues": [], "incomplete": False,
                "messages": [HumanMessage(feedback_message(decision.approver, decision.feedback))],
            })
        return Command(goto=END, update=update)

    async def create_work_order(self, state: AgentState) -> dict:
        draft = WorkOrderDraft.model_validate(state["proposal"])
        cited = set(draft.sources)
        work_order_id = await self.d.business.create_work_order(
            ticket_id=state["ticket_id"], draft=draft, requested_by=state["user_id"],
            approved_by=state["decision"]["approver"],
            sources=[e for e in state["evidence"] if e["source_id"] in cited],
        )
        return {"work_order_id": work_order_id}


def build_graph(deps: AgentDeps, checkpointer: Any) -> Any:
    nodes = Nodes(deps)
    graph = StateGraph(AgentState)
    graph.add_node("classify", nodes.classify)
    graph.add_node("rag_answer", nodes.rag_answer)
    graph.add_node("reason", nodes.reason)
    graph.add_node("tools", nodes.run_tools)
    graph.add_node("propose", nodes.propose)
    graph.add_node("approval", nodes.approval, destinations=("create_work_order", "reason", END))
    graph.add_node("create_work_order", nodes.create_work_order)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify", nodes.route_intent, {"rag_answer": "rag_answer", "reason": "reason"}
    )
    graph.add_conditional_edges("reason", nodes.after_reason, {"tools": "tools", "propose": "propose"})
    graph.add_edge("tools", "reason")
    graph.add_edge("propose", "approval")
    graph.add_edge("rag_answer", END)
    graph.add_edge("create_work_order", END)
    return graph.compile(checkpointer=checkpointer)

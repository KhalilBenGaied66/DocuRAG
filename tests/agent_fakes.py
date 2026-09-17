"""Scripted models and in-memory fakes for the agent tests (no LLM, no database)."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from docurag.models import Answer, Usage, User
from docurag.retrieval.retriever import RetrievalResult
from maintagent.config import AgentSettings
from maintagent.graph import AgentDeps, build_graph
from maintagent.service import MaintAgentService
from tests.conftest import make_hit

TECH = User(id="tech-42", groups=["maintenance"])
SUPERVISOR = User(id="sup-7", groups=["maintenance", "maintenance-supervisor"])
REPORT = "Pompe P-200-07 : vibrations fortes et alarme E-41 depuis ce matin."

DRAFT = {
    "equipment_tag": "P-200-07",
    "title": "Réalignement moteur-pompe",
    "diagnosis": "Alarme E-41 [doc:7] ; désalignement déjà constaté sur cet équipement.",
    "probable_cause": "Désalignement après remplacement du moteur",
    "actions": ["Consigner la pompe", "Contrôler l'alignement au laser"],
    "parts": [{"ref": "6309-2Z", "quantity": 1}],
    "priority": "P2",
    "safety_notes": ["Consignation selon § 2.1"],
    "sources": ["doc:7", "doc:999"],
    "confidence": "medium",
}


class ScriptedChat(BaseChatModel):
    """Plays back scripted responses (an Exception in the script is raised) and records calls."""

    script: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> ScriptedChat:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("unexpected model call")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai_tools(*calls: dict, tokens: int = 0) -> AIMessage:
    usage = {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens} if tokens else None
    return AIMessage(content="", tool_calls=list(calls), usage_metadata=usage)


def intent(kind: str, tag: str | None = None) -> AIMessage:
    return ai_tools(tool_call("Intent", {"kind": kind, "equipment_tag": tag}, "i1"))


def draft(**overrides: Any) -> AIMessage:
    return ai_tools(tool_call("WorkOrderDraft", {**DRAFT, **overrides}, "d1"))


def incident_script() -> list[Any]:
    return [
        ai_tools(
            tool_call("search_docs", {"query": "alarme E-41 vibration pompe P-200"}, "t1"),
            tool_call("equipment_history", {"equipment_tag": "P-200-07"}, "t2"),
            tokens=300,
        ),
        ai_tools(tool_call("parts_stock", {"part_refs": ["6309-2Z"]}, "t3"), tokens=200),
        AIMessage("Désalignement probable, récurrent sur cet équipement [doc:7]."),
        draft(),
    ]


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def retrieve(self, query: str, groups: list[str], **kwargs: Any) -> RetrievalResult:
        self.calls.append((query, list(groups)))
        hits = []
        if "maintenance" in groups:
            hits = [make_hit(
                7, section_path="5 Dépannage > 5.1 Codes alarme",
                body="E-41 : vibration > 7,1 mm/s. Contrôler l'alignement, puis les roulements 6309-2Z.",
            )]
        return RetrievalResult(hits=hits, candidates=hits)


class FakeRag:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def answer(self, question: str, user: User, **kwargs: Any) -> Answer:
        self.calls.append((question, user.groups))
        return Answer(
            request_id=kwargs.get("request_id") or "r", text="49 N·m [S1]", abstained=False,
            standalone_question=question, prompt_version="answer-v1",
            usage=Usage(input_tokens=120, output_tokens=8),
        )


class FakeBusiness:
    def __init__(self) -> None:
        self.orders: dict[str, str] = {}
        self.create_calls = 0

    async def equipment(self, tag: str) -> dict | None:
        return {"tag": tag, "site": "Lyon", "model": "P-200"} if tag.startswith("P-200-") else None

    async def interventions(self, tag: str, since_days: int, limit: int = 20) -> list[dict]:
        return [{"ot_id": "OT-2025-0412", "date": "2025-10-14", "fault_code": "E-41",
                 "diagnosis": "Désalignement de 0,18 mm"}]

    async def create_work_order(self, *, ticket_id: str, draft: Any, requested_by: str,
                                approved_by: str, sources: list[dict]) -> str:
        self.create_calls += 1
        return self.orders.setdefault(ticket_id, f"WO-{len(self.orders) + 1:04d}")


class FakeErp:
    STOCK = {
        "6309-2Z": {"ref": "6309-2Z", "available": 0, "lead_time_days": 7},
        "GM-45": {"ref": "GM-45", "available": 2, "lead_time_days": 10},
    }

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[list[str]] = []

    async def stock(self, refs: list[str]) -> dict[str, dict | None]:
        self.calls.append(list(refs))
        if self.delay:
            await asyncio.sleep(self.delay)
        return {r: self.STOCK.get(r) for r in refs}


class Harness:
    def __init__(self, chat_script: list[Any], fast_script: list[Any], *, erp_delay: float = 0.0,
                 checkpointer: Any = None, **settings: Any) -> None:
        self.chat = ScriptedChat(script=list(chat_script))
        self.fast = ScriptedChat(script=list(fast_script))
        self.retriever, self.rag = FakeRetriever(), FakeRag()
        self.business, self.erp = FakeBusiness(), FakeErp(erp_delay)
        self.settings = AgentSettings(_env_file=None, **settings)
        self.deps = AgentDeps(
            chat=self.chat, fast_chat=self.fast, retriever=self.retriever, rag=self.rag,
            business=self.business, erp=self.erp, settings=self.settings,
        )
        self.checkpointer = checkpointer or InMemorySaver()
        self.agent = MaintAgentService(build_graph(self.deps, self.checkpointer), self.settings)

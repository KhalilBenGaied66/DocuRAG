"""Contract tests for the Ollama integration, without Ollama.

A fake server speaks Ollama's HTTP API — streamed NDJSON chat responses, tool calls, JSON-schema
structured output, /api/embed — and the *real* langchain-ollama clients built by
`docurag.providers` talk to it through both pipelines. This checks what unit tests with fake
chat models cannot: message conversion (tool calls, tool results, system prompts), request
options (context window, temperature), structured-output parsing, token usage, streaming.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

pytest.importorskip("langchain_ollama")
from ollama import AsyncClient  # noqa: E402

from docurag.config import Settings  # noqa: E402
from docurag.models import User  # noqa: E402
from docurag.providers import build_chat_model, build_embeddings  # noqa: E402
from docurag.service import RAGService  # noqa: E402
from maintagent.config import AgentSettings  # noqa: E402
from maintagent.graph import AgentDeps, build_graph  # noqa: E402
from maintagent.service import MaintAgentService  # noqa: E402
from tests.agent_fakes import REPORT, TECH, FakeBusiness, FakeErp, FakeRag, FakeRetriever  # noqa: E402
from tests.conftest import FakeRepo, make_hit, section  # noqa: E402
from tests.conftest import FakeRetriever as RagRetriever

PROMPT_TOKENS, COMPLETION_TOKENS = 120, 15


def ollama_settings() -> Settings:
    return Settings(
        _env_file=None,
        env="test",
        provider="ollama",
        chat_model="mistral-nemo",
        fast_model="mistral-nemo",
        embedding_model="bge-m3",
        embedding_dimensions=1024,
        chat_temperature=0,
        context_budget_tokens=6000,
        ollama_num_ctx=8192,
        ollama_base_url="http://ollama.test",
    )


class FakeOllama:
    """Scripted replies in Ollama's wire format; records every request body."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.chats: list[dict[str, Any]] = []
        self.embeds: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            self.embeds.append(body)
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return httpx.Response(
                200, json={"model": body["model"], "embeddings": [[0.01] * 1024] * len(inputs)}
            )
        assert request.url.path == "/api/chat", request.url.path
        self.chats.append(body)
        reply = self.replies.pop(0)
        lines = stream(**reply) if isinstance(reply, dict) else stream(content=reply)
        body_bytes = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode()
        return httpx.Response(200, content=body_bytes, headers={"content-type": "application/x-ndjson"})


def stream(content: str = "", tool_calls: list[tuple[str, dict]] | None = None) -> list[dict]:
    base = {"model": "mistral-nemo", "created_at": "2026-09-16T12:00:00Z"}
    pieces = [content[i : i + 7] for i in range(0, len(content), 7)]
    lines = [{**base, "message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
    if tool_calls:
        calls = [{"function": {"name": name, "arguments": args}} for name, args in tool_calls]
        lines.append(
            {**base, "message": {"role": "assistant", "content": "", "tool_calls": calls}, "done": False}
        )
    lines.append(
        {
            **base,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": PROMPT_TOKENS,
            "eval_count": COMPLETION_TOKENS,
            "total_duration": 1,
            "load_duration": 1,
            "prompt_eval_duration": 1,
            "eval_duration": 1,
        }
    )
    return lines


def wire(model: Any, fake: FakeOllama) -> Any:
    """Point a client built by docurag.providers at the fake server (same code path otherwise)."""
    model._async_client = AsyncClient(host="http://ollama.test", transport=httpx.MockTransport(fake.handle))
    return model


# ------------------------------------------------------------------ DocuRAG


async def test_rag_answer_streams_through_chat_ollama():
    settings = ollama_settings()
    answer_text = "Le couple de serrage du carter est de 49 N·m [S1]."
    fake = FakeOllama([answer_text])
    chat = wire(build_chat_model(settings), fake)
    service = RAGService(
        settings, FakeRepo({10: section("Carter : M10, 49 N·m.")}), RagRetriever([make_hit()]), chat, chat
    )

    events = [e async for e in service.stream("Couple du carter ?", User(id="t", groups=["maintenance"]))]

    tokens = "".join(e.data for e in events if e.type == "token")
    answer = events[-1].data
    assert tokens == answer_text == answer.text and not answer.abstained
    assert [c.source_id for c in answer.citations] == ["S1"]
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (PROMPT_TOKENS, COMPLETION_TOKENS)

    request = fake.chats[0]
    assert request["model"] == "mistral-nemo" and request["stream"] is True
    assert request["options"]["num_ctx"] == 8192 and request["options"]["temperature"] == 0
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert '<source id="S1"' in request["messages"][1]["content"]


async def test_no_answer_sentinel_split_across_stream_chunks():
    settings = ollama_settings()
    fake = FakeOllama(["NO_ANSWER"])  # streamed as "NO_ANSW" + "ER"
    chat = wire(build_chat_model(settings), fake)
    service = RAGService(settings, FakeRepo({10: section("x")}), RagRetriever([make_hit()]), chat, chat)

    events = [e async for e in service.stream("Prix d'une garniture ?", User(id="t", groups=["maintenance"]))]

    assert events[-1].data.abstained
    assert "NO_ANSWER" not in "".join(e.data for e in events if e.type == "token")


async def test_embeddings_through_ollama_embed_endpoint():
    settings = ollama_settings()
    fake = FakeOllama([])
    embeddings = wire(build_embeddings(settings), fake)

    vectors = await embeddings.aembed_documents(["Manuel P-200 > 4 Couples", "Alarme E-41"])

    assert len(vectors) == 2 and all(len(v) == settings.embedding_dimensions for v in vectors)
    assert fake.embeds[0]["model"] == "bge-m3" and len(fake.embeds[0]["input"]) == 2


# ------------------------------------------------------------------ MaintAgent

DRAFT = {
    "equipment_tag": "P-200-07",
    "title": "Réalignement moteur-pompe",
    "diagnosis": "Alarme E-41 [doc:7], désalignement déjà constaté (OT-2025-0412).",
    "probable_cause": "Désalignement",
    "actions": ["Consigner la pompe", "Aligner au laser"],
    "parts": [{"ref": "GM-45", "quantity": 1}],
    "priority": "P2",
    "safety_notes": [],
    "sources": ["doc:7"],
    "confidence": "medium",
}


async def test_agent_incident_through_chat_ollama():
    settings = ollama_settings()
    fake = FakeOllama(
        [
            "pas du JSON",  # classify: invalid …
            json.dumps({"kind": "incident", "equipment_tag": "P-200-07"}),  # … corrected on retry
            {
                "tool_calls": [
                    ("search_docs", {"query": "alarme E-41 vibration P-200"}),
                    ("equipment_history", {"equipment_tag": "P-200-07"}),
                ]
            },
            "Désalignement probable, déjà constaté sur cet équipement [doc:7].",
            json.dumps(DRAFT, ensure_ascii=False),
        ]
    )
    chat = wire(build_chat_model(settings), fake)
    retriever = FakeRetriever()
    deps = AgentDeps(
        chat=chat,
        fast_chat=chat,
        retriever=retriever,
        rag=FakeRag(),
        business=FakeBusiness(),
        erp=FakeErp(),
        settings=AgentSettings(_env_file=None),
    )
    from langgraph.checkpoint.memory import InMemorySaver

    agent = MaintAgentService(build_graph(deps, InMemorySaver()), deps.settings)

    view = await agent.submit("T-OLLAMA-1", REPORT, TECH)

    assert view.status == "awaiting_approval", view
    assert (view.intent, view.equipment_tag) == ("incident", "P-200-07")
    assert [c["tool"] for c in view.tool_calls] == ["search_docs", "equipment_history"]
    assert all(c["ok"] for c in view.tool_calls)
    assert view.proposal["parts"] == [{"ref": "GM-45", "quantity": 1}]
    assert view.proposal["sources"] == ["doc:7"] and not view.incomplete
    assert view.tokens == 5 * (PROMPT_TOKENS + COMPLETION_TOKENS)
    assert retriever.calls == [("alarme E-41 vibration P-200", ["maintenance"])]

    classify, retry, reason, reason_after_tools, propose = fake.chats
    assert classify["format"]["properties"]["kind"]["enum"] == ["question", "incident"]
    assert "Sortie invalide" in retry["messages"][-1]["content"]
    assert {t["function"]["name"] for t in reason["tools"]} == {
        "search_docs",
        "equipment_history",
        "parts_stock",
    }
    assert "format" not in reason or not reason["format"]

    roles = [m["role"] for m in reason_after_tools["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    assistant = reason_after_tools["messages"][2]
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["search_docs", "equipment_history"]
    tool_results = [json.loads(m["content"]) for m in reason_after_tools["messages"][3:]]
    assert tool_results[0]["results"][0]["source_id"] == "doc:7"
    assert tool_results[1]["interventions"][0]["ot_id"] == "OT-2025-0412"

    assert "parts" in propose["format"]["properties"]
    assert (
        propose["messages"][0]["role"] == "system" and "ordre de travail" in propose["messages"][0]["content"]
    )

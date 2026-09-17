"""Question → standalone query → retrieval → context → streamed, cited answer, or abstention."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from docurag.config import Settings
from docurag.db.repository import Repository, SearchMode
from docurag.generation.context import assemble_context, render_sources, verify_citations
from docurag.generation.prompts import (
    ANSWER_PROMPT_VERSION,
    NO_ANSWER,
    answer_messages,
    rewrite_messages,
)
from docurag.models import Answer, ChatTurn, Citation, SearchHit, Usage, User, sha256
from docurag.providers import content_text
from docurag.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

EventType = Literal["sources", "token", "done"]


@dataclass(slots=True)
class Event:
    type: EventType
    data: Any


def _ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 1)


def usage_of(message: BaseMessage) -> Usage:
    meta = getattr(message, "usage_metadata", None) or {}
    details = meta.get("input_token_details") or {}
    return Usage(
        input_tokens=meta.get("input_tokens") or 0,
        output_tokens=meta.get("output_tokens") or 0,
        cached_tokens=details.get("cache_read") or 0,
    )


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cached_tokens=a.cached_tokens + b.cached_tokens,
    )


def _hit_citation(index: int, hit: SearchHit) -> Citation:
    return Citation(
        source_id=f"S{index}", document_id=hit.document_id, title=hit.title,
        source_uri=hit.source_uri, section_path=hit.section_path,
        page_start=hit.page_start, page_end=hit.page_end,
    )


class SentinelGate:
    """Holds back the first characters of a stream until we know the model is not
    answering NO_ANSWER, so the sentinel never reaches the user."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pending = ""
        self.decided = False
        self.abstained = False

    def feed(self, piece: str) -> list[str]:
        self.parts.append(piece)
        if self.decided:
            return [] if self.abstained else [piece]
        self.pending += piece
        head = self.pending.lstrip()
        if len(head) < len(NO_ANSWER) and NO_ANSWER.startswith(head):
            return []  # could still become the sentinel
        self.decided = True
        self.abstained = head.startswith(NO_ANSWER)
        return [] if self.abstained else [self.pending]

    def close(self) -> list[str]:
        if self.decided:
            return []
        self.decided = True
        head = self.pending.strip()
        self.abstained = not head or NO_ANSWER.startswith(head)
        return [] if self.abstained else [self.pending]

    @property
    def text(self) -> str:
        return "".join(self.parts)


class RAGService:
    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        retriever: Retriever,
        chat: BaseChatModel,
        fast_chat: BaseChatModel,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.retriever = retriever
        self.chat = chat
        self.fast_chat = fast_chat

    async def answer(
        self,
        question: str,
        user: User,
        *,
        history: list[ChatTurn] | None = None,
        request_id: str | None = None,
        mode: SearchMode = "hybrid",
        include_context: bool = False,
    ) -> Answer:
        final: Answer | None = None
        async for event in self.stream(
            question, user, history=history, request_id=request_id, mode=mode,
            include_context=include_context,
        ):
            if event.type == "done":
                final = event.data
        if final is None:
            raise RuntimeError("answer stream ended without a final event")
        return final

    async def stream(
        self,
        question: str,
        user: User,
        *,
        history: list[ChatTurn] | None = None,
        request_id: str | None = None,
        mode: SearchMode = "hybrid",
        include_context: bool = False,
    ) -> AsyncIterator[Event]:
        request_id = request_id or uuid4().hex
        started = perf_counter()
        timings: dict[str, float] = {}
        usage = Usage()

        standalone = question
        if history:
            t = perf_counter()
            standalone, rewrite_usage = await self._rewrite(question, history)
            usage = _add(usage, rewrite_usage)
            timings["rewrite_ms"] = _ms(t)

        result = await self.retriever.retrieve(standalone, user.groups, mode=mode)
        timings.update(result.timings_ms)
        hits = result.hits
        top_score = hits[0].score if hits else None
        abstained = not hits or self._below_threshold(top_score)

        sources = []
        if not abstained:
            t = perf_counter()
            sources = await assemble_context(
                hits, self.repo,
                parent_max_tokens=self.settings.parent_max_tokens,
                budget_tokens=self.settings.context_budget_tokens,
            )
            timings["context_ms"] = _ms(t)
        yield Event("sources", [s.citation().model_dump(mode="json") for s in sources])

        text = ""
        if not abstained:
            t = perf_counter()
            gate = SentinelGate()
            messages = answer_messages(render_sources(sources), standalone)
            async for chunk in self.chat.astream(messages):
                usage = _add(usage, usage_of(chunk))
                piece = content_text(chunk)
                if not piece:
                    continue
                timings.setdefault("first_token_ms", _ms(started))
                for out in gate.feed(piece):
                    yield Event("token", out)
            for out in gate.close():
                yield Event("token", out)
            timings["generation_ms"] = _ms(t)
            abstained, text = gate.abstained, gate.text

        if abstained:
            text, citations, invalid = self.settings.abstain_message, [], []
            yield Event("token", text)
            shown = [_hit_citation(i, h) for i, h in enumerate(hits[:3], start=1)]
        else:
            text, citations, invalid = verify_citations(text, sources)
            shown = [s.citation() for s in sources]
            if invalid:
                log.warning("request %s: model cited unknown sources %s", request_id, invalid)

        timings["total_ms"] = _ms(started)
        answer = Answer(
            request_id=request_id,
            text=text,
            abstained=abstained,
            citations=citations,
            sources=shown,
            invalid_citations=invalid,
            top_score=top_score,
            standalone_question=standalone,
            prompt_version=ANSWER_PROMPT_VERSION,
            usage=usage,
            timings_ms=timings,
            contexts=[s.text for s in sources] if include_context else None,
        )
        await self._log(answer, user, question)
        yield Event("done", answer)

    def _below_threshold(self, top_score: float | None) -> bool:
        threshold = self.settings.abstain_threshold
        # Thresholds are on the reranker's score scale; RRF scores are not comparable.
        if threshold is None or self.retriever.reranker.name == "none" or top_score is None:
            return False
        return top_score < threshold

    async def _rewrite(self, question: str, history: list[ChatTurn]) -> tuple[str, Usage]:
        response = await self.fast_chat.ainvoke(rewrite_messages(history, question))
        text = content_text(response).strip().strip("\"«» ")
        return (text if 0 < len(text) <= 1000 else question), usage_of(response)

    async def _log(self, answer: Answer, user: User, question: str) -> None:
        try:
            await self.repo.log_query(
                {
                    "request_id": answer.request_id,
                    "user_id": user.id,
                    "question": question if self.settings.log_questions else None,
                    "question_hash": sha256(question.strip().lower()),
                    "abstained": answer.abstained,
                    "top_score": answer.top_score,
                    "n_citations": len(answer.citations),
                    "invalid_citations": len(answer.invalid_citations),
                    "prompt_version": answer.prompt_version,
                    "chat_model": self.settings.chat_model,
                    "input_tokens": answer.usage.input_tokens,
                    "output_tokens": answer.usage.output_tokens,
                    "cached_tokens": answer.usage.cached_tokens,
                    "timings_ms": answer.timings_ms,
                }
            )
        except Exception:  # telemetry must never fail an answer
            log.exception("query_log write failed for request %s", answer.request_id)

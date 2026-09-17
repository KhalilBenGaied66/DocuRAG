"""Query embedding → hybrid search → reranking, with timings for each stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from docurag.config import Settings
from docurag.db.repository import Repository, SearchMode
from docurag.models import SearchHit
from docurag.retrieval.rerank import Reranker


def _ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 1)


@dataclass(slots=True)
class RetrievalResult:
    hits: list[SearchHit]
    candidates: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(
        self, repo: Repository, embeddings: Embeddings, reranker: Reranker, settings: Settings
    ) -> None:
        self.repo = repo
        self.embeddings = embeddings
        self.reranker = reranker
        self.settings = settings

    async def retrieve(
        self,
        query: str,
        groups: list[str],
        *,
        mode: SearchMode = "hybrid",
        top_n: int | None = None,
        rerank: bool = True,
    ) -> RetrievalResult:
        top_n = top_n or self.settings.rerank_top_n
        timings: dict[str, float] = {}

        start = perf_counter()
        query_vec = await self.embeddings.aembed_query(query) if mode != "keyword" else None
        timings["embed_ms"] = _ms(start)

        start = perf_counter()
        candidates = await self.repo.search(
            query_text=query, query_vec=query_vec, groups=groups, mode=mode
        )
        timings["search_ms"] = _ms(start)

        start = perf_counter()
        if rerank:
            hits = await self.reranker.rerank(query, list(candidates), top_n)
        else:
            hits = candidates[:top_n]
        timings["rerank_ms"] = _ms(start)
        return RetrievalResult(hits=hits, candidates=candidates, timings_ms=timings)


class DocuRAGRetriever(BaseRetriever):
    """LangChain adapter, so agents (e.g. a LangGraph tool) can use the same ACL-aware search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any
    groups: list[str]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("DocuRAGRetriever is async: use ainvoke()")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = await self.retriever.retrieve(query, self.groups)
        return [
            Document(
                page_content=hit.body,
                metadata={
                    "chunk_id": hit.chunk_id,
                    "document_id": str(hit.document_id),
                    "title": hit.title,
                    "source_uri": hit.source_uri,
                    "section_path": hit.section_path,
                    "page_start": hit.page_start,
                    "page_end": hit.page_end,
                    "score": hit.score,
                },
            )
            for hit in result.hits
        ]

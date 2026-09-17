"""Second-stage ranking. A cross-encoder reads question and chunk together, which a
bi-encoder (the embedding model) cannot do, and reorders the fused candidates."""

from __future__ import annotations

import asyncio
from typing import Protocol

from docurag.config import Settings
from docurag.models import SearchHit


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...


class NoReranker:
    """Keeps the fused order. Abstention by score is disabled with this reranker."""

    name = "none"

    async def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        return hits[:top_n]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("reranker=cross-encoder needs the rerank extra") from exc
        self._model = CrossEncoder(model_name, max_length=max_length)

    async def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, f"{h.section_path}\n{h.body}") for h in hits]
        # CPU/GPU-bound: run it off the event loop so other requests keep flowing.
        scores = await asyncio.to_thread(self._model.predict, pairs, show_progress_bar=False)
        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = float(score)
        return sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)[:top_n]


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model)
    return NoReranker()

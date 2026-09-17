"""Retrieval and abstention metrics. Pure functions: easy to test, easy to trust."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

from docurag.evaluation.dataset import RelevantSource


class Located(Protocol):
    source_uri: str
    page_start: int | None
    page_end: int | None


def matches(item: Located, relevant: RelevantSource) -> bool:
    if item.source_uri != relevant.source_uri:
        return False
    if not relevant.pages or item.page_start is None:
        return True  # labelled at document level, or a source without pages (KB, records)
    end = item.page_end if item.page_end is not None else item.page_start
    return any(item.page_start <= p <= end for p in relevant.pages)


def recall_at_k(
    ranked: Sequence[Located], relevant: Sequence[RelevantSource], k: int
) -> float | None:
    """Share of the relevant sources found in the top k."""
    if not relevant:
        return None
    top = ranked[:k]
    return sum(any(matches(h, r) for h in top) for r in relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[Located], relevant: Sequence[RelevantSource]) -> float | None:
    if not relevant:
        return None
    for rank, item in enumerate(ranked, start=1):
        if any(matches(item, r) for r in relevant):
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float | int | bool | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def percentile(values: Iterable[float | None], p: float) -> float | None:
    kept = sorted(v for v in values if v is not None)
    if not kept:
        return None
    rank = max(1, math.ceil(p / 100 * len(kept)))  # nearest-rank method
    return kept[rank - 1]


def best_threshold(scored: Sequence[tuple[float | None, bool]]) -> dict | None:
    """Abstention threshold on the top reranker score that best separates answerable
    (score >= t → answer) from unanswerable questions."""
    points = [(s, answerable) for s, answerable in scored if s is not None]
    if not points or all(a for _, a in points) or not any(a for _, a in points):
        return None  # needs both answerable and unanswerable questions
    best: dict | None = None
    for t in sorted({s for s, _ in points}):
        accuracy = sum((s >= t) == a for s, a in points) / len(points)
        if best is None or accuracy > best["accuracy"]:
            best = {"threshold": round(t, 4), "accuracy": round(accuracy, 4)}
    return best

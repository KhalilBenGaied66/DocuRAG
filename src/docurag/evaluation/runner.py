"""Run the eval set against the live pipeline and compare with a baseline run.

Retrieval is scored separately from generation, so a regression can be traced to its stage.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from docurag.container import Container
from docurag.db.repository import SearchMode
from docurag.evaluation.dataset import EvalItem
from docurag.evaluation.metrics import (
    best_threshold,
    matches,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from docurag.generation.prompts import ANSWER_PROMPT_VERSION
from docurag.models import User

MEAN_METRICS = [
    "candidates_recall@5", "candidates_recall@10", "candidates_recall@20", "candidates_mrr",
    "final_recall", "final_mrr",
    "abstention_correct", "answered_when_answerable", "abstained_when_unanswerable",
    "cited_relevant", "invalid_citations", "input_tokens", "output_tokens",
]
LATENCY_METRICS = ["total_ms", "first_token_ms"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for key in MEAN_METRICS:
        value = mean(r.get(key) for r in rows)
        if value is not None:
            summary[key] = value
    for key in LATENCY_METRICS:
        values = [r.get(key) for r in rows]
        if any(v is not None for v in values):
            summary[f"{key}_p50"] = percentile(values, 50)
            summary[f"{key}_p95"] = percentile(values, 95)
    return summary


async def ragas_scores(
    rows: list[dict[str, Any]], judge: BaseChatModel, embeddings: Embeddings
) -> dict[str, float]:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise RuntimeError("--ragas needs the eval extra: pip install -e '.[eval]'") from exc

    samples = [
        {
            "user_input": r["question"],
            "retrieved_contexts": r["contexts"],
            "response": r["answer"],
            "reference": r["reference"],
        }
        for r in rows
    ]
    if not samples:
        return {}
    dataset = EvaluationDataset.from_list(samples)
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
               LLMContextRecall()]
    # RAGAS drives its own event loop: run it in a worker thread.
    result = await asyncio.to_thread(
        evaluate, dataset=dataset, metrics=metrics, llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(embeddings), show_progress=False,
    )
    frame = result.to_pandas()
    return {
        f"ragas_{col}": round(float(frame[col].mean()), 4)
        for col in frame.columns if frame[col].dtype.kind == "f"
    }


async def run_eval(
    container: Container,
    items: list[EvalItem],
    *,
    label: str,
    mode: SearchMode,
    default_groups: list[str],
    generation: bool = True,
    ragas: bool = False,
    judge: BaseChatModel | None = None,
    calibrate: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        groups = item.groups or default_groups
        retrieval = await container.retriever.retrieve(item.question, groups, mode=mode)
        row: dict[str, Any] = {
            "id": item.id,
            "category": item.category,
            "answerable": item.answerable,
            "top_score": retrieval.hits[0].score if retrieval.hits else None,
        }
        if item.answerable:
            for k in (5, 10, 20):
                row[f"candidates_recall@{k}"] = recall_at_k(retrieval.candidates, item.relevant, k)
            row["candidates_mrr"] = reciprocal_rank(retrieval.candidates, item.relevant)
            row["final_recall"] = recall_at_k(retrieval.hits, item.relevant, len(retrieval.hits) or 1)
            row["final_mrr"] = reciprocal_rank(retrieval.hits, item.relevant)

        if generation:
            answer = await container.service.answer(
                item.question, User(id="eval", groups=groups), mode=mode, include_context=True
            )
            row.update(
                {
                    "question": item.question,
                    "reference": item.reference,
                    "answer": answer.text,
                    "contexts": answer.contexts or [],
                    "abstained": answer.abstained,
                    "abstention_correct": answer.abstained != item.answerable,
                    "answered_when_answerable": (not answer.abstained) if item.answerable else None,
                    "abstained_when_unanswerable": answer.abstained if not item.answerable else None,
                    "cited_relevant": (
                        any(matches(c, r) for c in answer.citations for r in item.relevant)
                        if item.answerable and not answer.abstained else None
                    ),
                    "invalid_citations": len(answer.invalid_citations),
                    "input_tokens": answer.usage.input_tokens,
                    "output_tokens": answer.usage.output_tokens,
                    "total_ms": answer.timings_ms.get("total_ms"),
                    "first_token_ms": answer.timings_ms.get("first_token_ms"),
                }
            )
        rows.append(row)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    summary = summarize(rows)
    if ragas:
        if not generation:
            raise ValueError("--ragas needs generation (drop --retrieval-only)")
        judged = [
            r for r in rows
            if r["answerable"] and r.get("reference") and not r.get("abstained")
        ]
        summary.update(await ragas_scores(judged, judge or container.chat, container.embeddings))

    s = container.settings
    return {
        "label": label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "mode": mode,
            "reranker": container.retriever.reranker.name,
            "chat_model": s.chat_model,
            "embedding_model": s.embedding_model,
            "prompt_version": ANSWER_PROMPT_VERSION,
            "chunk_target_tokens": s.chunk_target_tokens,
            "rerank_top_n": s.rerank_top_n,
            "abstain_threshold": s.abstain_threshold,
        },
        "n": len(rows),
        "summary": summary,
        "by_category": {cat: summarize(rs) | {"n": len(rs)} for cat, rs in sorted(by_category.items())},
        "calibration": (
            best_threshold([(r["top_score"], r["answerable"]) for r in rows]) if calibrate else None
        ),
        "rows": [{k: v for k, v in r.items() if k != "contexts"} for r in rows],
    }


def save_result(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in result["label"])
    path = out_dir / f"{stamp}_{safe_label}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _fmt(value: float | None, key: str = "") -> str:
    if value is None:
        return "—"
    if key.endswith(("_ms", "_p50", "_p95", "tokens")) or abs(value) >= 100:
        return f"{value:.0f}"  # milliseconds and token counts: whole numbers
    return f"{value:.3f}"


def format_report(result: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    lines = [
        f"# Eval: {result['label']}  ({result['n']} questions)",
        "",
        "config: " + ", ".join(f"{k}={v}" for k, v in result["config"].items()),
        "",
    ]
    base = (baseline or {}).get("summary", {})
    if baseline:
        lines += [f"| metric | {baseline['label']} | {result['label']} | Δ |", "|---|---:|---:|---:|"]
    else:
        lines += ["| metric | value |", "|---|---:|"]
    for key, value in result["summary"].items():
        if baseline:
            old = base.get(key)
            delta = (value - old) if (value is not None and old is not None) else None
            sign = "+" if delta is not None and delta > 0 else ""
            lines.append(f"| {key} | {_fmt(old, key)} | {_fmt(value, key)} | {sign}{_fmt(delta, key)} |")
        else:
            lines.append(f"| {key} | {_fmt(value, key)} |")
    lines += ["", "## By category", "", "| category | n | final_recall | abstention_correct |",
              "|---|---:|---:|---:|"]
    for cat, stats in result["by_category"].items():
        lines.append(
            f"| {cat} | {stats['n']} | {_fmt(stats.get('final_recall'))} |"
            f" {_fmt(stats.get('abstention_correct'))} |"
        )
    if result.get("calibration"):
        c = result["calibration"]
        lines += ["", f"Suggested DOCURAG_ABSTAIN_THRESHOLD={c['threshold']}"
                      f" (abstention accuracy {c['accuracy']:.3f} on this set)"]
    return "\n".join(lines)

from __future__ import annotations

import json
from pathlib import Path

from docurag.evaluation.dataset import load_dataset
from docurag.evaluation.runner import format_report, save_result, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_summary_ignores_metrics_that_do_not_apply():
    rows = [
        {"final_recall": 1.0, "abstention_correct": True, "total_ms": 900.0},
        {"final_recall": 0.0, "abstention_correct": True, "total_ms": 1500.0},
        {"final_recall": None, "abstention_correct": False, "total_ms": 3000.0},  # unanswerable
    ]
    summary = summarize(rows)
    assert summary["final_recall"] == 0.5
    assert summary["abstention_correct"] == round(2 / 3, 4)
    assert summary["total_ms_p50"] == 1500.0 and summary["total_ms_p95"] == 3000.0


def test_report_shows_deltas_against_baseline(tmp_path):
    base = {"label": "vector", "summary": {"final_recall": 0.61}}
    current = {
        "label": "hybrid", "n": 2, "config": {"mode": "hybrid"},
        "summary": {"final_recall": 0.78}, "by_category": {"table": {"n": 2, "final_recall": 0.5}},
        "calibration": {"threshold": 0.42, "accuracy": 0.9},
    }
    report = format_report(current, base)
    assert "| final_recall | 0.610 | 0.780 | +0.170 |" in report
    assert "DOCURAG_ABSTAIN_THRESHOLD=0.42" in report

    path = save_result(current | {"rows": []}, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["label"] == "hybrid"


def test_example_eval_set_is_valid_and_matches_the_manifest():
    items = load_dataset(ROOT / "examples" / "eval" / "questions.jsonl")
    assert len(items) >= 10
    assert any(not item.answerable for item in items)
    manifest = (ROOT / "examples" / "corpus" / "manifest.toml").read_text(encoding="utf-8")
    for item in items:
        for rel in item.relevant:
            assert rel.source_uri.removeprefix("corpus://maintenance/") in manifest, rel.source_uri

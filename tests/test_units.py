"""Configuration, manifest, SQL composition and evaluation metrics."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docurag.config import Settings
from docurag.db.pool import render_schema
from docurag.db.repository import _natural_key, _search_sql, vector_literal
from docurag.evaluation.dataset import RelevantSource
from docurag.evaluation.metrics import (
    best_threshold,
    matches,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from docurag.ingestion.manifest import load_manifest
from tests.conftest import make_hit

# ------------------------------------------------------------------ config


@pytest.mark.parametrize(
    "overrides",
    [
        {"env": "prod", "auth_mode": "dev"},
        {"env": "prod", "provider": "fake", "auth_mode": "jwt"},
        {"text_search_config": "french'); DROP TABLE chunks; --"},
        {"embedding_dimensions": 5000},
        {"chunk_target_tokens": 800, "chunk_max_tokens": 600},
    ],
)
def test_unsafe_settings_are_refused(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    ("url", "valid"),
    [
        ("postgresql://u:p@ep-x.eu-central-1.aws.neon.tech/neondb?sslmode=require", True),
        ("postgres://u@localhost/db", True),
        ("host=localhost port=5433 dbname=docurag user=postgres", True),
        ("", False),
        ("mysql://u@localhost/db", False),
        ("ep-x.eu-central-1.aws.neon.tech", False),
    ],
)
def test_database_url_formats(url, valid):
    if valid:
        assert Settings(_env_file=None, database_url=url).database_url == url
    else:
        with pytest.raises(ValidationError, match="PostgreSQL connection string"):
            Settings(_env_file=None, database_url=url)


def test_ollama_context_window_must_fit_the_sources():
    with pytest.raises(ValidationError, match="ollama_num_ctx"):
        Settings(_env_file=None, provider="ollama", ollama_num_ctx=4096)


def test_ollama_provider_builds_local_clients(settings):
    pytest.importorskip("langchain_ollama")
    from docurag.providers import build_chat_model, build_embeddings

    local = settings.model_copy(update={
        "provider": "ollama", "chat_model": "mistral-nemo",
        "embedding_model": "bge-m3", "embedding_dimensions": 1024,
    })
    chat, embeddings = build_chat_model(local), build_embeddings(local)
    assert type(chat).__name__ == "ChatOllama" and chat.num_ctx == 8192
    assert type(embeddings).__name__ == "OllamaEmbeddings" and embeddings.model == "bge-m3"


def test_schema_renders_validated_values(settings):
    sql = render_schema(settings)
    assert "halfvec(256)" in sql and "'french'::regconfig" in sql and "${" not in sql


# ------------------------------------------------------------------ SQL


def test_search_sql_per_mode():
    hybrid, vector, keyword = _search_sql("hybrid"), _search_sql("vector"), _search_sql("keyword")
    assert "<=>" in hybrid and "tsvector_to_array" in hybrid
    assert "<=>" in vector and "tsvector_to_array" not in vector
    assert "<=>" not in keyword and "tsvector_to_array" in keyword
    for sql in (hybrid, vector, keyword):
        assert sql.count("acl_groups && %(groups)s::text[]") == sql.count("FROM chunks")


def test_version_ordering_is_natural():
    assert _natural_key("2.10") > _natural_key("2.9")
    assert _natural_key("rev C") > _natural_key("rev B")
    assert _natural_key("3") > _natural_key(None)


def test_vector_literal():
    assert vector_literal([0.5, -1.0, 1e-9]) == "[0.5,-1,1e-09]"


# ------------------------------------------------------------------ manifest


def test_manifest_merges_defaults_and_builds_stable_uris(tmp_path):
    (tmp_path / "corpus").mkdir()
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
uri_prefix = "corpus://maintenance/"
base_dir = "corpus"

[defaults]
acl_groups = ["maintenance"]

[[documents]]
path = "manuals/p200.pdf"
doc_key = "manual-p200"
version = "C"

[[documents]]
path = "records/ot.csv"
acl_groups = ["maintenance", "methods"]
record_id_field = "ot_id"
record_text_fields = ["symptome"]
""",
        encoding="utf-8",
    )
    loaded = load_manifest(manifest)
    first, second = loaded.documents
    assert first.source_uri == "corpus://maintenance/manuals/p200.pdf"
    assert first.acl_groups == ["maintenance"] and first.key == "manual-p200"
    assert second.acl_groups == ["maintenance", "methods"]
    assert second.path.endswith("ot.csv")


def test_manifest_rejects_duplicates_and_missing_acl(tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[documents]]\npath = "a.md"\nacl_groups = ["x"]\n'
                        '[[documents]]\npath = "a.md"\nacl_groups = ["x"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(manifest)
    manifest.write_text('[[documents]]\npath = "a.md"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="acl_groups"):
        load_manifest(manifest)


# ------------------------------------------------------------------ metrics

REL = [RelevantSource(source_uri="corpus://manuals/p200.pdf", pages=[12])]


def test_page_level_matching():
    assert matches(make_hit(pages=(11, 13)), REL[0])
    assert not matches(make_hit(pages=(20, 21)), REL[0])
    assert matches(make_hit(pages=(None, None)), REL[0])  # sources without pages: document level
    assert not matches(make_hit(uri="corpus://other.pdf"), REL[0])


def test_recall_and_mrr():
    ranked = [make_hit(1, pages=(30, 30)), make_hit(2, pages=(12, 12))]
    assert recall_at_k(ranked, REL, 1) == 0.0
    assert recall_at_k(ranked, REL, 2) == 1.0
    assert reciprocal_rank(ranked, REL) == 0.5
    assert recall_at_k(ranked, [], 5) is None  # unanswerable questions don't count


def test_percentile_nearest_rank():
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([None], 50) is None


def test_best_threshold_separates_answerable_from_unanswerable():
    scored = [(0.9, True), (0.7, True), (0.4, False), (0.2, False), (0.65, False)]
    assert best_threshold(scored) == {"threshold": 0.7, "accuracy": 1.0}
    assert best_threshold([(0.9, True)]) is None

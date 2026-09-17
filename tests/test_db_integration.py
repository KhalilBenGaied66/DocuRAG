"""End-to-end against PostgreSQL + pgvector, in a throwaway schema.

Run with DOCURAG_TEST_DATABASE_URL set (environment or .env): pytest -m db
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docurag.container import build_container
from docurag.db.pool import SchemaMismatch
from docurag.ingestion.manifest import Manifest
from docurag.models import DocumentSpec, User
from docurag.worker import run_worker
from tests.conftest import TEST_DATABASE_URL

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not TEST_DATABASE_URL, reason="set DOCURAG_TEST_DATABASE_URL (PostgreSQL + pgvector)"),
]

P200 = """# Manuel P-200

## 4.3 Couples de serrage

Serrer les vis M8 du carter au couple de 25 N·m, en croix et à froid.

## 5.2 Alarme E-41

L'alarme E-41 signale une vibration excessive du palier côté accouplement.
Contrôler l'alignement puis l'état du roulement.
"""

HR = """# Politique RH

## Grille des salaires

La grille des salaires des techniciens est confidentielle.
"""


@pytest.fixture
async def container(db_settings):
    built = await build_container(db_settings)
    try:
        yield built
    finally:
        await built.close()


def spec(path: Path, acl: list[str], **kw) -> DocumentSpec:
    return DocumentSpec(source_uri=f"corpus://{path.name}", path=str(path), acl_groups=acl, **kw)


@pytest.fixture
def corpus(tmp_path) -> dict[str, Path]:
    (tmp_path / "p200.md").write_text(P200, encoding="utf-8")
    (tmp_path / "hr.md").write_text(HR, encoding="utf-8")
    return {"p200": tmp_path / "p200.md", "hr": tmp_path / "hr.md"}


async def search(container, text, groups, mode="hybrid"):
    vec = await container.embeddings.aembed_query(text) if mode != "keyword" else None
    return await container.repo.search(query_text=text, query_vec=vec, groups=groups, mode=mode)


async def test_acl_never_leaks_in_any_mode(container, corpus):
    await container.pipeline.ingest(spec(corpus["p200"], ["maintenance"]))
    await container.pipeline.ingest(spec(corpus["hr"], ["hr"]))

    for mode in ("hybrid", "vector", "keyword"):
        hits = await search(container, "grille des salaires confidentielle", ["maintenance"], mode)
        assert all(h.source_uri != "corpus://hr.md" for h in hits), mode
        hr_hits = await search(container, "grille des salaires", ["hr"], mode)
        assert hr_hits and all(h.source_uri == "corpus://hr.md" for h in hr_hits), mode
    assert await search(container, "salaires", []) == []


async def test_keyword_leg_finds_exact_codes(container, corpus):
    await container.pipeline.ingest(spec(corpus["p200"], ["maintenance"]))
    hits = await search(container, "Que signifie l'alarme E-41 ?", ["maintenance"], "keyword")
    assert hits[0].section_path.endswith("5.2 Alarme E-41")
    assert hits[0].kw_rank == 1 and hits[0].vec_rank is None


async def test_incremental_ingestion(container, corpus):
    first = await container.pipeline.ingest(spec(corpus["p200"], ["maintenance"]))
    assert first.status == "indexed" and first.report["embeddings_cached"] == 0

    again = await container.pipeline.ingest(spec(corpus["p200"], ["maintenance"]))
    assert again.status == "unchanged"

    moved = await container.pipeline.ingest(spec(corpus["p200"], ["methods"]))
    assert moved.status == "acl_updated"
    assert await search(container, "couple de serrage", ["maintenance"]) == []
    assert await search(container, "couple de serrage", ["methods"])

    corpus["p200"].write_text(P200.replace("25 N·m", "27 N·m"), encoding="utf-8")
    changed = await container.pipeline.ingest(spec(corpus["p200"], ["methods"]))
    assert changed.status == "indexed"
    assert changed.report["embeddings_cached"] >= 1  # the untouched section wasn't re-embedded
    bodies = " ".join(h.body for h in await search(container, "couple vis M8", ["methods"]))
    assert "27 N·m" in bodies and "25 N·m" not in bodies


async def test_only_the_latest_version_is_searchable(container, tmp_path):
    rev_b, rev_c = tmp_path / "p200_b.md", tmp_path / "p200_c.md"
    rev_b.write_text(P200, encoding="utf-8")
    rev_c.write_text(P200.replace("25 N·m", "30 N·m"), encoding="utf-8")
    await container.pipeline.ingest(spec(rev_c, ["maintenance"], doc_key="p200", version="C"))
    await container.pipeline.ingest(spec(rev_b, ["maintenance"], doc_key="p200", version="B"))

    hits = await search(container, "couple de serrage vis M8", ["maintenance"])
    assert {h.source_uri for h in hits} == {"corpus://p200_c.md"}


async def test_prune_propagates_deletions(container, corpus):
    specs = [spec(corpus["p200"], ["maintenance"]), spec(corpus["hr"], ["hr"])]
    await container.pipeline.sync(Manifest("corpus://", specs))
    _, removed = await container.pipeline.sync(Manifest("corpus://", specs[:1]), prune=True)
    assert removed == ["corpus://hr.md"]
    assert await search(container, "salaires", ["hr"]) == []


async def test_answer_end_to_end_with_citation(container, corpus):
    await container.pipeline.ingest(spec(corpus["p200"], ["maintenance"]))
    answer = await container.service.answer(
        "Quel couple de serrage pour les vis M8 du carter ?", User(id="t", groups=["maintenance"])
    )
    assert not answer.abstained
    assert answer.citations[0].section_path.endswith("4.3 Couples de serrage")
    async with container.pool.connection() as conn:
        row = await (await conn.execute("SELECT count(*) AS n FROM query_log")).fetchone()
    assert row["n"] == 1


async def test_job_queue_and_worker(container, corpus):
    job_id = await container.repo.enqueue_job(spec(corpus["p200"], ["maintenance"]), "admin")
    assert await run_worker(container, once=True) == 1
    job = await container.repo.get_job(job_id)
    assert job["status"] == "done" and job["result"]["status"] == "indexed"
    assert await container.repo.claim_job() is None


async def test_index_built_with_other_settings_is_refused(container, db_settings):
    other = db_settings.model_copy(update={"embedding_model": "another-model"})
    with pytest.raises(SchemaMismatch):
        await build_container(other)

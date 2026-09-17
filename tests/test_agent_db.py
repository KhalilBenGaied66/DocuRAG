"""Agent persistence against PostgreSQL, in a throwaway schema.

Run with DOCURAG_TEST_DATABASE_URL set (environment or .env): pytest -m db
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from docurag.db.pool import open_pool
from maintagent.business import BusinessRepository
from maintagent.models import WorkOrderDraft
from tests.agent_fakes import DRAFT, REPORT, SUPERVISOR, TECH, Harness, incident_script, intent
from tests.conftest import TEST_DATABASE_URL

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "business"
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not TEST_DATABASE_URL, reason="set DOCURAG_TEST_DATABASE_URL (PostgreSQL)"),
]


@pytest.fixture
async def business(db_settings):
    pool = await open_pool(db_settings)
    repo = BusinessRepository(pool)
    await repo.init_schema()
    await repo.seed(EXAMPLES)
    yield repo
    await pool.close()


async def test_reads_run_in_read_only_transactions(business):
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        await business._read("INSERT INTO equipment (tag, site, model) VALUES ('X-01', 'x', 'x')", {})


async def test_seeded_registry_and_history(business):
    assert (await business.equipment("P-200-07"))["criticality"] == "A"
    assert await business.equipment("P-999") is None
    history = await business.interventions("P-200-07", since_days=3650)
    assert history[0]["ot_id"] == "OT-2026-0702" and len(history) == 3


async def test_seed_is_repeatable(business):
    assert await business.seed(EXAMPLES) == {"equipment": 3, "interventions": 5}
    assert len(await business.interventions("P-200-07", since_days=3650)) == 3


async def test_work_order_creation_is_idempotent(business):
    draft = WorkOrderDraft.model_validate(DRAFT)
    first = await business.create_work_order(ticket_id="T-1", draft=draft, requested_by="tech-42",
                                             approved_by="sup-7", sources=[])
    second = await business.create_work_order(ticket_id="T-1", draft=draft, requested_by="tech-42",
                                              approved_by="sup-7", sources=[])
    assert first == second
    async with business.pool.connection() as conn:
        row = await (await conn.execute("SELECT count(*) AS n FROM work_orders")).fetchone()
    assert row["n"] == 1


async def test_a_pending_approval_survives_a_restart(db_settings):
    async def checkpointer():
        pool = await open_pool(db_settings, prepare_threshold=0)
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        return pool, saver

    pool_a, saver_a = await checkpointer()
    before = Harness(incident_script(), [intent("incident", "P-200-07")], checkpointer=saver_a)
    assert (await before.agent.submit("T-1", REPORT, TECH)).status == "awaiting_approval"
    await pool_a.close()  # "the process restarts"

    pool_b, saver_b = await checkpointer()
    after = Harness([], [], checkpointer=saver_b)  # a fresh process: no model calls expected
    view = await after.agent.decide("T-1", SUPERVISOR, approved=True)
    await pool_b.close()

    assert view.status == "work_order_created" and view.proposal["sources"] == ["doc:7"]

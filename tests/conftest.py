from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import psycopg
import pytest
from dotenv import dotenv_values

os.environ.setdefault("DOCURAG_TOKENIZER", "approx")  # deterministic, offline token counts
if sys.platform == "win32":
    # psycopg's async driver needs a selector event loop on Windows.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Only this key is read from .env: the other DOCURAG_* values there must not leak into tests.
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
if not os.environ.get("DOCURAG_TEST_DATABASE_URL") and _ENV_FILE.exists():
    _value = dotenv_values(_ENV_FILE).get("DOCURAG_TEST_DATABASE_URL")
    if _value:
        os.environ["DOCURAG_TEST_DATABASE_URL"] = _value
TEST_DATABASE_URL = os.environ.get("DOCURAG_TEST_DATABASE_URL")

from docurag.config import Settings  # noqa: E402
from docurag.models import SearchHit  # noqa: E402
from docurag.retrieval.retriever import RetrievalResult  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        env="test",
        provider="fake",
        embedding_model="hashing-test",
        embedding_dimensions=256,
        auth_mode="dev",
        chunk_target_tokens=120,
        chunk_max_tokens=200,
        table_max_tokens=300,
        parent_max_tokens=400,
        context_budget_tokens=1500,
        storage_dir=str(tmp_path / "uploads"),
    )


@pytest.fixture
async def db_settings(settings):
    """Settings pointing at a throwaway schema in the test database, dropped afterwards.

    Uses the db_schema setting (SET search_path per connection) rather than a startup
    `options` parameter, which connection proxies such as Neon's may not pass through.
    """
    if not TEST_DATABASE_URL:
        pytest.skip("set DOCURAG_TEST_DATABASE_URL (environment or .env)")
    schema = f"docurag_test_{uuid4().hex[:10]}"
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield settings.model_copy(
            update={"database_url": TEST_DATABASE_URL, "db_schema": schema, "db_pool_max": 4}
        )
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as conn:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def make_hit(
    i: int = 1,
    *,
    section_id: int = 10,
    ord_: int = 0,
    body: str = "Serrer les vis M8 au couple de 25 N·m.",
    score: float = 0.5,
    rerank: float | None = None,
    pages: tuple[int | None, int | None] = (12, 12),
    uri: str = "corpus://manuals/p200.pdf",
    section_path: str = "4 Maintenance > 4.3 Couples de serrage",
) -> SearchHit:
    return SearchHit(
        chunk_id=i, document_id=UUID(int=i), section_id=section_id, ord=ord_,
        title="Manuel P-200", source_uri=uri, section_path=section_path,
        page_start=pages[0], page_end=pages[1], kind="text", body=body,
        fused_score=score, rerank_score=rerank,
    )


class FakeRepo:
    """In-memory stand-in for the parts of Repository the answer path uses."""

    def __init__(self, sections: dict[int, dict] | None = None) -> None:
        self.sections = sections or {}
        self.logged: list[dict] = []

    async def fetch_sections(self, ids: list[int]) -> dict[int, dict]:
        return {i: self.sections[i] for i in ids if i in self.sections}

    async def fetch_window(self, section_id: int, ord_: int, window: int = 1) -> list[dict]:
        return [{"ord": ord_, "body": f"window {section_id}/{ord_}", "page_start": 3,
                 "page_end": 4}]

    async def log_query(self, row: dict) -> None:
        self.logged.append(row)


class FakeRetriever:
    def __init__(self, hits: list[SearchHit], reranker: str = "none") -> None:
        self.hits = hits
        self.reranker = SimpleNamespace(name=reranker)
        self.queries: list[tuple[str, list[str]]] = []

    async def retrieve(self, query, groups, *, mode="hybrid", top_n=None, rerank=True):
        self.queries.append((query, groups))
        return RetrievalResult(hits=list(self.hits), candidates=list(self.hits),
                               timings_ms={"search_ms": 1.0})


def section(content: str, tokens: int = 50, pages: tuple[int, int] = (12, 13)) -> dict:
    return {"id": 0, "path": "4 Maintenance > 4.3", "content": content, "tokens": tokens,
            "page_start": pages[0], "page_end": pages[1]}

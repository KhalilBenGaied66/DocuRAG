"""Connection pool, schema rendering and the start-up checks that prevent a mixed index."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from importlib.resources import files
from string import Template

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from docurag.config import Settings

MIN_PGVECTOR = (0, 7, 0)  # halfvec
ITERATIVE_SCAN_PGVECTOR = (0, 8, 0)


class SchemaMismatch(RuntimeError):
    """The database was built for another embedding model, dimension or text-search config."""


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:3] if part.isdigit())


def schema_configure(schema: str | None) -> Callable[[AsyncConnection], Awaitable[None]] | None:
    """Per-connection setup putting `schema` first on the search path (public keeps pgvector)."""
    if schema is None:
        return None

    async def configure(conn: AsyncConnection) -> None:
        await conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        if not conn.autocommit:
            await conn.commit()

    return configure


async def open_pool(
    settings: Settings, *, max_size: int | None = None, prepare_threshold: int | None = 5
) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min,
        max_size=max_size or settings.db_pool_max,
        kwargs={"row_factory": dict_row, "autocommit": True, "prepare_threshold": prepare_threshold},
        configure=schema_configure(settings.db_schema),
        # Neon's free tier suspends after 5 idle minutes and drops connections: check first.
        check=AsyncConnectionPool.check_connection if settings.db_check_connection else None,
        open=False,
    )
    try:
        await pool.open(wait=True, timeout=settings.db_connect_timeout_s)
    except Exception:
        await pool.close()
        raise
    return pool


def render_schema(settings: Settings) -> str:
    template = Template((files("docurag.db") / "schema.sql").read_text(encoding="utf-8"))
    # Both values are validated by Settings (closed list / bounded int) before reaching SQL.
    return template.substitute(
        dim=int(settings.embedding_dimensions), ts_config=settings.text_search_config
    )


async def init_schema(pool: AsyncConnectionPool, settings: Settings) -> tuple[int, ...]:
    """Create the schema if needed; refuse to run against an index built differently.

    Returns the pgvector version.
    """
    expected = {
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": str(settings.embedding_dimensions),
        "text_search_config": settings.text_search_config,
    }
    async with pool.connection() as conn:
        if settings.db_schema:
            await conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(settings.db_schema))
            )
        await conn.execute(render_schema(settings))
        row = await (
            await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()
        version = _version_tuple(row["extversion"]) if row else (0,)
        if version < MIN_PGVECTOR:
            raise RuntimeError(f"pgvector >= 0.7.0 is required (found {row and row['extversion']})")

        stored = {
            r["key"]: r["value"]
            for r in await (await conn.execute("SELECT key, value FROM schema_info")).fetchall()
        }
        if not stored:
            async with conn.transaction():
                for key, value in expected.items():
                    await conn.execute(
                        "INSERT INTO schema_info (key, value) VALUES (%s, %s)", (key, value)
                    )
            return version
        diffs = {k: (stored.get(k), v) for k, v in expected.items() if stored.get(k) != v}
        if diffs:
            detail = ", ".join(f"{k}: index={old!r} settings={new!r}" for k, (old, new) in diffs.items())
            raise SchemaMismatch(
                f"this database was indexed with different settings ({detail}). Build a new "
                "index in a fresh database and switch over when the eval passes."
            )
    return version


def supports_iterative_scan(version: tuple[int, ...]) -> bool:
    return version >= ITERATIVE_SCAN_PGVECTOR

"""All SQL lives here. Every search goes through the ACL filter; there is no unfiltered path."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from docurag.config import Settings
from docurag.models import ChunkDraft, DocType, DocumentSpec, SearchHit, SectionDraft

SearchMode = Literal["hybrid", "vector", "keyword"]

_EMPTY_LEG = "SELECT NULL::bigint AS id, NULL::bigint AS r WHERE false"

# Question words are OR-ed (not AND-ed): a long natural-language question would otherwise
# match nothing. ts_rank then favours chunks matching more of the terms. Not ts_rank_cd:
# cover density is ~29x slower when common words match most chunks (measured, 20k chunks).
# Normalisation 1 divides by 1 + log(length), so long chunks aren't favoured for their size.
# MATERIALIZED: inlined, the tsquery is rebuilt and parsed again for every ranked row (80 ms vs
# 30 ms at 20k matching chunks); materialized, it is computed once and the GIN index still applies.
_KW_QUERY = """
q AS MATERIALIZED (
  SELECT NULLIF(array_to_string(ARRAY(
           SELECT quote_literal(lexeme)
           FROM unnest(tsvector_to_array(to_tsvector(%(ts_config)s::regconfig, %(qtext)s))) AS lexeme
         ), ' | '), '')::tsquery AS tsq
)"""

_VEC_LEG = """
  SELECT id, row_number() OVER (ORDER BY embedding <=> %(qvec)s::halfvec) AS r
  FROM chunks
  WHERE is_current AND acl_groups && %(groups)s::text[]
  ORDER BY embedding <=> %(qvec)s::halfvec
  LIMIT %(per_leg)s"""

_KW_LEG = """
  SELECT c.id, row_number() OVER (ORDER BY ts_rank(c.content_tsv, q.tsq, 1) DESC) AS r
  FROM chunks c, q
  WHERE c.is_current AND c.acl_groups && %(groups)s::text[] AND c.content_tsv @@ q.tsq
  ORDER BY ts_rank(c.content_tsv, q.tsq, 1) DESC
  LIMIT %(per_leg)s"""

_FUSE = """
WITH {kw_query}
vec AS ({vec_leg}),
kw AS ({kw_leg}),
fused AS (
  SELECT id,
         (COALESCE(1.0 / (%(rrf_k)s + vec.r), 0) + COALESCE(1.0 / (%(rrf_k)s + kw.r), 0))::float8
           AS score,
         vec.r AS vec_rank, kw.r AS kw_rank
  FROM vec FULL OUTER JOIN kw USING (id)
)
SELECT c.id AS chunk_id, c.document_id, c.section_id, c.ord, d.title, d.source_uri,
       c.section_path, c.page_start, c.page_end, c.kind, c.body,
       f.score, f.vec_rank, f.kw_rank
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC, c.id
LIMIT %(limit)s"""


def _search_sql(mode: SearchMode) -> str:
    return _FUSE.format(
        kw_query=(_KW_QUERY + ",") if mode != "vector" else "",
        vec_leg=_VEC_LEG if mode != "keyword" else _EMPTY_LEG,
        kw_leg=_KW_LEG if mode != "vector" else _EMPTY_LEG,
    )


def vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def _natural_key(version: str | None) -> tuple:
    if not version:
        return ()
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", version) if p)


class Repository:
    def __init__(
        self, pool: AsyncConnectionPool, settings: Settings, *, iterative_scan: bool
    ) -> None:
        self.pool = pool
        self.settings = settings
        self.iterative_scan = iterative_scan

    # ------------------------------------------------------------ documents

    async def get_document_by_uri(self, source_uri: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, content_hash, embedding_model, acl_groups, metadata, doc_key, version "
                "FROM documents WHERE source_uri = %s",
                (source_uri,),
            )
            return await cur.fetchone()

    async def get_document(self, document_id: UUID) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, source_uri, doc_key, version, title, doc_type, acl_groups, is_current,"
                " report, ingested_at FROM documents WHERE id = %s",
                (document_id,),
            )
            return await cur.fetchone()

    async def cached_embeddings(self, hashes: list[str], model: str) -> dict[str, list[float]]:
        if not hashes:
            return {}
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT content_hash, embedding::text AS e FROM embedding_cache "
                "WHERE model = %s AND content_hash = ANY(%s)",
                (model, hashes),
            )
            return {r["content_hash"]: json.loads(r["e"]) for r in await cur.fetchall()}

    async def store_embeddings(self, items: dict[str, list[float]], model: str) -> None:
        if not items:
            return
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO embedding_cache (content_hash, model, embedding) "
                "VALUES (%s, %s, %s::halfvec) ON CONFLICT DO NOTHING",
                [(h, model, vector_literal(v)) for h, v in items.items()],
            )

    async def write_document(
        self,
        *,
        spec: DocumentSpec,
        doc_type: DocType,
        title: str,
        content_hash: str,
        report: dict[str, Any],
        sections: list[SectionDraft],
        chunks: list[ChunkDraft],
        embeddings: dict[str, list[float]],
    ) -> UUID:
        """Replace a document's sections and chunks atomically: readers see old or new, never half."""
        model = self.settings.embedding_model
        async with self.pool.connection() as conn, conn.transaction():
            row = await (
                await conn.execute(
                    "SELECT id FROM documents WHERE source_uri = %s FOR UPDATE", (spec.source_uri,)
                )
            ).fetchone()
            doc_id: UUID = row["id"] if row else uuid4()
            params = {
                "id": doc_id, "uri": spec.source_uri, "key": spec.key, "version": spec.version,
                "title": title, "type": doc_type, "lang": spec.language,
                "acl": spec.acl_groups, "hash": content_hash, "meta": Jsonb(spec.metadata),
                "model": model, "report": Jsonb(report),
            }
            if row:
                await conn.execute("DELETE FROM sections WHERE document_id = %s", (doc_id,))
                await conn.execute(
                    "UPDATE documents SET doc_key=%(key)s, version=%(version)s, title=%(title)s,"
                    " doc_type=%(type)s, language=%(lang)s, acl_groups=%(acl)s,"
                    " content_hash=%(hash)s, metadata=%(meta)s, embedding_model=%(model)s,"
                    " report=%(report)s, ingested_at=now() WHERE id=%(id)s",
                    params,
                )
            else:
                await conn.execute(
                    "INSERT INTO documents (id, source_uri, doc_key, version, title, doc_type,"
                    " language, acl_groups, content_hash, metadata, embedding_model, report)"
                    " VALUES (%(id)s, %(uri)s, %(key)s, %(version)s, %(title)s, %(type)s,"
                    " %(lang)s, %(acl)s, %(hash)s, %(meta)s, %(model)s, %(report)s)",
                    params,
                )

            # One round trip for all sections: with a hosted database, one INSERT per section
            # would cost hundreds of network round trips on a long manual.
            rows = await (
                await conn.execute(
                    "INSERT INTO sections (document_id, ord, path, content, tokens, page_start, page_end)"
                    " SELECT %s, * FROM unnest(%s::int[], %s::text[], %s::text[], %s::int[],"
                    " %s::int[], %s::int[]) RETURNING ord, id",
                    (
                        doc_id, [s.ord for s in sections], [s.path_str for s in sections],
                        [s.content for s in sections], [s.tokens for s in sections],
                        [s.page_start for s in sections], [s.page_end for s in sections],
                    ),
                )
            ).fetchall()
            section_ids: dict[int, int] = {r["ord"]: r["id"] for r in rows}

            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks (document_id, section_id, ord, kind, section_path,"
                    " page_start, page_end, body, content, embedding, embedding_model,"
                    " acl_groups, content_hash, tokens)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::halfvec, %s, %s, %s, %s)",
                    [
                        (
                            doc_id, section_ids[c.section_ord], c.ord, c.kind, c.section_path,
                            c.page_start, c.page_end, c.body, c.content,
                            vector_literal(embeddings[c.content_hash]), model,
                            spec.acl_groups, c.content_hash, c.tokens,
                        )
                        for c in chunks
                    ],
                )
            await self._refresh_current_version(conn, spec.key)
        return doc_id

    async def _refresh_current_version(self, conn: Any, doc_key: str) -> None:
        """Only the latest version of a logical document is searchable by default."""
        rows = await (
            await conn.execute(
                "SELECT id, version, ingested_at FROM documents WHERE doc_key = %s", (doc_key,)
            )
        ).fetchall()
        if not rows:
            return
        current = max(rows, key=lambda r: (_natural_key(r["version"]), r["ingested_at"]))["id"]
        await conn.execute(
            "UPDATE documents SET is_current = (id = %s) WHERE doc_key = %s", (current, doc_key)
        )
        await conn.execute(
            "UPDATE chunks c SET is_current = (c.document_id = %s) FROM documents d"
            " WHERE d.id = c.document_id AND d.doc_key = %s"
            " AND c.is_current IS DISTINCT FROM (c.document_id = %s)",
            (current, doc_key, current),
        )

    async def update_access(self, document_id: UUID, acl_groups: list[str], metadata: dict) -> None:
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                "UPDATE documents SET acl_groups = %s, metadata = %s WHERE id = %s",
                (acl_groups, Jsonb(metadata), document_id),
            )
            await conn.execute(
                "UPDATE chunks SET acl_groups = %s WHERE document_id = %s", (acl_groups, document_id)
            )

    async def delete_document(self, document_id: UUID) -> bool:
        async with self.pool.connection() as conn, conn.transaction():
            row = await (
                await conn.execute(
                    "DELETE FROM documents WHERE id = %s RETURNING doc_key", (document_id,)
                )
            ).fetchone()
            if row:
                await self._refresh_current_version(conn, row["doc_key"])
            return row is not None

    async def delete_missing(self, uri_prefix: str, keep: set[str]) -> list[str]:
        """Propagate deletions: documents under uri_prefix that are no longer in the source."""
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT id, source_uri FROM documents WHERE starts_with(source_uri, %s)",
                    (uri_prefix,),
                )
            ).fetchall()
        removed = []
        for row in rows:
            if row["source_uri"] not in keep and await self.delete_document(row["id"]):
                removed.append(row["source_uri"])
        return removed

    # ------------------------------------------------------------ search

    async def search(
        self,
        *,
        query_text: str,
        query_vec: Sequence[float] | None,
        groups: list[str],
        mode: SearchMode = "hybrid",
        per_leg: int | None = None,
        limit: int | None = None,
    ) -> list[SearchHit]:
        if not groups:
            return []  # no group, no document: never fall back to an unfiltered search
        if mode != "keyword" and query_vec is None:
            raise ValueError(f"mode={mode} needs a query vector")
        s = self.settings
        params = {
            "qtext": query_text,
            "qvec": vector_literal(query_vec) if query_vec is not None else None,
            "groups": groups,
            "ts_config": s.text_search_config,
            "per_leg": per_leg or s.candidates_per_leg,
            "limit": limit or s.candidates_per_leg,
            "rrf_k": s.rrf_k,
        }
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(f"SET LOCAL hnsw.ef_search = {int(s.hnsw_ef_search)}")
            if self.iterative_scan:
                # Keep scanning the HNSW graph when the ACL filter removes candidates.
                await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = await (await conn.execute(_search_sql(mode), params)).fetchall()
        return [
            SearchHit(
                chunk_id=r["chunk_id"], document_id=r["document_id"], section_id=r["section_id"],
                ord=r["ord"], title=r["title"], source_uri=r["source_uri"],
                section_path=r["section_path"], page_start=r["page_start"],
                page_end=r["page_end"], kind=r["kind"], body=r["body"],
                fused_score=r["score"], vec_rank=r["vec_rank"], kw_rank=r["kw_rank"],
            )
            for r in rows
        ]

    async def fetch_sections(self, section_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not section_ids:
            return {}
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, path, content, tokens, page_start, page_end FROM sections"
                " WHERE id = ANY(%s)",
                (section_ids,),
            )
            return {r["id"]: r for r in await cur.fetchall()}

    async def fetch_window(self, section_id: int, ord_: int, window: int = 1) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT ord, body, page_start, page_end FROM chunks"
                " WHERE section_id = %s AND ord BETWEEN %s AND %s ORDER BY ord",
                (section_id, ord_ - window, ord_ + window),
            )
            return await cur.fetchall()

    # ------------------------------------------------------------ jobs

    async def enqueue_job(self, spec: DocumentSpec, created_by: str) -> UUID:
        job_id = uuid4()
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO ingestion_jobs (id, status, spec, created_by)"
                " VALUES (%s, 'queued', %s, %s)",
                (job_id, Jsonb(spec.model_dump()), created_by),
            )
        return job_id

    async def claim_job(self) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE ingestion_jobs SET status = 'running', attempts = attempts + 1,"
                " updated_at = now() WHERE id = ("
                "  SELECT id FROM ingestion_jobs WHERE status = 'queued'"
                "  ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)"
                " RETURNING id, spec, attempts"
            )
            return await cur.fetchone()

    async def finish_job(
        self, job_id: UUID, *, result: dict | None = None, error: str | None = None,
        retry: bool = False,
    ) -> None:
        status = "done" if error is None else ("queued" if retry else "failed")
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status = %s, result = %s, error = %s,"
                " updated_at = now() WHERE id = %s",
                (status, Jsonb(result) if result is not None else None, error, job_id),
            )

    async def requeue_stale_jobs(self, older_than_s: int, max_attempts: int) -> int:
        """Jobs left 'running' by a crashed worker go back to the queue (or fail for good)."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE ingestion_jobs SET status = CASE WHEN attempts < %s THEN 'queued'"
                " ELSE 'failed' END, error = coalesce(error, 'worker lost the job'),"
                " updated_at = now()"
                " WHERE status = 'running' AND updated_at < now() - make_interval(secs => %s)",
                (max_attempts, older_than_s),
            )
            return cur.rowcount

    async def get_job(self, job_id: UUID) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, status, attempts, result, error, created_by, created_at, updated_at"
                " FROM ingestion_jobs WHERE id = %s",
                (job_id,),
            )
            return await cur.fetchone()

    # ------------------------------------------------------------ query log

    async def log_query(self, row: dict[str, Any]) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO query_log (request_id, user_id, question, question_hash, abstained,"
                " top_score, n_citations, invalid_citations, prompt_version, chat_model,"
                " input_tokens, output_tokens, cached_tokens, timings_ms) VALUES"
                " (%(request_id)s, %(user_id)s, %(question)s, %(question_hash)s, %(abstained)s,"
                " %(top_score)s, %(n_citations)s, %(invalid_citations)s, %(prompt_version)s,"
                " %(chat_model)s, %(input_tokens)s, %(output_tokens)s, %(cached_tokens)s,"
                " %(timings_ms)s)",
                {**row, "timings_ms": Jsonb(row["timings_ms"])},
            )

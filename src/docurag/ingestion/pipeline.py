"""Incremental ingestion: skip what hasn't changed, re-embed only new text, swap atomically.

Permanent problems (unsupported file, unreadable document, bad spec) come back as a
'failed' result. Transient ones (embedding API, database) raise, so a job can be retried.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from time import perf_counter

from langchain_core.embeddings import Embeddings

from docurag.config import Settings
from docurag.db.repository import Repository
from docurag.ingestion.chunker import ChunkingConfig, chunk_document, chunk_stats
from docurag.ingestion.manifest import Manifest
from docurag.ingestion.parsers import UnsupportedDocument, parse_document
from docurag.models import ChunkDraft, DocumentSpec, IngestResult, sha256

log = logging.getLogger(__name__)

# Bump when parsing or chunking logic changes: every document is then re-processed once.
PIPELINE_VERSION = "ingest-v1"
EMBED_BATCH = 64


def batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class IngestionPipeline:
    def __init__(self, repo: Repository, embeddings: Embeddings, settings: Settings) -> None:
        self.repo = repo
        self.embeddings = embeddings
        self.settings = settings
        self.chunking = ChunkingConfig(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            table_max_tokens=settings.table_max_tokens,
        )

    def _signature(self, spec: DocumentSpec) -> str:
        """Everything that changes the indexed output, besides the file bytes."""
        c = self.chunking
        return "|".join(
            str(part)
            for part in (
                PIPELINE_VERSION, c.target_tokens, c.max_tokens, c.table_max_tokens,
                spec.key, spec.version, spec.title, spec.language, spec.doc_type,
                ",".join(spec.record_text_fields), spec.record_id_field,
                self.settings.ocr_backend,
            )
        )

    async def ingest(self, spec: DocumentSpec, *, force: bool = False) -> IngestResult:
        started = perf_counter()
        try:
            raw = await asyncio.to_thread(Path(spec.path).read_bytes)
        except OSError as exc:
            return IngestResult(source_uri=spec.source_uri, status="failed", error=str(exc))
        content_hash = sha256(sha256(raw) + "|" + self._signature(spec))

        existing = await self.repo.get_document_by_uri(spec.source_uri)
        if (
            existing
            and not force
            and existing["content_hash"] == content_hash
            and existing["embedding_model"] == self.settings.embedding_model
        ):
            if sorted(existing["acl_groups"]) != sorted(spec.acl_groups) or (
                existing["metadata"] != spec.metadata
            ):
                await self.repo.update_access(existing["id"], spec.acl_groups, spec.metadata)
                return IngestResult(
                    source_uri=spec.source_uri, status="acl_updated", document_id=existing["id"]
                )
            return IngestResult(
                source_uri=spec.source_uri, status="unchanged", document_id=existing["id"]
            )

        try:
            # Parsing is CPU-bound: keep it off the event loop.
            parsed, doc_type = await asyncio.to_thread(parse_document, spec, self.settings)
        except (UnsupportedDocument, ValueError, RuntimeError) as exc:
            return IngestResult(source_uri=spec.source_uri, status="failed", error=str(exc))
        except Exception as exc:  # corrupt files make parsers raise all sorts of things
            log.exception("parsing failed for %s", spec.source_uri)
            return IngestResult(
                source_uri=spec.source_uri, status="failed", error=f"unreadable document: {exc}"
            )

        sections, chunks = chunk_document(parsed.title, parsed.elements, self.chunking)
        report = {
            "title": parsed.title,
            "doc_type": doc_type,
            "sections": len(sections),
            **parsed.stats,
            **chunk_stats(chunks),
            "warnings": parsed.warnings,
        }
        if not chunks:
            return IngestResult(
                source_uri=spec.source_uri, status="failed", report=report,
                error="no indexable text (scanned document without OCR?)",
            )

        vectors, computed = await self._embed(chunks)
        report["embeddings_computed"] = computed
        report["embeddings_cached"] = len(vectors) - computed
        report["duration_ms"] = round((perf_counter() - started) * 1000)

        document_id = await self.repo.write_document(
            spec=spec, doc_type=doc_type, title=parsed.title, content_hash=content_hash,
            report=report, sections=sections, chunks=chunks, embeddings=vectors,
        )
        return IngestResult(
            source_uri=spec.source_uri, status="indexed", document_id=document_id, report=report
        )

    async def _embed(self, chunks: list[ChunkDraft]) -> tuple[dict[str, list[float]], int]:
        model, dim = self.settings.embedding_model, self.settings.embedding_dimensions
        texts = {c.content_hash: c.content for c in chunks}
        cached = await self.repo.cached_embeddings(list(texts), model)
        missing = [h for h in texts if h not in cached]
        computed: dict[str, list[float]] = {}
        for batch in batched(missing, EMBED_BATCH):
            vectors = await self.embeddings.aembed_documents([texts[h] for h in batch])
            for content_hash, vector in zip(batch, vectors, strict=True):
                if len(vector) != dim:
                    raise ValueError(
                        f"embedding model returned {len(vector)} dimensions, the index expects {dim}"
                    )
                computed[content_hash] = vector
        await self.repo.store_embeddings(computed, model)
        return {**cached, **computed}, len(computed)

    async def sync(
        self, manifest: Manifest, *, prune: bool = False
    ) -> tuple[list[IngestResult], list[str]]:
        """Bring the index in line with the manifest; optionally delete what left it."""
        results: list[IngestResult] = []
        for spec in manifest.documents:
            try:
                result = await self.ingest(spec)
            except Exception as exc:
                log.exception("ingestion failed for %s", spec.source_uri)
                result = IngestResult(source_uri=spec.source_uri, status="failed", error=str(exc))
            results.append(result)
            log.info("%s: %s", spec.source_uri, result.status)
        removed: list[str] = []
        if prune:
            removed = await self.repo.delete_missing(
                manifest.uri_prefix, {s.source_uri for s in manifest.documents}
            )
        return results, removed

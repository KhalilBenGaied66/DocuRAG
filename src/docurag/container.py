"""Wires the components together once, for the API, the worker, the CLI and the eval runner."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from psycopg_pool import AsyncConnectionPool

from docurag.config import Settings, get_settings
from docurag.db.pool import init_schema, open_pool, supports_iterative_scan
from docurag.db.repository import Repository
from docurag.ingestion.pipeline import IngestionPipeline
from docurag.providers import build_chat_model, build_embeddings
from docurag.retrieval.rerank import build_reranker
from docurag.retrieval.retriever import Retriever
from docurag.service import RAGService


@dataclass(slots=True)
class Container:
    settings: Settings
    pool: AsyncConnectionPool
    repo: Repository
    embeddings: Embeddings
    chat: BaseChatModel
    retriever: Retriever
    service: RAGService
    pipeline: IngestionPipeline

    async def close(self) -> None:
        await self.pool.close()


async def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    pool = await open_pool(settings)
    try:
        version = await init_schema(pool, settings)
    except Exception:
        await pool.close()
        raise
    repo = Repository(pool, settings, iterative_scan=supports_iterative_scan(version))
    embeddings = build_embeddings(settings)
    chat = build_chat_model(settings)
    retriever = Retriever(repo, embeddings, build_reranker(settings), settings)
    service = RAGService(settings, repo, retriever, chat, build_chat_model(settings, fast=True))
    pipeline = IngestionPipeline(repo, embeddings, settings)
    return Container(settings, pool, repo, embeddings, chat, retriever, service, pipeline)

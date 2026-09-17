"""Builds the agent on top of DocuRAG's components, sharing its database and models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from docurag.config import Settings
from docurag.container import Container, build_container
from docurag.db.pool import open_pool
from maintagent.business import BusinessRepository
from maintagent.config import INPROCESS_ERP, AgentSettings, get_agent_settings
from maintagent.erp import ErpClient
from maintagent.graph import AgentDeps, build_graph
from maintagent.mock_erp import create_mock_erp
from maintagent.service import MaintAgentService


@dataclass(slots=True)
class AgentContainer:
    core: Container
    agent: MaintAgentService
    business: BusinessRepository
    erp: ErpClient
    checkpoint_pool: AsyncConnectionPool

    # DocuRAG's routes read these attributes from the container.
    @property
    def settings(self) -> Settings:
        return self.core.settings

    @property
    def pool(self) -> AsyncConnectionPool:
        return self.core.pool

    @property
    def repo(self) -> Any:
        return self.core.repo

    @property
    def service(self) -> Any:
        return self.core.service

    async def close(self) -> None:
        await self.erp.aclose()
        await self.checkpoint_pool.close()
        await self.core.close()


def build_erp_client(settings: AgentSettings) -> ErpClient:
    """The real ERP over HTTP, or — by default, for development — the mock served in-process,
    so there is no second process to start. The client code path (retries, timeouts) is the same."""
    common = {"timeout_s": settings.erp_timeout_s, "retries": settings.erp_retries}
    if settings.erp_base_url == INPROCESS_ERP:
        transport = httpx.ASGITransport(app=create_mock_erp())
        return ErpClient("http://mock-erp.local", transport=transport, **common)
    return ErpClient(settings.erp_base_url, **common)


async def build_agent_container(
    settings: Settings | None = None, agent_settings: AgentSettings | None = None
) -> AgentContainer:
    agent_settings = agent_settings or get_agent_settings()
    core = await build_container(settings)
    checkpoint_pool: AsyncConnectionPool | None = None
    erp: ErpClient | None = None
    try:
        business = BusinessRepository(core.pool)
        await business.init_schema()
        # The checkpointer gets its own pool with the connection options LangGraph expects
        # (autocommit, dict rows, no prepared statements), same schema and connection checks.
        checkpoint_pool = await open_pool(
            core.settings, max_size=agent_settings.checkpoint_pool_max, prepare_threshold=0
        )
        checkpointer = AsyncPostgresSaver(checkpoint_pool)
        await checkpointer.setup()
        erp = build_erp_client(agent_settings)
        deps = AgentDeps(
            chat=core.chat, fast_chat=core.service.fast_chat, retriever=core.retriever,
            rag=core.service, business=business, erp=erp, settings=agent_settings,
        )
        agent = MaintAgentService(build_graph(deps, checkpointer), agent_settings)
    except Exception:
        if erp is not None:
            await erp.aclose()
        if checkpoint_pool is not None:
            await checkpoint_pool.close()
        await core.close()
        raise
    return AgentContainer(core, agent, business, erp, checkpoint_pool)

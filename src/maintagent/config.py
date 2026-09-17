"""Agent settings, read from MAINTAGENT_* environment variables or the shared .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

INPROCESS_ERP = "inprocess"  # erp_base_url value: serve the mock ERP inside the app


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAINTAGENT_", env_file=".env", extra="ignore")

    # Bounds: an agent loop must end, whatever the model does.
    max_steps: int = Field(default=8, ge=1, le=20)  # reasoning turns per attempt
    token_budget: int = Field(default=40_000, ge=1_000)  # per ticket, all attempts
    max_rejections: int = Field(default=2, ge=0, le=5)  # supervisor send-backs before giving up
    recursion_limit: int = 60  # LangGraph super-steps, a last-resort guard
    tool_timeout_s: float = 30.0  # local models: a search embeds the query and reranks on this machine
    max_parallel_tool_calls: int = 4

    # Spare-parts ERP. "inprocess" serves the mock inside the app; a real deployment sets the
    # client's ERP URL (or http://127.0.0.1:8081 for `maintagent erp` in its own process).
    erp_base_url: str = INPROCESS_ERP
    erp_timeout_s: float = 5.0
    erp_retries: int = 2

    supervisor_group: str = "maintenance-supervisor"
    checkpoint_pool_max: int = 5


@lru_cache(maxsize=1)
def get_agent_settings() -> AgentSettings:
    return AgentSettings()

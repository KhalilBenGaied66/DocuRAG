"""Typed settings, read from DOCURAG_* environment variables or a .env file."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Rendered into SQL (generated tsvector column), so it must come from a closed list.
TEXT_SEARCH_CONFIGS = frozenset(
    {"simple", "french", "english", "german", "spanish", "italian", "portuguese", "dutch"}
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCURAG_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    database_url: str = "postgresql://docurag@localhost:5432/docurag"
    db_schema: str | None = None  # tables go in this schema instead of public
    db_pool_min: int = 1
    db_pool_max: int = 10
    db_connect_timeout_s: float = 30.0  # hosted databases that scale to zero need a few seconds
    db_check_connection: bool = True  # verify a pooled connection before use (idle ones get closed)

    # Models. With provider=azure, the *_model values are deployment names.
    provider: Literal["azure", "openai", "ollama", "fake"] = "azure"
    azure_endpoint: str | None = None
    azure_api_version: str = "2024-10-21"
    azure_use_entra_id: bool = False
    api_key: SecretStr | None = None
    chat_model: str = "gpt-4.1"
    fast_model: str = "gpt-4.1-mini"
    chat_temperature: float | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    # Local models (provider=ollama): free, private, no rate limits
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192  # Ollama's default window is small: sources would be cut silently

    # Retrieval
    text_search_config: str = "french"
    candidates_per_leg: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 6
    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    abstain_threshold: float | None = None
    hnsw_ef_search: int = 100

    # Chunking and context assembly (tokens)
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 600
    table_max_tokens: int = 1500
    parent_max_tokens: int = 1500
    context_budget_tokens: int = 6000

    # Ingestion
    storage_dir: str = "var/uploads"
    ocr_backend: Literal["none", "azure-di"] = "none"
    azure_di_endpoint: str | None = None
    min_text_chars_per_page: int = 50
    worker_poll_interval_s: float = 2.0
    job_max_attempts: int = 3

    # API and auth
    auth_mode: Literal["jwt", "dev"] = "jwt"
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_jwks_url: str | None = None
    jwt_groups_claim: str = "groups"
    admin_group: str = "docurag-admin"
    log_questions: bool = False
    abstain_message: str = (
        "Je ne trouve pas la réponse dans la documentation à laquelle vous avez accès."
    )

    @field_validator("database_url")
    @classmethod
    def _postgres_url(cls, value: str) -> str:
        value = value.strip()
        is_url = value.startswith(("postgresql://", "postgres://"))
        is_keyword_form = "=" in value and "://" not in value  # libpq: "host=… dbname=…"
        if not (is_url or is_keyword_form):
            raise ValueError(
                "DOCURAG_DATABASE_URL must be a PostgreSQL connection string (postgresql://…); "
                "paste the one from your Neon dashboard"
            )
        return value

    @field_validator("db_schema")
    @classmethod
    def _schema_identifier(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", value):
            raise ValueError("db_schema must be a lowercase identifier (letters, digits, _)")
        return value

    @field_validator("text_search_config")
    @classmethod
    def _known_ts_config(cls, value: str) -> str:
        if value not in TEXT_SEARCH_CONFIGS:
            raise ValueError(f"text_search_config must be one of {sorted(TEXT_SEARCH_CONFIGS)}")
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def _indexable_dimensions(cls, value: int) -> int:
        # pgvector's HNSW index supports halfvec up to 4,000 dimensions.
        if not 1 <= value <= 4000:
            raise ValueError("embedding_dimensions must be between 1 and 4000 (halfvec HNSW limit)")
        return value

    @model_validator(mode="after")
    def _safe_combinations(self) -> Settings:
        if self.env == "prod" and self.auth_mode == "dev":
            raise ValueError("auth_mode=dev trusts client headers and is refused when env=prod")
        if self.env == "prod" and self.provider == "fake":
            raise ValueError("provider=fake is for tests only")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("chunk_target_tokens must not exceed chunk_max_tokens")
        if self.provider == "ollama" and self.ollama_num_ctx < self.context_budget_tokens + 1024:
            raise ValueError(
                "ollama_num_ctx must exceed context_budget_tokens + ~1,024 tokens for the prompt "
                "and the answer, or Ollama truncates the sources without any error"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

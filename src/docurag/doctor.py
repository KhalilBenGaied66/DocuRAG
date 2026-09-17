"""`docurag doctor`: checks the configuration and every dependency, and says how to fix each problem.

Run it from the folder that holds .env, after any change to the setup.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import psycopg
from psycopg import sql
from pydantic import BaseModel, Field, ValidationError

from docurag.config import Settings
from docurag.db.pool import MIN_PGVECTOR, _version_tuple, supports_iterative_scan

Status = Literal["ok", "warn", "fail", "skip"]
MARK = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "skip": "[skip]"}


@dataclass(slots=True)
class Check:
    area: str
    status: Status
    detail: str
    hint: str | None = None


def _ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def _short(exc: BaseException, limit: int = 220) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:limit]


# ------------------------------------------------------------------ configuration


def check_env_file(cwd: Path) -> Check:
    if (cwd / ".env").exists():
        return Check("config", "ok", f".env found in {cwd}")
    return Check(
        "config",
        "warn",
        f"no .env in {cwd}: only environment variables are used",
        "run the commands from the docurag folder, or copy .env.example to .env",
    )


UNSET_DATABASE_URL = "postgresql://not-set@not-set.invalid/not-set"


def load_settings(factory: Callable[..., Settings]) -> tuple[Settings | None, Check]:
    try:
        settings = factory()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'settings'}: {e['msg'].removeprefix('Value error, ')}"
            for e in exc.errors()
        )
        failure = Check("config", "fail", problems, "fix these values in .env")
        only_database = all(e["loc"] == ("database_url",) for e in exc.errors())
        if only_database and isinstance(factory, type):
            # Keep checking the rest (Ollama, reranker…): the database may simply not be set up yet.
            return factory(database_url=UNSET_DATABASE_URL), failure
        return None, failure
    detail = (
        f"provider={settings.provider}, chat={settings.chat_model}, "
        f"embeddings={settings.embedding_model} ({settings.embedding_dimensions} d), "
        f"reranker={settings.reranker}"
    )
    return settings, Check("config", "ok", detail)


def check_consistency(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    if settings.provider == "fake":
        checks.append(
            Check(
                "models",
                "warn",
                "provider=fake: offline stand-ins, fine for plumbing, meaningless for quality",
                "set DOCURAG_PROVIDER=ollama for real answers",
            )
        )
    if settings.abstain_threshold is not None and settings.reranker == "none":
        checks.append(
            Check(
                "retrieval",
                "warn",
                "DOCURAG_ABSTAIN_THRESHOLD is ignored without a reranker",
                "enable DOCURAG_RERANKER=cross-encoder, or clear the threshold",
            )
        )
    if settings.provider == "ollama" and settings.chat_temperature is None:
        checks.append(
            Check(
                "models",
                "warn",
                "no chat temperature set: Ollama models default to ~0.8 (less stable answers)",
                "set DOCURAG_CHAT_TEMPERATURE=0",
            )
        )
    return checks


# ------------------------------------------------------------------ database


async def check_database(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    host = urlsplit(settings.database_url).hostname or ""
    if "-pooler" in host:
        checks.append(
            Check(
                "database",
                "warn",
                f"{host} is Neon's pooled endpoint",
                "use the direct connection string (without -pooler): the app pools its own connections",
            )
        )
    started = perf_counter()
    try:
        conn = await asyncio.wait_for(
            psycopg.AsyncConnection.connect(settings.database_url, autocommit=True),
            timeout=settings.db_connect_timeout_s,
        )
    except TimeoutError:
        checks.append(
            Check(
                "database",
                "fail",
                f"no answer from {host or 'the database'} in {settings.db_connect_timeout_s:.0f} s",
                "check the host in DOCURAG_DATABASE_URL; a suspended Neon database wakes up "
                "in a few seconds, retry",
            )
        )
        return checks
    except psycopg.Error as exc:
        message = _short(exc)
        hint = "check DOCURAG_DATABASE_URL"
        if "password" in message.lower() or "authentication" in message.lower():
            hint = "the user or password in DOCURAG_DATABASE_URL is wrong: copy the string again from Neon"
        elif "translate host" in message.lower() or "resolve" in message.lower():
            hint = "the host name in DOCURAG_DATABASE_URL is wrong, or there is no internet connection"
        elif "does not exist" in message.lower():
            hint = "the database name at the end of DOCURAG_DATABASE_URL doesn't exist"
        checks.append(Check("database", "fail", f"cannot connect: {message}", hint))
        return checks

    async with conn:
        version = (await (await conn.execute("SHOW server_version")).fetchone())[0]
        checks.append(Check("database", "ok", f"connected to PostgreSQL {version} in {_ms(started)} ms"))
        if settings.db_schema:
            await conn.execute(
                sql.SQL("SET search_path TO {}, public").format(sql.Identifier(settings.db_schema))
            )

        installed = await (
            await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()
        if installed:
            found = _version_tuple(installed[0])
            if found < MIN_PGVECTOR:
                checks.append(
                    Check(
                        "pgvector",
                        "fail",
                        f"pgvector {installed[0]} is too old (0.7+ needed)",
                        "ALTER EXTENSION vector UPDATE;",
                    )
                )
            elif not supports_iterative_scan(found):
                checks.append(
                    Check(
                        "pgvector",
                        "warn",
                        f"pgvector {installed[0]}: no iterative index scans",
                        "0.8+ keeps filtered searches complete: ALTER EXTENSION vector UPDATE;",
                    )
                )
            else:
                checks.append(Check("pgvector", "ok", f"pgvector {installed[0]}"))
        else:
            available = await (
                await conn.execute(
                    "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
                )
            ).fetchone()
            if available:
                checks.append(
                    Check(
                        "pgvector", "ok", f"pgvector {available[0]} available, created by `docurag init-db`"
                    )
                )
            else:
                checks.append(
                    Check(
                        "pgvector",
                        "fail",
                        "pgvector is not available on this server",
                        "use a server that ships pgvector (Neon does)",
                    )
                )
                return checks

        has_schema = (await (await conn.execute("SELECT to_regclass('schema_info') IS NOT NULL")).fetchone())[
            0
        ]
        if not has_schema:
            checks.append(Check("index", "warn", "schema not created yet", "run: docurag init-db"))
            return checks
        stored = dict(await (await conn.execute("SELECT key, value FROM schema_info")).fetchall())
        expected = {
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": str(settings.embedding_dimensions),
            "text_search_config": settings.text_search_config,
        }
        diffs = [
            f"{k}: index={stored.get(k)!r}, .env={v!r}" for k, v in expected.items() if stored.get(k) != v
        ]
        if diffs:
            checks.append(
                Check(
                    "index",
                    "fail",
                    "index built with other settings: " + "; ".join(diffs),
                    "restore the previous values in .env, or use a new database "
                    "(or DOCURAG_DB_SCHEMA) for the new index",
                )
            )
            return checks
        docs, chunks = await (
            await conn.execute("SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)")
        ).fetchone()
        if chunks == 0:
            checks.append(
                Check(
                    "index", "warn", "the index is empty", "run: docurag sync examples/corpus/manifest.toml"
                )
            )
        else:
            checks.append(Check("index", "ok", f"{docs} documents, {chunks} chunks"))

        if (await (await conn.execute("SELECT to_regclass('equipment') IS NOT NULL")).fetchone())[0]:
            n = (await (await conn.execute("SELECT count(*) FROM equipment")).fetchone())[0]
            if n == 0:
                checks.append(
                    Check(
                        "agent-data", "warn", "no equipment loaded", "run: maintagent seed examples/business"
                    )
                )
            else:
                checks.append(Check("agent-data", "ok", f"{n} equipment records"))
    return checks


# ------------------------------------------------------------------ models


class SearchDocs(BaseModel):
    """Recherche dans la documentation technique."""

    query: str = Field(description="Question de recherche")


def installed_ollama_models(tags_payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for model in tags_payload.get("models", []):
        name = model.get("name") or model.get("model") or ""
        names.add(name)
        if name.endswith(":latest"):
            names.add(name.removesuffix(":latest"))
    return names


async def check_ollama_server(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> list[Check]:
    wanted = sorted({settings.chat_model, settings.fast_model, settings.embedding_model})
    try:
        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=5, transport=transport
        ) as client:
            version = (await client.get("/api/version")).json().get("version", "?")
            tags = (await client.get("/api/tags")).json()
    except (httpx.HTTPError, ValueError) as exc:
        return [
            Check(
                "ollama",
                "fail",
                f"Ollama not reachable at {settings.ollama_base_url} ({_short(exc, 80)})",
                "start the Ollama app (it runs in the background once installed)",
            )
        ]
    installed = installed_ollama_models(tags)
    missing = [m for m in wanted if m not in installed]
    if missing:
        return [
            Check(
                "ollama",
                "fail",
                f"Ollama {version} running, missing model(s): {', '.join(missing)}",
                "run: " + " ; ".join(f"ollama pull {m}" for m in missing),
            )
        ]
    return [Check("ollama", "ok", f"Ollama {version}, models present: {', '.join(wanted)}")]


async def check_models(settings: Settings, timeout_s: float = 180.0) -> list[Check]:
    from docurag.providers import build_chat_model, build_embeddings

    checks: list[Check] = []
    try:
        embeddings = build_embeddings(settings)
        started = perf_counter()
        vector = await asyncio.wait_for(embeddings.aembed_query("alarme E-41 pompe"), timeout_s)
        if len(vector) != settings.embedding_dimensions:
            checks.append(
                Check(
                    "embeddings",
                    "fail",
                    f"{settings.embedding_model} returns {len(vector)} dimensions, "
                    f"settings say {settings.embedding_dimensions}",
                    f"set DOCURAG_EMBEDDING_DIMENSIONS={len(vector)} (before `docurag init-db`)",
                )
            )
        else:
            checks.append(Check("embeddings", "ok", f"{len(vector)} dimensions in {_ms(started)} ms"))
    except Exception as exc:
        checks.append(
            Check(
                "embeddings",
                "fail",
                f"embedding call failed: {_short(exc)}",
                "check the embedding model name and the provider settings",
            )
        )

    try:
        chat = build_chat_model(settings)
        started = perf_counter()
        reply = await asyncio.wait_for(chat.ainvoke("Réponds uniquement : OK"), timeout_s)
        text = " ".join(str(reply.content).split())[:40]
        checks.append(
            Check(
                "chat",
                "ok",
                f"{settings.chat_model} answered {text!r} in {_ms(started)} ms "
                "(the first call also loads the model)",
            )
        )
    except Exception as exc:
        checks.append(
            Check(
                "chat",
                "fail",
                f"chat call failed: {_short(exc)}",
                "check the chat model name and the provider settings",
            )
        )
        return checks

    if settings.provider == "fake":
        checks.append(Check("tools", "skip", "tool calling not checked for the offline fake model"))
        return checks
    try:
        started = perf_counter()
        reply = await asyncio.wait_for(
            chat.bind_tools([SearchDocs]).ainvoke(
                "Utilise l'outil SearchDocs pour chercher la signification de l'alarme E-41."
            ),
            timeout_s,
        )
        if getattr(reply, "tool_calls", None):
            checks.append(Check("tools", "ok", f"tool calling works ({_ms(started)} ms)"))
        else:
            checks.append(
                Check(
                    "tools",
                    "warn",
                    "the model answered without calling the tool",
                    "MaintAgent relies on tool calls: expect weaker diagnoses; "
                    "try another model (e.g. qwen2.5:14b)",
                )
            )
    except Exception as exc:
        checks.append(
            Check(
                "tools",
                "fail",
                f"tool calling failed: {_short(exc)}",
                "the model may not support tools in Ollama; pick a model with the 'tools' tag",
            )
        )
    return checks


def check_reranker(settings: Settings) -> list[Check]:
    if settings.reranker == "none":
        return [Check("reranker", "skip", "disabled: fused ranking only, no score-based abstention")]
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401  (imported by sentence-transformers anyway)
    except ImportError:
        return [
            Check(
                "reranker",
                "fail",
                "sentence-transformers is not installed",
                'pip install -e ".[rerank]" (GPU: install PyTorch with CUDA first, see README)',
            )
        ]
    import torch

    if torch.cuda.is_available():
        device = f"GPU {torch.cuda.get_device_name(0)}"
        status: Status = "ok"
        hint = None
    else:
        device = "CPU only"
        status = "warn"
        hint = "reranking 50 candidates on CPU takes seconds: install the CUDA build of PyTorch (README)"
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = isinstance(try_to_load_from_cache(settings.reranker_model, "config.json"), str)
    except Exception:
        cached = False
    detail = f"{settings.reranker_model} on {device}"
    if not cached:
        detail += "; model not downloaded yet (~2.3 GB on first use)"
    return [Check("reranker", status, detail, hint)]


# ------------------------------------------------------------------ agent


async def check_agent() -> list[Check]:
    try:
        from maintagent.config import INPROCESS_ERP, AgentSettings
    except ImportError:
        return []
    try:
        agent = AgentSettings()
    except ValidationError as exc:
        return [Check("agent", "fail", _short(exc), "fix the MAINTAGENT_* values in .env")]
    if agent.erp_base_url == INPROCESS_ERP:
        return [
            Check(
                "agent",
                "ok",
                "mock ERP served in-process; "
                f"max {agent.max_steps} steps, {agent.token_budget} tokens per ticket",
            )
        ]
    try:
        async with httpx.AsyncClient(base_url=agent.erp_base_url, timeout=agent.erp_timeout_s) as client:
            (await client.get("/v1/stock", params={"refs": "GM-45"})).raise_for_status()
    except httpx.HTTPError as exc:
        return [
            Check(
                "agent",
                "fail",
                f"ERP not reachable at {agent.erp_base_url} ({_short(exc, 80)})",
                "start `maintagent erp`, or set MAINTAGENT_ERP_BASE_URL=inprocess",
            )
        ]
    return [Check("agent", "ok", f"ERP reachable at {agent.erp_base_url}")]


def check_test_database(env_file: Path) -> Check:
    from dotenv import dotenv_values

    value = os.environ.get("DOCURAG_TEST_DATABASE_URL")
    if not value and env_file.exists():
        value = dotenv_values(env_file).get("DOCURAG_TEST_DATABASE_URL")
    if value:
        return Check("tests", "ok", "DOCURAG_TEST_DATABASE_URL set: `pytest -m db` runs the database tests")
    return Check(
        "tests",
        "warn",
        "DOCURAG_TEST_DATABASE_URL not set: database tests will be skipped",
        "set it in .env (the same Neon string works: tests use throwaway schemas)",
    )


# ------------------------------------------------------------------ runner


async def run_doctor(
    settings_factory: Callable[[], Settings],
    *,
    cwd: Path,
    check_model_calls: bool = True,
    ollama_transport: httpx.AsyncBaseTransport | None = None,
) -> list[Check]:
    checks = [check_env_file(cwd)]
    settings, config_check = load_settings(settings_factory)
    checks.append(config_check)
    if settings is None:
        return checks
    checks += check_consistency(settings)
    if settings.database_url == UNSET_DATABASE_URL:
        checks.append(Check("database", "skip", "not checked until DOCURAG_DATABASE_URL is set"))
    else:
        checks += await check_database(settings)

    models_reachable = True
    if settings.provider == "ollama":
        server = await check_ollama_server(settings, ollama_transport)
        checks += server
        models_reachable = all(c.status != "fail" for c in server)
    if not check_model_calls:
        checks.append(Check("models", "skip", "model calls skipped (--skip-models)"))
    elif models_reachable:
        checks += await check_models(settings)
    checks += check_reranker(settings)
    checks += await check_agent()
    checks.append(check_test_database(cwd / ".env"))
    return checks


def render(checks: list[Check]) -> str:
    lines = ["DocuRAG doctor", ""]
    width = max(len(c.area) for c in checks)
    for c in checks:
        lines.append(f"{MARK[c.status]} {c.area:<{width}}  {c.detail}")
        if c.hint and c.status in ("warn", "fail"):
            lines.append(f"{'':>6} {'':<{width}}  -> {c.hint}")
    failures = sum(c.status == "fail" for c in checks)
    warnings = sum(c.status == "warn" for c in checks)
    lines += [
        "",
        "All good." if not failures and not warnings else f"{failures} problem(s), {warnings} warning(s).",
    ]
    return "\n".join(lines)


async def doctor_main(settings_factory: Callable[[], Settings], *, check_model_calls: bool = True) -> int:
    checks = await run_doctor(settings_factory, cwd=Path.cwd(), check_model_calls=check_model_calls)
    print(render(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0

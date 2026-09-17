from __future__ import annotations

import httpx
import pytest

from docurag.config import Settings
from docurag.doctor import check_ollama_server, installed_ollama_models, render, run_doctor
from maintagent.config import AgentSettings
from maintagent.container import build_erp_client
from tests.conftest import TEST_DATABASE_URL


def by_area(checks):
    return {c.area: c for c in reversed(checks)}  # first check of an area wins


async def test_invalid_configuration_stops_with_a_readable_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCURAG_DATABASE_URL", "")
    checks = await run_doctor(lambda: Settings(_env_file=None), cwd=tmp_path)
    config = [c for c in checks if c.area == "config"]
    assert config[0].status == "warn"  # no .env in that folder
    assert config[1].status == "fail" and "Neon" in config[1].detail
    assert "[FAIL] config" in render(checks)


async def test_unreachable_database_and_ollama_explain_the_fix(tmp_path):
    settings = Settings(
        _env_file=None,
        env="test",
        provider="ollama",
        database_url="postgresql://u:p@127.0.0.1:1/db",
        chat_temperature=0,
        db_connect_timeout_s=5,
        ollama_base_url="http://127.0.0.1:1",
        chat_model="mistral-nemo",
        fast_model="mistral-nemo",
        embedding_model="bge-m3",
        embedding_dimensions=1024,
    )
    checks = await run_doctor(lambda: settings, cwd=tmp_path)
    areas = by_area(checks)
    assert areas["database"].status == "fail" and areas["database"].hint
    assert areas["ollama"].status == "fail" and "Ollama app" in areas["ollama"].hint
    assert "embeddings" not in areas  # no model calls when the server is down


async def test_ollama_missing_models_are_named_with_the_pull_command():
    settings = Settings(
        _env_file=None,
        provider="ollama",
        chat_model="mistral-nemo",
        fast_model="mistral-nemo",
        embedding_model="bge-m3",
        embedding_dimensions=1024,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        return httpx.Response(200, json={"models": [{"name": "bge-m3:latest"}]})

    [check] = await check_ollama_server(settings, httpx.MockTransport(handler))
    assert check.status == "fail" and check.hint == "run: ollama pull mistral-nemo"


def test_installed_model_names_accept_the_latest_tag():
    names = installed_ollama_models({"models": [{"name": "bge-m3:latest"}, {"name": "qwen2.5:14b"}]})
    assert {"bge-m3", "bge-m3:latest", "qwen2.5:14b"} <= names and "qwen2.5" not in names


async def test_pooled_neon_url_is_flagged():
    from docurag.doctor import check_database

    # hostaddr skips DNS and port 1 refuses at once: no network needed.
    settings = Settings(
        _env_file=None,
        db_connect_timeout_s=5,
        database_url="postgresql://u:p@ep-cool-1-pooler.eu-central-1.aws.neon.tech:1/db?hostaddr=127.0.0.1",
    )
    warning, failure = await check_database(settings)
    assert warning.status == "warn" and "without -pooler" in warning.hint
    assert failure.status == "fail"


async def test_inprocess_erp_client_serves_the_mock():
    erp = build_erp_client(AgentSettings(_env_file=None))
    try:
        stock = await erp.stock(["GM-45", "NOPE-1"])
    finally:
        await erp.aclose()
    assert stock["GM-45"]["available"] == 2 and stock["NOPE-1"] is None


@pytest.mark.db
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="set DOCURAG_TEST_DATABASE_URL")
async def test_doctor_follows_the_database_through_setup(db_settings, tmp_path):
    from docurag.container import build_container

    before = by_area(await run_doctor(lambda: db_settings, cwd=tmp_path))
    assert before["database"].status == "ok" and before["pgvector"].status == "ok"
    assert before["index"].status == "warn" and "init-db" in before["index"].hint
    assert before["tools"].status == "skip"  # fake provider

    container = await build_container(db_settings)
    await container.close()
    after = by_area(await run_doctor(lambda: db_settings, cwd=tmp_path))
    assert after["index"].status == "warn" and "sync" in after["index"].hint

    other = db_settings.model_copy(update={"embedding_dimensions": 512})
    mismatch = by_area(await run_doctor(lambda: other, cwd=tmp_path, check_model_calls=False))
    assert mismatch["index"].status == "fail" and "embedding_dimensions" in mismatch["index"].detail


async def test_missing_database_url_still_checks_the_models(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCURAG_DATABASE_URL", "")
    monkeypatch.setenv("DOCURAG_PROVIDER", "fake")
    checks = await run_doctor(Settings, cwd=tmp_path)
    areas = by_area(checks)
    assert [c.status for c in checks if c.area == "config"][1] == "fail"
    assert areas["database"].status == "skip"
    assert areas["embeddings"].status == "ok" and areas["chat"].status == "ok"

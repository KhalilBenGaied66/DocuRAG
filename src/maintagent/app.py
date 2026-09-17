"""One service: DocuRAG's endpoints plus the ticket endpoints. Run with `maintagent serve`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI

from docurag.api.app import create_app as create_docurag_app
from maintagent.api import router


def create_app(container_factory: Callable[[], Awaitable[Any]] | None = None) -> FastAPI:
    if container_factory is None:
        from maintagent.container import build_agent_container

        container_factory = build_agent_container
    app = create_docurag_app(container_factory)
    app.title = "DocuRAG · MaintAgent"
    app.include_router(router)
    return app

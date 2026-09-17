"""The agent's tools, designed like public APIs: narrow, typed, documented, compact output.

The model chooses a tool and its arguments; it never chooses *who* it acts for. The user's
groups come from the graph state and are injected here, so the model cannot widen its own
access. Every failure — unknown tool, invalid arguments, timeout, crash — comes back to the
model as an observation instead of ending the run.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from maintagent.models import EQUIPMENT_TAG, PART_REF

log = logging.getLogger(__name__)

EXTRACT_CHARS = 900
MAX_DOC_RESULTS = 5


class RetrieverLike(Protocol):
    async def retrieve(self, query: str, groups: list[str], **kwargs: Any) -> Any: ...


class BusinessLike(Protocol):
    async def equipment(self, tag: str) -> dict[str, Any] | None: ...
    async def interventions(self, tag: str, since_days: int, limit: int = 20) -> list[dict]: ...


class ErpLike(Protocol):
    async def stock(self, refs: list[str]) -> dict[str, dict[str, Any] | None]: ...


@dataclass(slots=True)
class ToolContext:
    user_id: str
    groups: list[str]
    ticket_id: str
    retriever: RetrieverLike
    business: BusinessLike
    erp: ErpLike


@dataclass(slots=True)
class ToolResult:
    content: dict[str, Any]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    error: bool = False


# ------------------------------------------------------------------ arguments


class _Args(BaseModel):
    # Unexpected arguments are refused, not ignored: a model passing e.g. "groups" is told no.
    model_config = ConfigDict(extra="forbid")


class SearchDocsArgs(_Args):
    query: str = Field(
        min_length=3, max_length=300,
        description="Question autonome en français, avec les codes et références exacts",
    )


class EquipmentHistoryArgs(_Args):
    equipment_tag: str = Field(pattern=EQUIPMENT_TAG, description="Repère, ex. P-200-07")
    since_days: int = Field(default=730, ge=1, le=1825, description="Profondeur d'historique en jours")


class PartsStockArgs(_Args):
    part_refs: list[Annotated[str, Field(pattern=PART_REF)]] = Field(
        min_length=1, max_length=10, description="Références exactes, ex. [\"6309-2Z\"]"
    )


# ------------------------------------------------------------------ implementations


def _pages(hit: Any) -> str | None:
    if hit.page_start is None:
        return None
    if hit.page_end in (None, hit.page_start):
        return str(hit.page_start)
    return f"{hit.page_start}-{hit.page_end}"


async def search_docs(args: SearchDocsArgs, ctx: ToolContext) -> ToolResult:
    result = await ctx.retriever.retrieve(args.query, ctx.groups)
    items, evidence = [], []
    for hit in result.hits[:MAX_DOC_RESULTS]:
        source_id = f"doc:{hit.chunk_id}"
        items.append({
            "source_id": source_id, "document": hit.title, "section": hit.section_path,
            "pages": _pages(hit), "extract": hit.body[:EXTRACT_CHARS],
        })
        evidence.append({
            "source_id": source_id, "document_id": str(hit.document_id), "title": hit.title,
            "source_uri": hit.source_uri, "section_path": hit.section_path,
            "page_start": hit.page_start, "page_end": hit.page_end,
        })
    if not items:
        return ToolResult({"result": "no_documents", "query": args.query})
    return ToolResult({"results": items}, evidence)


async def equipment_history(args: EquipmentHistoryArgs, ctx: ToolContext) -> ToolResult:
    equipment = await ctx.business.equipment(args.equipment_tag)
    if equipment is None:
        return ToolResult(
            {"error": "unknown_equipment", "equipment_tag": args.equipment_tag,
             "hint": "vérifier le repère auprès du technicien"},
            error=True,
        )
    history = await ctx.business.interventions(args.equipment_tag, args.since_days)
    return ToolResult({"equipment": equipment, "interventions": history})


async def parts_stock(args: PartsStockArgs, ctx: ToolContext) -> ToolResult:
    stock = await ctx.erp.stock(args.part_refs)
    return ToolResult({
        "parts": [item for item in stock.values() if item is not None],
        "unknown_refs": [ref for ref, item in stock.items() if item is None],
    })


# ------------------------------------------------------------------ registry


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[[Any, ToolContext], Awaitable[ToolResult]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args.model_json_schema(),
            },
        }


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "search_docs",
            "Recherche dans la documentation technique à laquelle l'utilisateur a accès "
            "(manuels, procédures). À utiliser pour les codes alarme, procédures, valeurs et "
            "références de pièces. Renvoie des extraits avec leur identifiant doc:… à citer.",
            SearchDocsArgs, search_docs,
        ),
        ToolSpec(
            "equipment_history",
            "Fiche et historique d'interventions d'UN équipement (lecture seule). À utiliser pour "
            "repérer un défaut récurrent avant de proposer un diagnostic. Ne donne pas le stock.",
            EquipmentHistoryArgs, equipment_history,
        ),
        ToolSpec(
            "parts_stock",
            "Stock des pièces de rechange dans l'ERP, par référence exacte (ex. 6309-2Z). "
            "Uniquement pour des références trouvées dans la documentation.",
            PartsStockArgs, parts_stock,
        ),
    )
}


def tool_schemas() -> list[dict[str, Any]]:
    return [spec.schema() for spec in TOOLS.values()]


async def run_tool_call(
    call: dict[str, Any], ctx: ToolContext, *, timeout_s: float
) -> tuple[ToolResult, float]:
    """Execute one model tool call. Never raises: failures become observations."""
    started = perf_counter()
    spec = TOOLS.get(call.get("name", ""))
    if spec is None:
        result = ToolResult({"error": "unknown_tool", "available": sorted(TOOLS)}, error=True)
        return result, 0.0
    try:
        args = spec.args.model_validate(call.get("args") or {})
    except ValidationError as exc:
        details = [{"field": ".".join(map(str, e["loc"])), "problem": e["msg"]} for e in exc.errors()]
        return ToolResult({"error": "invalid_arguments", "details": details}, error=True), 0.0
    try:
        async with asyncio.timeout(timeout_s):
            result = await spec.fn(args, ctx)
    except TimeoutError:
        result = ToolResult({"error": "timeout", "after_s": timeout_s}, error=True)
    except Exception as exc:
        log.exception("tool %s failed on ticket %s", spec.name, ctx.ticket_id)
        result = ToolResult({"error": "tool_failed", "message": str(exc)[:300]}, error=True)
    return result, round((perf_counter() - started) * 1000, 1)

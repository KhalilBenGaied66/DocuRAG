"""A stand-in for the client's ERP, to develop and demo against. Not part of the product.

MOCK_ERP_FAILURE_RATE=0.3 makes 30% of calls fail with 503, to watch the client's retries work.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

DEFAULT_PARTS = {
    "GM-45": {"designation": "Garniture mécanique P-200", "available": 2, "lead_time_days": 10},
    "JT-80x3": {"designation": "Joint torique de chemise", "available": 12, "lead_time_days": 3},
    "GR-S1": {"designation": "Graisse de montage", "available": 5, "lead_time_days": 2},
    "GR-L2": {"designation": "Graisse lithium NLGI 2", "available": 8, "lead_time_days": 2},
    "6309-2Z": {"designation": "Roulement côté accouplement", "available": 0, "lead_time_days": 7},
    "6307-2Z": {"designation": "Roulement côté opposé", "available": 3, "lead_time_days": 7},
}


def create_mock_erp(parts: dict[str, dict] | None = None, failure_rate: float = 0.0) -> FastAPI:
    stock = parts if parts is not None else DEFAULT_PARTS
    app = FastAPI(title="Mock ERP — spare parts")

    @app.get("/v1/stock")
    async def get_stock(refs: str = Query(..., max_length=500)) -> JSONResponse:
        if failure_rate and random.random() < failure_rate:
            return JSONResponse({"error": "busy"}, status_code=503, headers={"Retry-After": "0.2"})
        wanted = [r.strip() for r in refs.split(",") if r.strip()][:50]
        items = [{"ref": r, "site": "Lyon", **stock[r]} for r in wanted if r in stock]
        unknown = [r for r in wanted if r not in stock]
        return JSONResponse({"items": items, "unknown": unknown})

    return app


def app_from_env() -> FastAPI:
    parts_file = os.environ.get("MOCK_ERP_PARTS_FILE")
    parts = json.loads(Path(parts_file).read_text(encoding="utf-8")) if parts_file else None
    return create_mock_erp(parts, float(os.environ.get("MOCK_ERP_FAILURE_RATE", "0")))

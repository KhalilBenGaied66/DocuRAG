"""Business data: equipment registry, intervention history (read-only) and work orders (GMAO).

Reads go through views, with bound parameters, in READ ONLY transactions: even a bug in a
tool cannot write through this path. The only write is work-order creation, which is
idempotent on the ticket ID — a retried or resumed run can't create a second order.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any
from uuid import uuid4

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from maintagent.models import WorkOrderDraft

BUSINESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS equipment (
  tag             text PRIMARY KEY,
  site            text NOT NULL,
  model           text NOT NULL,
  commissioned_on date,
  running_hours   int,
  criticality     text CHECK (criticality IN ('A', 'B', 'C'))
);
CREATE TABLE IF NOT EXISTS interventions (
  ot_id         text PRIMARY KEY,
  equipment_tag text NOT NULL REFERENCES equipment(tag),
  date          date NOT NULL,
  status        text NOT NULL,
  fault_code    text,
  symptom       text,
  diagnosis     text,
  action        text
);
CREATE INDEX IF NOT EXISTS interventions_tag_date_idx ON interventions (equipment_tag, date DESC);
CREATE OR REPLACE VIEW v_equipment AS
  SELECT tag, site, model, commissioned_on, running_hours, criticality FROM equipment;
CREATE OR REPLACE VIEW v_interventions AS
  SELECT ot_id, equipment_tag, date, status, fault_code, symptom, diagnosis, action
  FROM interventions;
CREATE TABLE IF NOT EXISTS work_orders (
  id            text PRIMARY KEY,
  ticket_id     text NOT NULL UNIQUE,          -- idempotency key
  equipment_tag text NOT NULL,
  priority      text NOT NULL,
  title         text NOT NULL,
  draft         jsonb NOT NULL,
  sources       jsonb NOT NULL,
  requested_by  text NOT NULL,
  approved_by   text NOT NULL,
  status        text NOT NULL DEFAULT 'open',
  created_at    timestamptz NOT NULL DEFAULT now()
);
"""


def _none_if_blank(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    raw = path.read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t")
    return list(csv.DictReader(io.StringIO(raw), dialect=dialect))


class BusinessRepository:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def init_schema(self) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(BUSINESS_SCHEMA)

    async def _read(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            return await (await conn.execute(sql, params)).fetchall()

    async def equipment(self, tag: str) -> dict[str, Any] | None:
        rows = await self._read(
            "SELECT tag, site, model, commissioned_on, running_hours, criticality"
            " FROM v_equipment WHERE tag = %(tag)s",
            {"tag": tag},
        )
        return rows[0] if rows else None

    async def interventions(self, tag: str, since_days: int, limit: int = 20) -> list[dict[str, Any]]:
        return await self._read(
            "SELECT ot_id, date, status, fault_code, symptom, diagnosis, action"
            " FROM v_interventions WHERE equipment_tag = %(tag)s"
            " AND date > current_date - %(days)s::int ORDER BY date DESC LIMIT %(limit)s",
            {"tag": tag, "days": since_days, "limit": limit},
        )

    async def create_work_order(
        self,
        *,
        ticket_id: str,
        draft: WorkOrderDraft,
        requested_by: str,
        approved_by: str,
        sources: list[dict[str, Any]],
    ) -> str:
        async with self.pool.connection() as conn, conn.transaction():
            row = await (
                await conn.execute(
                    "INSERT INTO work_orders (id, ticket_id, equipment_tag, priority, title, draft,"
                    " sources, requested_by, approved_by)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT (ticket_id) DO NOTHING RETURNING id",
                    (
                        f"WO-{uuid4().hex[:10].upper()}", ticket_id, draft.equipment_tag,
                        draft.priority, draft.title, Jsonb(draft.model_dump()), Jsonb(sources),
                        requested_by, approved_by,
                    ),
                )
            ).fetchone()
            if row is None:  # already created for this ticket: return the same order
                row = await (
                    await conn.execute("SELECT id FROM work_orders WHERE ticket_id = %s", (ticket_id,))
                ).fetchone()
            return row["id"]

    async def seed(self, directory: Path) -> dict[str, int]:
        """Load equipment.csv and interventions.csv (upsert), e.g. from examples/business."""
        equipment = _read_csv(directory / "equipment.csv")
        interventions = _read_csv(directory / "interventions.csv")
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO equipment (tag, site, model, commissioned_on, running_hours, criticality)"
                " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (tag) DO UPDATE SET"
                " site = EXCLUDED.site, model = EXCLUDED.model,"
                " commissioned_on = EXCLUDED.commissioned_on,"
                " running_hours = EXCLUDED.running_hours, criticality = EXCLUDED.criticality",
                [
                    (r["tag"], r["site"], r["model"], _none_if_blank(r.get("commissioned_on")),
                     int(r["running_hours"]) if _none_if_blank(r.get("running_hours")) else None,
                     _none_if_blank(r.get("criticality")))
                    for r in equipment
                ],
            )
            await cur.executemany(
                "INSERT INTO interventions (ot_id, equipment_tag, date, status, fault_code, symptom,"
                " diagnosis, action) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (ot_id) DO UPDATE SET status = EXCLUDED.status,"
                " fault_code = EXCLUDED.fault_code, symptom = EXCLUDED.symptom,"
                " diagnosis = EXCLUDED.diagnosis, action = EXCLUDED.action",
                [
                    (r["ot_id"], r["equipment_tag"], r["date"], r["status"],
                     _none_if_blank(r.get("fault_code")), _none_if_blank(r.get("symptom")),
                     _none_if_blank(r.get("diagnosis")), _none_if_blank(r.get("action")))
                    for r in interventions
                ],
            )
        return {"equipment": len(equipment), "interventions": len(interventions)}

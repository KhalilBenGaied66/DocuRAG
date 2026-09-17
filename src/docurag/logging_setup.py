"""JSON logs carrying the request ID, so one question can be followed across services."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "openai", "psycopg.pool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

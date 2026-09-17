"""Spare-parts ERP client: timeouts, bounded retries with backoff, Retry-After honoured."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 502, 503, 504}
MAX_WAIT_S = 5.0


class ErpClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 5.0,
        retries: int = 2,
        backoff_s: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.retries = retries
        self.backoff_s = backoff_s
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)

    async def stock(self, refs: list[str]) -> dict[str, dict[str, Any] | None]:
        """Stock per reference; None for references the ERP doesn't know."""
        response = await self._get("/v1/stock", {"refs": ",".join(refs)})
        payload = response.json()
        found = {item["ref"]: item for item in payload.get("items", [])}
        return {ref: found.get(ref) for ref in refs}

    async def _get(self, path: str, params: dict[str, str]) -> httpx.Response:
        delay = self.backoff_s
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            try:
                response = await self._client.get(path, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if last:
                    raise
                log.warning("ERP %s failed (%s), retry %s", path, type(exc).__name__, attempt + 1)
            else:
                if response.status_code not in RETRYABLE_STATUS or last:
                    response.raise_for_status()
                    return response
                delay = _retry_after(response, delay)
                log.warning("ERP %s returned %s, retry %s", path, response.status_code, attempt + 1)
            await asyncio.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._client.aclose()


def _retry_after(response: httpx.Response, default: float) -> float:
    try:
        return min(float(response.headers.get("retry-after", default)), MAX_WAIT_S)
    except ValueError:  # an HTTP date: use our own backoff
        return default

"""Token counting. Uses tiktoken when available, a 4-chars-per-token estimate otherwise."""

from __future__ import annotations

import functools
import logging
import math
import os

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _encoder():  # noqa: ANN202 - tiktoken.Encoding | None
    if os.environ.get("DOCURAG_TOKENIZER") == "approx":
        return None
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:  # network needed on first use; never block ingestion on it
        log.warning("tiktoken unavailable (%s); using a 4-chars-per-token estimate", exc)
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return max(1, math.ceil(len(text) / 4))
    return len(enc.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    enc = _encoder()
    if enc is None:
        return text[: max_tokens * 4]
    return enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])

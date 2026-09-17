"""Text clean-up applied to every parsed document before chunking."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from docurag.models import Element, ParsedDocument

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_HSPACE = re.compile(r"[ \t   ]+")
_DIGITS = re.compile(r"\d+")
PAGE_MARGIN = 0.08  # top/bottom share of the page where headers and footers live


def clean_text(text: str) -> str:
    # NFC, not NFKC: NFKC would turn "m²" into "m2" and "½" into "1/2".
    text = unicodedata.normalize("NFC", text).replace("­", "")
    for ligature, plain in _LIGATURES.items():
        text = text.replace(ligature, plain)
    lines = [_HSPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _in_margin(el: Element) -> bool:
    return el.y0 is not None and el.y1 is not None and (
        el.y0 < PAGE_MARGIN or el.y1 > 1 - PAGE_MARGIN
    )


def _signature(text: str) -> str:
    return _DIGITS.sub("#", text.lower()).strip()


def drop_page_furniture(
    elements: list[Element], page_count: int | None, min_share: float = 0.5
) -> tuple[list[Element], int]:
    """Remove running headers, footers and page numbers repeated across pages."""
    if not page_count or page_count < 3:
        return elements, 0
    pages_by_sig: dict[str, set[int]] = defaultdict(set)
    for el in elements:
        if el.page is not None and _in_margin(el) and len(el.text) < 200:
            pages_by_sig[_signature(el.text)].add(el.page)
    threshold = max(3, min_share * page_count)
    furniture = {sig for sig, pages in pages_by_sig.items() if len(pages) >= threshold}
    kept = [el for el in elements if not (_in_margin(el) and _signature(el.text) in furniture)]
    return kept, len(elements) - len(kept)


def normalize(parsed: ParsedDocument) -> ParsedDocument:
    elements, removed = drop_page_furniture(parsed.elements, parsed.page_count)
    cleaned: list[Element] = []
    for el in elements:
        el.text = clean_text(el.text)
        if el.text:
            cleaned.append(el)
    parsed.elements = cleaned
    parsed.title = clean_text(parsed.title) or parsed.title
    parsed.stats["furniture_removed"] = removed
    return parsed

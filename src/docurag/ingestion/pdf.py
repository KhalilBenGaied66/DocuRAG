"""PDF parsing with PyMuPDF.

- Headings come from font size relative to the body text, and from numbering
  ("4.3 Couples de serrage"), so sections follow the document's own structure.
- Tables are found with PyMuPDF's table finder and emitted whole, as Markdown; the text
  blocks they cover are skipped so values are not indexed twice as word soup.
- Pages with (almost) no text layer but with images are reported, so the pipeline can
  route the document to OCR instead of silently indexing nothing.

Complex multi-column layouts are better served by the layout model (ocr_backend=azure-di).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf

from docurag.models import Element, ParsedDocument

log = logging.getLogger(__name__)

NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s+\S")
HEADING_SIZE_RATIO = 1.15
MAX_HEADING_CHARS = 150
MAX_HEADING_LEVEL = 4
BOLD_FLAG = 16
_JUNK_TITLES = re.compile(r"^(untitled|microsoft word|document\d*)", re.I)


def _line_text(line: dict) -> str:
    return "".join(span["text"] for span in line["spans"]).strip()


def _join_lines(lines: list[str]) -> str:
    """Join visual lines into a paragraph, repairing end-of-line hyphenation."""
    out = ""
    for line in lines:
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            out = out[:-1] + line  # "main-" + "tenance"
        else:
            out = f"{out} {line}"
    return out


def _body_font_size(page_dicts: list[dict]) -> float:
    sizes: Counter[float] = Counter()
    for page in page_dicts:
        for block in page["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _find_tables(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    found: list[tuple[pymupdf.Rect, str]] = []
    try:
        for table in page.find_tables().tables:
            if table.row_count < 2 or table.col_count < 2:
                continue  # single rows/columns are usually layout boxes, not tables
            markdown = table.to_markdown().strip()
            if markdown:
                found.append((pymupdf.Rect(table.bbox), markdown))
    except Exception as exc:  # table detection is best effort; never lose the page for it
        log.warning("table detection failed on page %s: %s", page.number + 1, exc)
    return found


def _center_inside(block: pymupdf.Rect, table: pymupdf.Rect) -> bool:
    cx, cy = (block.x0 + block.x1) / 2, (block.y0 + block.y1) / 2
    return table.x0 <= cx <= table.x1 and table.y0 <= cy <= table.y1


def _is_heading(text: str, n_lines: int, size: float, bold: bool, body_size: float) -> bool:
    if n_lines > 2 or len(text) > MAX_HEADING_CHARS or not re.search(r"[^\W\d_]", text):
        return False
    if text.endswith((".", ",", ";", ":")):
        return False
    if size >= body_size * HEADING_SIZE_RATIO:
        return True
    return bold and bool(NUMBERED_HEADING.match(text))


def _assign_levels(headings: list[tuple[Element, float]]) -> None:
    unnumbered_sizes = sorted(
        {round(size, 1) for el, size in headings if not NUMBERED_HEADING.match(el.text)},
        reverse=True,
    )
    for el, size in headings:
        numbered = NUMBERED_HEADING.match(el.text)
        if numbered:
            level = numbered.group(1).count(".") + 1
        else:
            level = unnumbered_sizes.index(round(size, 1)) + 1
        el.level = min(level, MAX_HEADING_LEVEL)


def _title(doc: pymupdf.Document, elements: list[Element], path: Path) -> str:
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if len(meta_title) > 3 and not _JUNK_TITLES.match(meta_title):
        return meta_title
    first = next((el.text for el in elements if el.kind == "heading" and el.level == 1), None)
    return first or path.stem.replace("_", " ")


def parse_pdf(path: Path, *, min_text_chars_per_page: int = 50) -> ParsedDocument:
    doc = pymupdf.open(path)
    try:
        pages = [(page, page.get_text("dict"), _find_tables(page)) for page in doc]
        body_size = _body_font_size([d for _, d, _ in pages])

        elements: list[Element] = []
        headings: list[tuple[Element, float]] = []
        pages_without_text: list[int] = []
        n_tables = 0

        for page, page_dict, tables in pages:
            pno = page.number + 1
            height = page_dict.get("height") or 1.0
            page_elements: list[Element] = []
            page_chars = 0

            for block in page_dict["blocks"]:
                if block.get("type") != 0:
                    continue
                rect = pymupdf.Rect(block["bbox"])
                if any(_center_inside(rect, t_rect) for t_rect, _ in tables):
                    continue
                lines = block.get("lines", [])
                text = _join_lines([_line_text(line) for line in lines])
                if not text:
                    continue
                page_chars += len(text)
                spans = [s for line in lines for s in line["spans"] if s["text"].strip()]
                size = max((s["size"] for s in spans), default=body_size)
                bold = bool(spans) and all(s["flags"] & BOLD_FLAG for s in spans)
                is_heading = _is_heading(text, len(lines), size, bold, body_size)
                el = Element(
                    kind="heading" if is_heading else "text",
                    text=text,
                    page=pno,
                    y0=rect.y0 / height,
                    y1=rect.y1 / height,
                )
                if is_heading:
                    headings.append((el, size))
                page_elements.append(el)

            # Insert tables where they sit vertically, keeping the text blocks' own order.
            for t_rect, markdown in sorted(tables, key=lambda t: t[0].y0):
                table_el = Element(
                    kind="table", text=markdown, page=pno,
                    y0=t_rect.y0 / height, y1=t_rect.y1 / height,
                )
                idx = next(
                    (i for i, el in enumerate(page_elements) if (el.y0 or 0) > table_el.y0),
                    len(page_elements),
                )
                page_elements.insert(idx, table_el)
                page_chars += len(markdown)
                n_tables += 1

            if page_chars < min_text_chars_per_page and page.get_images():
                pages_without_text.append(pno)
            elements.extend(page_elements)

        _assign_levels(headings)
        warnings = []
        if pages_without_text:
            listed = ", ".join(map(str, pages_without_text[:20]))
            more = "…" if len(pages_without_text) > 20 else ""
            warnings.append(f"pages without a text layer (scanned?): {listed}{more}")

        return ParsedDocument(
            title=_title(doc, elements, path),
            elements=elements,
            page_count=doc.page_count,
            warnings=warnings,
            stats={
                "pages": doc.page_count,
                "pages_without_text": len(pages_without_text),
                "tables": n_tables,
                "headings": len(headings),
            },
        )
    finally:
        doc.close()

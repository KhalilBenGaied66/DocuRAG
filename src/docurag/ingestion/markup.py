"""Knowledge-base formats: Markdown (also the output of Azure Document Intelligence) and HTML."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from docurag.models import Element, ParsedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_PAGE_BREAK = re.compile(r"^\s*<!--\s*PageBreak\s*-->\s*$", re.I)
_HTML_COMMENT = re.compile(r"^\s*<!--.*-->\s*$")  # incl. DI's PageHeader/PageFooter/PageNumber

_HTML_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table", "blockquote",
                "dt", "dd"]
_HTML_CONTAINERS = {"p", "li", "pre", "table", "blockquote", "dt", "dd"}
_HTML_NOISE = ["script", "style", "nav", "header", "footer", "aside", "noscript", "form"]


def html_table_to_markdown(table: Tag | None) -> str:
    if table is None:
        return ""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for cell in tr.find_all(["th", "td"]):
            text = " ".join(cell.get_text(" ", strip=True).split()).replace("|", "\\|")
            try:
                span = int(cell.get("colspan", 1))
            except (TypeError, ValueError):
                span = 1
            cells.extend([text] * max(1, min(span, 20)))
        if cells:
            rows.append(cells)
    return rows_to_markdown(rows)


def rows_to_markdown(rows: list[list[str]]) -> str:
    """A Markdown table whose first row is the header; short rows are padded."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def parse_markdown(text: str, *, fallback_title: str, track_pages: bool = False) -> ParsedDocument:
    """Parse Markdown into elements. track_pages follows DI's <!-- PageBreak --> markers."""
    elements: list[Element] = []
    page: int | None = 1 if track_pages else None
    buf: list[str] = []

    def flush() -> None:
        if buf:
            elements.append(Element("text", "\n".join(buf).strip(), page=page))
            buf.clear()

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            flush()
            block = [line]
            i += 1
            while i < len(lines) and not _FENCE.match(lines[i]):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
            elements.append(Element("text", "\n".join(block), page=page))
            i += 1
            continue
        if _PAGE_BREAK.match(line):
            flush()
            if page is not None:
                page += 1
            i += 1
            continue
        if _HTML_COMMENT.match(line):
            i += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            elements.append(
                Element("heading", heading.group(2).strip(), page=page, level=len(heading.group(1)))
            )
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            elements.append(Element("table", "\n".join(rows), page=page))
            continue
        if line.lstrip().lower().startswith("<table"):
            flush()
            html = []
            while i < len(lines):
                html.append(lines[i])
                i += 1
                if "</table>" in html[-1].lower():
                    break
            markdown = html_table_to_markdown(BeautifulSoup("\n".join(html), "html.parser").table)
            if markdown:
                elements.append(Element("table", markdown, page=page))
            continue
        if not line.strip():
            flush()
        else:
            buf.append(line.rstrip())
        i += 1
    flush()

    title = next(
        (el.text for el in elements if el.kind == "heading" and el.level == 1), fallback_title
    )
    return ParsedDocument(
        title=title,
        elements=elements,
        page_count=page if track_pages else None,
        stats={
            "headings": sum(el.kind == "heading" for el in elements),
            "tables": sum(el.kind == "table" for el in elements),
        },
    )


def parse_html(html: str, *, fallback_title: str) -> ParsedDocument:
    soup = BeautifulSoup(html, "html.parser")
    page_title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(_HTML_NOISE):
        tag.decompose()
    for tag in soup.select('[role="navigation"], .breadcrumb, .breadcrumbs'):
        tag.decompose()
    root = soup.body or soup

    elements: list[Element] = []
    for el in root.find_all(_HTML_BLOCKS):
        if any(parent.name in _HTML_CONTAINERS for parent in el.parents):
            continue  # already covered by its container
        name = el.name
        if name[0] == "h" and name[1:].isdigit():
            text = el.get_text(" ", strip=True)
            if text:
                elements.append(Element("heading", text, level=int(name[1])))
        elif name == "table":
            markdown = html_table_to_markdown(el)
            if markdown:
                elements.append(Element("table", markdown))
        elif name == "pre":
            elements.append(Element("text", el.get_text()))
        else:
            text = el.get_text(" ", strip=True)
            if text:
                elements.append(Element("text", f"- {text}" if name == "li" else text))

    warnings: list[str] = []
    structured_chars = sum(len(el.text) for el in elements)
    total_chars = len(root.get_text(" ", strip=True))
    if total_chars and structured_chars < 0.5 * total_chars:
        # Text living directly in <div>s (common in wiki exports): keep it, lose structure.
        warnings.append("html: most text is outside block tags; fell back to plain-text paragraphs")
        headings = [el for el in elements if el.kind == "heading"]
        paragraphs = [p.strip() for p in root.get_text("\n").split("\n\n") if p.strip()]
        elements = headings[:1] + [Element("text", " ".join(p.split())) for p in paragraphs]

    title = page_title or next(
        (el.text for el in elements if el.kind == "heading" and el.level == 1), fallback_title
    )
    return ParsedDocument(
        title=title,
        elements=elements,
        warnings=warnings,
        stats={
            "headings": sum(el.kind == "heading" for el in elements),
            "tables": sum(el.kind == "table" for el in elements),
        },
    )

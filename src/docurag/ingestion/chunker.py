"""Structure-aware chunking.

Sections follow the document's headings; they are the *parents* the LLM reads.
Chunks are the small units that are embedded and searched. Each chunk's indexed content
starts with a contextual header ("Manuel P-200 > 4. Maintenance > 4.3 Couples de serrage")
so it can be matched on where it sits, not only on what it says.

Tables are kept whole (with the short caption line that introduces them) unless they exceed
table_max_tokens; then they are split by rows and the header row is repeated in each part.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from docurag.models import ChunkDraft, Element, SectionDraft
from docurag.tokens import count_tokens

CAPTION_MAX_TOKENS = 40


@dataclass(slots=True)
class ChunkingConfig:
    target_tokens: int = 400
    max_tokens: int = 600
    table_max_tokens: int = 1500


@dataclass(slots=True)
class _Section:
    path: list[str]
    elements: list[Element] = field(default_factory=list)


@dataclass(slots=True)
class _Piece:
    kind: str
    body: str
    sources: list[Element]
    tokens: int


def split_sections(elements: list[Element]) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(path=[])
    for el in elements:
        if el.kind == "heading":
            if current.elements:
                sections.append(current)
            level = max(1, el.level or 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, el.text))
            current = _Section(path=[text for _, text in stack])
        else:
            current.elements.append(el)
    if current.elements:
        sections.append(current)
    return sections


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def drop_title_heading(sections: list[_Section], title: str) -> list[_Section]:
    """A heading that wraps the whole document and repeats its title is the title, not a
    section. Keep it out of section paths so chunk headers don't say it twice."""
    if not sections or not sections[0].path:
        return sections
    first = sections[0].path[0]
    t, f = _norm(title), _norm(first)
    if not t or not f or not (t in f or f in t):
        return sections
    if not all(s.path and s.path[0] == first for s in sections):
        return sections
    for s in sections:
        s.path = s.path[1:]
    return sections


def _pages(elements: list[Element]) -> tuple[int | None, int | None]:
    pages = [el.page for el in elements if el.page is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def split_table(markdown: str, max_tokens: int) -> list[str]:
    if count_tokens(markdown) <= max_tokens:
        return [markdown]
    lines = markdown.split("\n")
    is_pipe_table = (
        len(lines) >= 3 and "-" in lines[1] and set(lines[1].replace("|", "").strip()) <= set("-: ")
    )
    head, rows = (lines[:2], lines[2:]) if is_pipe_table else ([], lines)
    budget = max(1, max_tokens - count_tokens("\n".join(head)))
    parts: list[str] = []
    current: list[str] = []
    used = 0
    for row in rows:
        cost = count_tokens(row) + 1
        if current and used + cost > budget:
            parts.append("\n".join(head + current))
            current, used = [], 0
        current.append(row)
        used += cost
    if current:
        parts.append("\n".join(head + current))
    return parts


class _Packer:
    def __init__(self, cfg: ChunkingConfig, splitter: RecursiveCharacterTextSplitter) -> None:
        self.cfg = cfg
        self.splitter = splitter
        self.pieces: list[_Piece] = []
        self.buf: list[Element] = []
        self.buf_tokens = 0

    def flush(self) -> None:
        if self.buf:
            body = "\n\n".join(el.text for el in self.buf)
            self.pieces.append(_Piece("text", body, self.buf, count_tokens(body)))
            self.buf, self.buf_tokens = [], 0

    def add(self, el: Element) -> None:
        cfg = self.cfg
        tokens = count_tokens(el.text)
        if el.kind == "table":
            caption: list[Element] = []
            if self.buf and count_tokens(self.buf[-1].text) <= CAPTION_MAX_TOKENS:
                caption = [self.buf.pop()]
                self.buf_tokens = sum(count_tokens(e.text) for e in self.buf)
            self.flush()
            prefix = f"{caption[0].text}\n\n" if caption else ""
            for part in split_table(el.text, cfg.table_max_tokens):
                body = prefix + part
                self.pieces.append(_Piece("table", body, caption + [el], count_tokens(body)))
        elif tokens > cfg.max_tokens:
            self.flush()
            for part in self.splitter.split_text(el.text):
                self.pieces.append(_Piece("text", part, [el], count_tokens(part)))
        elif self.buf_tokens + tokens <= cfg.target_tokens or (
            self.buf_tokens < cfg.target_tokens // 2 and self.buf_tokens + tokens <= cfg.max_tokens
        ):
            self.buf.append(el)
            self.buf_tokens += tokens
        else:
            self.flush()
            self.buf.append(el)
            self.buf_tokens = tokens

    def finish(self) -> list[_Piece]:
        self.flush()
        pieces = self.pieces
        # A tiny trailing text piece is merged into the previous one rather than left alone.
        if len(pieces) >= 2:
            last, prev = pieces[-1], pieces[-2]
            if (
                last.kind == prev.kind == "text"
                and last.tokens < self.cfg.target_tokens // 4
                and prev.tokens + last.tokens <= self.cfg.max_tokens
            ):
                body = f"{prev.body}\n\n{last.body}"
                pieces[-2:] = [_Piece("text", body, prev.sources + last.sources, count_tokens(body))]
        return pieces


def chunk_document(
    title: str, elements: list[Element], cfg: ChunkingConfig
) -> tuple[list[SectionDraft], list[ChunkDraft]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.target_tokens,
        chunk_overlap=cfg.target_tokens // 10,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    sections: list[SectionDraft] = []
    chunks: list[ChunkDraft] = []
    for s_ord, section in enumerate(drop_title_heading(split_sections(elements), title)):
        path_str = " > ".join(section.path)
        content = "\n\n".join(el.text for el in section.elements)
        p0, p1 = _pages(section.elements)
        sections.append(
            SectionDraft(s_ord, section.path, content, count_tokens(content), p0, p1)
        )
        header = f"{title} > {path_str}" if path_str else title
        packer = _Packer(cfg, splitter)
        for el in section.elements:
            packer.add(el)
        for c_ord, piece in enumerate(packer.finish()):
            c0, c1 = _pages(piece.sources)
            chunks.append(
                ChunkDraft(
                    section_ord=s_ord,
                    ord=c_ord,
                    kind="table" if piece.kind == "table" else "text",
                    body=piece.body,
                    content=f"{header}\n\n{piece.body}",
                    section_path=path_str,
                    page_start=c0,
                    page_end=c1,
                    tokens=piece.tokens,
                )
            )
    return sections, chunks


def chunk_stats(chunks: list[ChunkDraft]) -> dict[str, int]:
    if not chunks:
        return {"chunks": 0}
    sizes = sorted(c.tokens for c in chunks)
    p95 = sizes[min(len(sizes) - 1, int(0.95 * len(sizes)))]
    return {
        "chunks": len(chunks),
        "table_chunks": sum(c.kind == "table" for c in chunks),
        "tokens_p50": int(statistics.median(sizes)),
        "tokens_p95": p95,
        "tokens_max": sizes[-1],
    }

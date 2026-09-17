"""Context assembly (small-to-big) and citation verification."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from docurag.db.repository import Repository
from docurag.models import Citation, SearchHit
from docurag.tokens import count_tokens, truncate_to_tokens

CITATION = re.compile(r"\[(S\d+)\]")


@dataclass(slots=True)
class ContextSource:
    sid: str
    hit: SearchHit
    text: str
    tokens: int
    page_start: int | None
    page_end: int | None
    parent: bool  # True when the whole section is shown, False for a window of chunks

    def citation(self) -> Citation:
        h = self.hit
        return Citation(
            source_id=self.sid, document_id=h.document_id, title=h.title,
            source_uri=h.source_uri, section_path=h.section_path,
            page_start=self.page_start, page_end=self.page_end,
        )


async def assemble_context(
    hits: list[SearchHit], repo: Repository, *, parent_max_tokens: int, budget_tokens: int
) -> list[ContextSource]:
    """Match on small chunks, show the model their section — or a window when it's too long.

    Hits arrive best-first; the budget is spent in that order, and a section is shown once.
    """
    sections = await repo.fetch_sections(sorted({h.section_id for h in hits}))
    sources: list[ContextSource] = []
    shown_sections: set[int] = set()
    shown_chunks: set[tuple[int, int]] = set()
    used = 0

    for hit in hits:
        if hit.section_id in shown_sections or (hit.section_id, hit.ord) in shown_chunks:
            continue
        section = sections.get(hit.section_id)
        if section and section["tokens"] <= parent_max_tokens:
            text, tokens, parent = section["content"], section["tokens"], True
            pages = (section["page_start"], section["page_end"])
        else:
            window = await repo.fetch_window(hit.section_id, hit.ord)
            text = "\n\n".join(w["body"] for w in window) or hit.body
            tokens, parent = count_tokens(text), False
            starts = [w["page_start"] for w in window if w["page_start"] is not None]
            ends = [w["page_end"] for w in window if w["page_end"] is not None]
            pages = (min(starts, default=hit.page_start), max(ends, default=hit.page_end))

        if used + tokens > budget_tokens:
            if sources:
                continue  # a smaller, lower-ranked source may still fit
            text, tokens = truncate_to_tokens(text, budget_tokens), budget_tokens

        if parent:
            shown_sections.add(hit.section_id)
        else:
            shown_chunks.update((hit.section_id, o) for o in range(hit.ord - 1, hit.ord + 2))
        sources.append(
            ContextSource(f"S{len(sources) + 1}", hit, text, tokens, pages[0], pages[1], parent)
        )
        used += tokens
    return sources


def render_sources(sources: list[ContextSource]) -> str:
    blocks = []
    for s in sources:
        pages = ""
        if s.page_start is not None:
            pages = f' pages="{s.page_start}' + (
                f'-{s.page_end}"' if s.page_end not in (None, s.page_start) else '"'
            )
        # Source text cannot close its own tag and smuggle text outside the data block.
        body = s.text.replace("</source", "<\\/source")
        blocks.append(
            f'<source id="{s.sid}" document="{html.escape(s.hit.title)}"'
            f' section="{html.escape(s.hit.section_path)}"{pages}>\n{body}\n</source>'
        )
    return "\n\n".join(blocks)


def verify_citations(
    text: str, sources: list[ContextSource]
) -> tuple[str, list[Citation], list[str]]:
    """Keep citations that point to a source the model was shown; strip and report the rest."""
    by_id = {s.sid: s for s in sources}
    found = CITATION.findall(text)
    invalid = sorted({sid for sid in found if sid not in by_id}, key=lambda s: int(s[1:]))
    cleaned = CITATION.sub(lambda m: m.group(0) if m.group(1) in by_id else "", text)
    cited: list[str] = []
    for sid in found:
        if sid in by_id and sid not in cited:
            cited.append(sid)
    return cleaned.strip(), [by_id[sid].citation() for sid in cited], invalid

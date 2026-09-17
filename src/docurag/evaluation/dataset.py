"""Evaluation set: one JSON object per line.

{"id": "q001", "question": "…", "reference": "…", "category": "table",
 "relevant": [{"source_uri": "corpus://maintenance/manuals/p200.pdf", "pages": [12]}]}

Unanswerable questions have "relevant": [] — the system must abstain on them.
Relevance is labelled at document + page level: realistic for a human to annotate.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class RelevantSource(BaseModel):
    source_uri: str
    pages: list[int] = Field(default_factory=list)


class EvalItem(BaseModel):
    id: str
    question: str
    reference: str | None = None
    relevant: list[RelevantSource] = Field(default_factory=list)
    category: str = "general"
    groups: list[str] | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def load_dataset(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            item = EvalItem.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path}:{n}: duplicate id {item.id}")
        seen.add(item.id)
        items.append(item)
    return items

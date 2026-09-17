"""Domain objects shared by ingestion, retrieval, generation and the API."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

ElementKind = Literal["heading", "text", "table"]
DocType = Literal["pdf", "kb", "record"]


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- ingestion


@dataclass(slots=True)
class Element:
    """One structural unit produced by a parser, in reading order."""

    kind: ElementKind
    text: str
    page: int | None = None
    level: int = 0  # headings only: 1 = top level
    y0: float | None = None  # vertical position as a fraction of page height (PDF only)
    y1: float | None = None


@dataclass(slots=True)
class ParsedDocument:
    title: str
    elements: list[Element]
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class SectionDraft:
    ord: int
    path: list[str]
    content: str
    tokens: int
    page_start: int | None
    page_end: int | None

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)


@dataclass(slots=True)
class ChunkDraft:
    section_ord: int
    ord: int
    kind: Literal["text", "table"]
    body: str
    content: str  # body with its contextual header: what gets embedded and indexed
    section_path: str
    page_start: int | None
    page_end: int | None
    tokens: int

    @property
    def content_hash(self) -> str:
        return sha256(self.content)


class DocumentSpec(BaseModel):
    """What the caller tells us about a document: the manifest entry or the upload form."""

    source_uri: str
    path: str
    title: str | None = None
    doc_key: str | None = None
    version: str | None = None
    doc_type: DocType | None = None
    language: str = "fr"
    acl_groups: list[str] = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)
    record_text_fields: list[str] = Field(default_factory=list)  # doc_type=record only
    record_id_field: str | None = None

    @property
    def key(self) -> str:
        return self.doc_key or self.source_uri


class IngestResult(BaseModel):
    source_uri: str
    status: Literal["indexed", "unchanged", "acl_updated", "failed"]
    document_id: UUID | None = None
    report: dict = Field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------- retrieval


@dataclass(slots=True)
class SearchHit:
    chunk_id: int
    document_id: UUID
    section_id: int
    ord: int
    title: str
    source_uri: str
    section_path: str
    page_start: int | None
    page_end: int | None
    kind: str
    body: str
    fused_score: float
    vec_rank: int | None = None
    kw_rank: int | None = None
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fused_score


# ---------------------------------------------------------------- answers


class Citation(BaseModel):
    source_id: str  # "S1", as written by the model
    document_id: UUID
    title: str
    source_uri: str
    section_path: str
    page_start: int | None = None
    page_end: int | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


class Answer(BaseModel):
    request_id: str
    text: str
    abstained: bool
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)  # everything the model was shown
    invalid_citations: list[str] = Field(default_factory=list)
    top_score: float | None = None
    standalone_question: str
    prompt_version: str
    usage: Usage = Field(default_factory=Usage)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    contexts: list[str] | None = None  # source texts, only filled for evaluation runs


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class User(BaseModel):
    id: str
    groups: list[str]

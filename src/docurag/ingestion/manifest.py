"""Corpus manifest (TOML): the list of documents, their versions and who may read them.

    uri_prefix = "corpus://maintenance/"
    base_dir = "corpus"                       # relative to the manifest file

    [defaults]
    acl_groups = ["maintenance"]
    language = "fr"

    [[documents]]
    path = "manuals/pompe_p200_rev_c.pdf"
    doc_key = "manual-p200"
    version = "C"
    acl_groups = ["maintenance", "methods"]
    metadata = { equipment = "P-200" }
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from docurag.models import DocumentSpec


@dataclass(slots=True)
class Manifest:
    uri_prefix: str
    documents: list[DocumentSpec]


def load_manifest(path: Path) -> Manifest:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    base = (path.parent / data.get("base_dir", ".")).resolve()
    prefix = data.get("uri_prefix") or f"file://{base.as_posix()}/"
    defaults = data.get("defaults", {})

    documents: list[DocumentSpec] = []
    seen: set[str] = set()
    for i, entry in enumerate(data.get("documents", []), start=1):
        merged = {**defaults, **entry}
        rel = merged.pop("path", None)
        if not rel:
            raise ValueError(f"manifest entry {i}: missing 'path'")
        merged["path"] = str((base / rel).resolve())
        merged.setdefault("source_uri", prefix + Path(rel).as_posix())
        try:
            spec = DocumentSpec(**merged)
        except ValidationError as exc:
            raise ValueError(f"manifest entry {i} ({rel}): {exc}") from exc
        if spec.source_uri in seen:
            raise ValueError(f"manifest entry {i}: duplicate source_uri {spec.source_uri}")
        seen.add(spec.source_uri)
        documents.append(spec)
    return Manifest(uri_prefix=prefix, documents=documents)

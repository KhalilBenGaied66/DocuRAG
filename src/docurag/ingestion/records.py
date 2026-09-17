"""Business records (CSV).

Structured values are answered by queries, not by similarity search: only the free-text
columns named in the spec are indexed, one section per record, so a note like "fuite au
niveau du joint" can be found and cited. Exact lookups (status, dates, counts) belong to
SQL tools, not to this index.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from docurag.models import Element, ParsedDocument


def parse_records(
    path: Path, *, title: str, id_field: str, text_fields: list[str]
) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8-sig")
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(
            raw[:4096], delimiters=",;\t|"
        )
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(raw), dialect=dialect)
    missing = [f for f in [id_field, *text_fields] if f not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"columns not found in {path.name}: {missing}")

    elements: list[Element] = []
    rows = without_text = 0
    for row in reader:
        rows += 1
        parts = [f"{f}: {row[f].strip()}" for f in text_fields if (row.get(f) or "").strip()]
        if not parts:
            without_text += 1
            continue
        elements.append(Element("heading", f"{id_field} {row[id_field].strip()}", level=1))
        elements.append(Element("text", "\n".join(parts)))

    warnings = [f"{without_text} record(s) had no text in {text_fields}"] if without_text else []
    return ParsedDocument(
        title=title,
        elements=elements,
        warnings=warnings,
        stats={"records": rows, "records_without_text": without_text},
    )

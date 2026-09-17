"""Choose the parser for a document and route scanned PDFs to OCR."""

from __future__ import annotations

from pathlib import Path

from docurag.config import Settings
from docurag.ingestion.markup import parse_html, parse_markdown
from docurag.ingestion.normalize import normalize
from docurag.ingestion.ocr import parse_with_azure_layout
from docurag.ingestion.pdf import parse_pdf
from docurag.ingestion.records import parse_records
from docurag.models import DocType, DocumentSpec, ParsedDocument

SUFFIX_TYPES: dict[str, DocType] = {
    ".pdf": "pdf",
    ".md": "kb",
    ".markdown": "kb",
    ".txt": "kb",
    ".html": "kb",
    ".htm": "kb",
    ".csv": "record",
}


class UnsupportedDocument(ValueError):
    pass


def infer_doc_type(path: Path) -> DocType:
    try:
        return SUFFIX_TYPES[path.suffix.lower()]
    except KeyError:
        raise UnsupportedDocument(
            f"unsupported file type {path.suffix!r}; expected one of {sorted(SUFFIX_TYPES)}"
        ) from None


def parse_document(spec: DocumentSpec, settings: Settings) -> tuple[ParsedDocument, DocType]:
    path = Path(spec.path)
    doc_type = spec.doc_type or infer_doc_type(path)
    suffix = path.suffix.lower()
    fallback_title = spec.title or path.stem.replace("_", " ")

    if suffix == ".pdf":
        parsed = parse_pdf(path, min_text_chars_per_page=settings.min_text_chars_per_page)
        if parsed.stats.get("pages_without_text"):
            if settings.ocr_backend == "azure-di":
                if not settings.azure_di_endpoint:
                    raise ValueError("ocr_backend=azure-di needs DOCURAG_AZURE_DI_ENDPOINT")
                parsed = parse_with_azure_layout(
                    path, endpoint=settings.azure_di_endpoint, fallback_title=fallback_title
                )
            else:
                parsed.warnings.append(
                    "scanned pages were NOT indexed: set DOCURAG_OCR_BACKEND=azure-di to OCR them"
                )
    elif suffix in {".md", ".markdown", ".txt"}:
        parsed = parse_markdown(path.read_text(encoding="utf-8"), fallback_title=fallback_title)
    elif suffix in {".html", ".htm"}:
        parsed = parse_html(path.read_text(encoding="utf-8"), fallback_title=fallback_title)
    elif suffix == ".csv":
        if not spec.record_id_field or not spec.record_text_fields:
            raise ValueError("CSV records need record_id_field and record_text_fields")
        parsed = parse_records(
            path,
            title=fallback_title,
            id_field=spec.record_id_field,
            text_fields=spec.record_text_fields,
        )
    else:
        raise UnsupportedDocument(f"unsupported file type {suffix!r}")

    if spec.title:
        parsed.title = spec.title
    return normalize(parsed), doc_type

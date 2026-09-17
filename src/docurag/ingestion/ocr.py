"""Scanned and complex-layout PDFs through Azure AI Document Intelligence (prebuilt-layout).

The layout model returns Markdown with headings, tables and <!-- PageBreak --> markers,
which the Markdown parser turns into elements with page numbers.
"""

from __future__ import annotations

from pathlib import Path

from docurag.ingestion.markup import parse_markdown
from docurag.models import ParsedDocument


def parse_with_azure_layout(path: Path, *, endpoint: str, fallback_title: str) -> ParsedDocument:
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import DocumentContentFormat
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise RuntimeError(
            "ocr_backend=azure-di needs the azure extra: pip install -e '.[azure]'"
        ) from exc

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=DefaultAzureCredential())
    with path.open("rb") as fh:
        poller = client.begin_analyze_document(
            "prebuilt-layout", fh, output_content_format=DocumentContentFormat.MARKDOWN
        )
        result = poller.result()

    parsed = parse_markdown(result.content or "", fallback_title=fallback_title, track_pages=True)
    parsed.page_count = len(result.pages or [])
    parsed.stats["pages"] = parsed.page_count
    parsed.stats["ocr_pages"] = parsed.page_count
    return parsed

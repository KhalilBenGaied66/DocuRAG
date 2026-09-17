from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from docurag.ingestion.markup import parse_html, parse_markdown
from docurag.ingestion.normalize import clean_text, drop_page_furniture, normalize
from docurag.ingestion.pdf import _join_lines, parse_pdf
from docurag.ingestion.records import parse_records
from docurag.models import Element

# ------------------------------------------------------------------ normalize


def test_clean_text_repairs_pdf_artefacts_but_keeps_units():
    assert clean_text("ﬁltre à  huile­ ") == "filtre à huile"
    assert clean_text("surface 2 m² — ½ tour") == "surface 2 m² — ½ tour"


def test_join_lines_repairs_hyphenation_only_before_lowercase():
    assert _join_lines(["la main-", "tenance préventive"]) == "la maintenance préventive"
    assert _join_lines(["pompe P-", "200"]) == "pompe P- 200"


def test_repeated_headers_and_footers_are_removed():
    elements = []
    for page in range(1, 5):
        elements += [
            Element("text", "Manuel P-200 — Rév. C", page=page, y0=0.02, y1=0.04),
            Element("text", f"Contenu de la page {page}", page=page, y0=0.4, y1=0.5),
            Element("text", f"Page {page} / 4", page=page, y0=0.95, y1=0.97),
        ]
    elements.insert(0, Element("heading", "Manuel de maintenance", page=1, y0=0.03, y1=0.06))

    kept, removed = drop_page_furniture(elements, page_count=4)

    assert removed == 8
    assert [e.text for e in kept][0] == "Manuel de maintenance"  # appears once: not furniture
    assert all("Contenu" in e.text for e in kept[1:])


# ------------------------------------------------------------------ PDF


def _make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    doc.set_metadata({"title": "Manuel de maintenance P-200"})
    body = ("La pompe doit être consignée avant toute intervention. Vérifier l'absence de "
            "pression résiduelle dans le circuit hydraulique et le niveau d'huile du palier. ")
    for n in range(1, 5):
        page = doc.new_page()
        page.insert_text((72, 30), "Manuel P-200 - Rev. C", fontsize=8)
        page.insert_text((280, 780), f"Page {n} / 4", fontsize=8)
        if n == 1:
            page.insert_text((72, 90), "1 Généralités", fontsize=16)
            page.insert_textbox(pymupdf.Rect(72, 110, 540, 220), body * 2, fontsize=10)
            page.insert_text((72, 250), "1.1 Sécurité", fontsize=13)
            page.insert_textbox(pymupdf.Rect(72, 270, 540, 330), body, fontsize=10)
            xs, ys = [72, 220, 368], [400, 420, 440, 460]
            for x in xs:
                page.draw_line((x, ys[0]), (x, ys[-1]))
            for y in ys:
                page.draw_line((xs[0], y), (xs[-1], y))
            cells = [["Vis", "Couple Nm"], ["M8", "25"], ["M10", "49"]]
            for r, row in enumerate(cells):
                for c, value in enumerate(row):
                    page.insert_text((xs[c] + 6, ys[r] + 14), value, fontsize=10)
        elif n in (2, 3):
            page.insert_textbox(pymupdf.Rect(72, 100, 540, 400), body * 3, fontsize=10)
        else:  # a "scanned" page: an image and nothing else
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
            pix.clear_with(200)
            page.insert_image(pymupdf.Rect(72, 100, 400, 400), pixmap=pix)
    doc.save(path)
    doc.close()


def test_pdf_structure_tables_and_scanned_pages(tmp_path):
    path = tmp_path / "p200.pdf"
    _make_pdf(path)

    parsed = normalize(parse_pdf(path))

    assert parsed.title == "Manuel de maintenance P-200"
    headings = [(e.text, e.level) for e in parsed.elements if e.kind == "heading"]
    assert headings == [("1 Généralités", 1), ("1.1 Sécurité", 2)]

    tables = [e for e in parsed.elements if e.kind == "table"]
    assert len(tables) == 1 and "M10" in tables[0].text and "49" in tables[0].text
    assert not any("M10" in e.text for e in parsed.elements if e.kind != "table")

    assert parsed.stats["pages_without_text"] == 1
    assert "4" in parsed.warnings[0]
    assert parsed.stats["furniture_removed"] >= 6
    assert not any("Rev. C" in e.text for e in parsed.elements)


# ------------------------------------------------------------------ Markdown / HTML

DI_MARKDOWN = """<!-- PageHeader="Manuel P-200" -->
# Manuel P-200

## 4.3 Couples de serrage

Texte avant le tableau.

<table><tr><th>Vis</th><th>Couple</th></tr><tr><td>M8</td><td>25</td></tr></table>

<!-- PageBreak -->
<!-- PageNumber="2" -->

## 5 Dépannage

~~~
E-41 : vibration palier
# pas un titre
~~~
"""


def test_markdown_with_document_intelligence_markers():
    parsed = parse_markdown(DI_MARKDOWN, fallback_title="x", track_pages=True)
    kinds = [(e.kind, e.page) for e in parsed.elements]
    assert kinds == [("heading", 1), ("heading", 1), ("text", 1), ("table", 1),
                     ("heading", 2), ("text", 2)]
    assert parsed.title == "Manuel P-200"
    table = next(e for e in parsed.elements if e.kind == "table")
    assert table.text.splitlines()[0] == "| Vis | Couple |"
    code = parsed.elements[-1].text
    assert "# pas un titre" in code  # headings inside code fences stay code
    assert not any("PageHeader" in e.text for e in parsed.elements)


def test_html_drops_navigation_and_converts_tables():
    html = """<html><head><title>Procédure P-200</title></head><body>
      <nav><a href="/">Accueil</a></nav>
      <h2>Démontage</h2><p>Consigner la pompe.</p>
      <ul><li>Retirer le carter</li><li>Déposer la roue</li></ul>
      <table><tr><td colspan="2">Couples</td></tr><tr><td>M8</td><td>25</td></tr></table>
      <footer>© Intranet</footer></body></html>"""
    parsed = parse_html(html, fallback_title="x")
    texts = [e.text for e in parsed.elements]
    assert parsed.title == "Procédure P-200"
    assert "Accueil" not in " ".join(texts) and "Intranet" not in " ".join(texts)
    assert "- Retirer le carter" in texts
    table = next(e.text for e in parsed.elements if e.kind == "table")
    assert table.splitlines()[0] == "| Couples | Couples |"


def test_html_without_block_tags_falls_back_to_text():
    html = "<body><h1>Note</h1><div>" + "Texte libre dans une div. " * 20 + "</div></body>"
    parsed = parse_html(html, fallback_title="x")
    assert parsed.warnings and "Texte libre" in parsed.elements[-1].text


# ------------------------------------------------------------------ records


def test_semicolon_csv_with_bom_indexes_only_text_fields(tmp_path):
    path = tmp_path / "ot.csv"
    path.write_text(
        "﻿ot_id;statut;symptome;action\n"
        "OT-1;clos;Fuite au joint;Remplacement du joint\n"
        "OT-2;ouvert;;\n",
        encoding="utf-8",
    )
    parsed = parse_records(path, title="Interventions", id_field="ot_id",
                           text_fields=["symptome", "action"])
    assert [e.text for e in parsed.elements] == [
        "ot_id OT-1", "symptome: Fuite au joint\naction: Remplacement du joint"
    ]
    assert parsed.stats == {"records": 2, "records_without_text": 1}
    assert "clos" not in " ".join(e.text for e in parsed.elements)


def test_csv_missing_columns_is_a_clear_error(tmp_path):
    path = tmp_path / "ot.csv"
    path.write_text("id,notes\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns not found"):
        parse_records(path, title="t", id_field="ot_id", text_fields=["notes"])

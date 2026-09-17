from __future__ import annotations

from docurag.ingestion.chunker import ChunkingConfig, chunk_document, split_table
from docurag.models import Element
from docurag.tokens import count_tokens

CFG = ChunkingConfig(target_tokens=120, max_tokens=200, table_max_tokens=300)
TABLE = "| Vis | Couple (N·m) |\n|---|---|\n| M8 | 25 |\n| M10 | 49 |"


def doc() -> list[Element]:
    return [
        Element("heading", "1 Généralités", level=1, page=1),
        Element("text", "Ce manuel couvre la pompe P-200.", page=1),
        Element("heading", "1.1 Sécurité", level=2, page=1),
        Element("text", "Consigner la pompe avant toute intervention.", page=2),
        Element("heading", "4 Maintenance", level=1, page=3),
        Element("heading", "4.3 Couples de serrage", level=2, page=3),
        Element("text", "Les vis doivent être serrées en croix, à froid.", page=3),
        Element("text", "Couples de serrage :", page=3),
        Element("table", TABLE, page=4),
    ]


def test_sections_follow_the_heading_hierarchy():
    sections, _ = chunk_document("Manuel P-200", doc(), CFG)
    assert [s.path for s in sections] == [
        ["1 Généralités"],
        ["1 Généralités", "1.1 Sécurité"],
        ["4 Maintenance", "4.3 Couples de serrage"],  # a heading with no text of its own is folded
    ]
    assert (sections[2].page_start, sections[2].page_end) == (3, 4)


def test_table_is_whole_with_its_caption_and_contextual_header():
    _, chunks = chunk_document("Manuel P-200", doc(), CFG)
    table = next(c for c in chunks if c.kind == "table")
    assert table.body == f"Couples de serrage :\n\n{TABLE}"
    assert table.content.startswith("Manuel P-200 > 4 Maintenance > 4.3 Couples de serrage\n\n")
    text_before = next(c for c in chunks if c.section_path.endswith("4.3 Couples de serrage")
                       and c.kind == "text")
    assert "Couples de serrage :" not in text_before.body  # caption moved, not duplicated
    assert (table.page_start, table.page_end) == (3, 4)


def test_long_paragraph_is_split_within_limits():
    long_text = " ".join(f"Phrase numéro {i} sur la maintenance préventive." for i in range(120))
    _, chunks = chunk_document("Doc", [Element("heading", "A", level=1), Element("text", long_text)], CFG)
    assert len(chunks) > 3
    assert all(c.tokens <= CFG.max_tokens for c in chunks)


def test_small_paragraphs_are_packed_not_emitted_one_by_one():
    paragraphs = [Element("text", f"Point {i} : vérifier le niveau d'huile.") for i in range(8)]
    _, chunks = chunk_document("Doc", [Element("heading", "A", level=1), *paragraphs], CFG)
    assert len(chunks) == 1


def test_tiny_trailing_piece_is_merged():
    body = "x" * 400  # ~100 tokens with the approx tokenizer
    elements = [Element("heading", "A", level=1), Element("text", body), Element("text", "fin.")]
    _, chunks = chunk_document("Doc", elements, ChunkingConfig(90, 200, 300))
    assert len(chunks) == 1 and chunks[0].body.endswith("fin.")


def test_big_table_is_split_by_rows_with_header_repeated():
    rows = [f"| M{i} | {i * 3} |" for i in range(200)]
    table = "| Vis | Couple |\n|---|---|\n" + "\n".join(rows)
    parts = split_table(table, 120)
    assert len(parts) > 1
    assert all(p.startswith("| Vis | Couple |\n|---|---|\n") for p in parts)
    assert all(count_tokens(p) <= 120 for p in parts)
    assert [r for p in parts for r in p.split("\n")[2:]] == rows


def test_heading_that_repeats_the_title_is_not_a_section():
    elements = [
        Element("heading", "Manuel P-200 (Rév. C)", level=1), Element("text", "Intro."),
        Element("heading", "4 Couples de serrage", level=2), Element("text", "Carter : 49 N·m."),
    ]
    sections, chunks = chunk_document("Manuel P-200", elements, CFG)
    assert [s.path for s in sections] == [[], ["4 Couples de serrage"]]
    assert chunks[-1].content.startswith("Manuel P-200 > 4 Couples de serrage\n\n")


def test_single_unrelated_top_heading_is_kept():
    elements = [Element("heading", "4 Couples de serrage", level=1), Element("text", "49 N·m.")]
    sections, _ = chunk_document("Manuel P-200", elements, CFG)
    assert sections[0].path == ["4 Couples de serrage"]


def test_text_before_first_heading_gets_title_only_header():
    _, chunks = chunk_document("Doc", [Element("text", "Préambule.")], CFG)
    assert chunks[0].content == "Doc\n\nPréambule." and chunks[0].section_path == ""

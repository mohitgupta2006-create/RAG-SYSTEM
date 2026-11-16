import pymupdf as fitz
import pytest

from ingestion_pipeline.ingestionPipeline import parse_pdf
from rag.errors import DocumentError


def make_pdf(path, pages):
    with fitz.open() as doc:
        for lines in pages:
            page = doc.new_page()
            for y, text in lines:
                page.insert_text((72, y), text)
        doc.save(path)
    return path


def test_single_page_keeps_body_and_margin(tmp_path):
    path = make_pdf(tmp_path / "one.pdf", [[(30, "Policy title"), (150, "Coverage includes hospitalization.")]])
    elements = parse_pdf(str(path))
    assert len(elements) == 2
    assert {e["page_num"] for e in elements} == {1}


def test_repeated_margins_removed_but_identical_body_retained(tmp_path):
    path = make_pdf(tmp_path / "two.pdf", [
        [(30, "Repeated title"), (150, "Repeated title"), (250, "Unique first page")],
        [(30, "Repeated title"), (150, "Second page benefits")],
    ])
    elements = parse_pdf(str(path))
    matching = [e for e in elements if e["content"] == "Repeated title"]
    assert len(matching) == 1
    assert not matching[0]["is_margin"]


def test_empty_and_over_limit_pdfs_rejected(tmp_path):
    blank = make_pdf(tmp_path / "blank.pdf", [[]])
    with pytest.raises(DocumentError):
        parse_pdf(str(blank))
    two = make_pdf(tmp_path / "two.pdf", [[(150, "a")], [(150, "b")]])
    with pytest.raises(DocumentError):
        parse_pdf(str(two), max_pages=1)

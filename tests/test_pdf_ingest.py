"""PDF fixtures that exercise reading order and text extraction failures."""

import fitz
import pytest

from semanticdiff.ingest.extract import extract_pdf
from semanticdiff.ingest.structure import build_blocks
from semanticdiff.ingest.segment import segment_blocks
from semanticdiff.models import ProvisionKind


def _pdf(pages):
    doc = fitz.open()
    for draw in pages:
        page = doc.new_page(width=600, height=800)
        draw(page)
    data = doc.tobytes()
    doc.close()
    return data


def test_two_column_paper_reads_down_columns():
    def draw(page):
        page.insert_text((50, 35), "A Study of Revision Detection")
        page.insert_text((50, 140), "Left paragraph one.")
        page.insert_text((330, 145), "Right paragraph one.")
        page.insert_text((50, 300), "Left paragraph two.")
        page.insert_text((330, 305), "Right paragraph two.")

    lines = [line.text for line in extract_pdf(_pdf([draw]))]
    assert lines == [
        "A Study of Revision Detection",
        "Left paragraph one.",
        "Left paragraph two.",
        "Right paragraph one.",
        "Right paragraph two.",
    ]


def test_full_width_heading_separates_two_column_regions():
    def draw(page):
        page.insert_text((50, 80), "Left top paragraph.")
        page.insert_text((330, 82), "Right top paragraph.")
        page.insert_text((50, 160), "Left top continuation.")
        page.insert_text((330, 162), "Right top continuation.")
        page.insert_text((70, 300), "A Full Width Section Heading Across Both Columns", fontsize=15)
        page.insert_text((50, 420), "Left bottom paragraph.")
        page.insert_text((330, 422), "Right bottom paragraph.")
        page.insert_text((50, 500), "Left bottom continuation.")
        page.insert_text((330, 502), "Right bottom continuation.")

    lines = [line.text for line in extract_pdf(_pdf([draw]))]
    assert lines == [
        "Left top paragraph.", "Left top continuation.",
        "Right top paragraph.", "Right top continuation.",
        "A Full Width Section Heading Across Both Columns",
        "Left bottom paragraph.", "Left bottom continuation.",
        "Right bottom paragraph.", "Right bottom continuation.",
    ]


def test_repeated_body_sentence_is_preserved_while_margin_header_is_removed():
    def draw(page):
        page.insert_text((50, 30), "Journal of Useful Results")
        page.insert_text((50, 300), "The control group received standard care.")

    lines = extract_pdf(_pdf([draw, draw, draw]))
    assert [line.text for line in lines] == [
        "The control group received standard care.",
        "The control group received standard care.",
        "The control group received standard care.",
    ]


def test_image_only_pdf_reports_ocr_requirement():
    with pytest.raises(ValueError, match="OCR"):
        extract_pdf(_pdf([lambda page: None]))


def test_scanned_page_is_ocrd_and_mixed_pages_keep_provenance():
    source = fitz.open()
    page = source.new_page(width=600, height=800)
    page.insert_text((50, 150), "The treatment group improved by 12 percent.", fontsize=18)
    image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
    source.close()

    def scanned(page):
        page.insert_image(page.rect, stream=image)

    def native(page):
        page.insert_text((50, 150), "The control group improved by 10 percent.", fontsize=18)

    lines = extract_pdf(_pdf([scanned, native]))
    assert any("treatment group improved by 12 percent" in line.text.lower()
               and line.page == 1 and line.ocr for line in lines)
    assert any("control group improved by 10 percent" in line.text.lower()
               and line.page == 2 and not line.ocr for line in lines)
    assert all(line.bbox is not None for line in lines)


def test_hybrid_page_keeps_native_header_and_ocrd_body():
    source = fitz.open()
    page = source.new_page(width=600, height=800)
    page.insert_text((50, 250), "The scanned body reports a 12 percent improvement.", fontsize=18)
    image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
    source.close()

    def hybrid(page):
        page.insert_image(page.rect, stream=image)
        page.insert_text((50, 40), "A native journal header with enough letters to bypass the old threshold")

    lines = extract_pdf(_pdf([hybrid]))
    assert any("native journal header" in line.text.lower() and not line.ocr for line in lines)
    assert any("scanned body reports a 12 percent improvement" in line.text.lower()
               and line.ocr for line in lines)


def test_raster_figure_labels_are_reviewable_without_duplicating_native_text(nlp):
    source = fitz.open()
    page = source.new_page(width=400, height=180)
    page.insert_text((30, 100), "BLEU score 41.2", fontsize=22)
    image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
    source.close()

    def paper(page):
        page.insert_text((50, 40), "Abstract")
        page.insert_text((50, 90), "The model improves translation quality.")
        page.insert_text((50, 160), "Figure 1: Translation results")
        page.insert_image(fitz.Rect(60, 210, 460, 390), stream=image)

    lines = extract_pdf(_pdf([paper]))
    assert any("BLEU score 41.2" in line.text and line.ocr and line.image_content for line in lines)
    assert sum("The model improves translation quality" in line.text for line in lines) == 1
    provisions = segment_blocks(build_blocks(lines, paper_mode=True), "v1", nlp, paper_mode=True)
    figure_text = next(p for p in provisions if "BLEU score 41.2" in p.text)
    assert figure_text.kind == ProvisionKind.NON_PROSE and figure_text.image_content

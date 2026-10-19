"""Text extraction from PDF / TXT into clean logical lines.

PDF handling (PyMuPDF):
  * two-column pages are read down each column, with full-width blocks as separators,
  * recurring text is removed only in the top/bottom page margins,
  * words hyphenated across a line break are re-joined,
  * wrapped lines inside a PDF block are merged into one logical line.
Image-only pages are OCRed locally with RapidOCR; source page numbers survive.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+)?[-–—]?\s*\d{1,4}\s*(?:of\s+\d{1,4})?\s*[-–—]?\s*$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")


@dataclass
class RawLine:
    text: str
    page: Optional[int] = None
    block: int = 0            # lines sharing a block id belong to the same paragraph
    blank_before: bool = False
    ocr: bool = False


@dataclass
class _PdfBlock:
    lines: list[str]
    x0: float
    y0: float
    x1: float
    y1: float


def _split_pdf_block(block: dict, width: float) -> list[_PdfBlock]:
    """Split MuPDF blocks that accidentally join neighbouring columns."""
    columns: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {}
    for line in block.get("lines", []):
        # Sideways journal/arXiv stamps are not part of the article prose.
        if abs(line.get("dir", (1.0, 0.0))[1]) > 0.2:
            continue
        text = _normalize_chars("".join(span.get("text", "") for span in line.get("spans", []))).strip()
        if not text:
            continue
        bbox = tuple(line["bbox"])
        x0, _, x1, _ = bbox
        if x1 - x0 >= width * 0.55 or (x0 < width * 0.46 and x1 > width * 0.54):
            side = "wide"
        else:
            side = "left" if (x0 + x1) / 2 < width / 2 else "right"
        columns.setdefault(side, []).append((text, bbox))

    result: list[_PdfBlock] = []
    for lines in columns.values():
        lines.sort(key=lambda item: (item[1][1], item[1][0]))
        group: list[tuple[str, tuple[float, float, float, float]]] = []

        def flush() -> None:
            if group:
                result.append(_PdfBlock([text for text, _ in group],
                                        min(box[0] for _, box in group), min(box[1] for _, box in group),
                                        max(box[2] for _, box in group), max(box[3] for _, box in group)))

        for item in lines:
            if group:
                prev = group[-1][1]
                line_height = max(prev[3] - prev[1], item[1][3] - item[1][1])
                if item[1][1] - prev[3] > line_height * 1.5:
                    flush()
                    group = []
            group.append(item)
        flush()
    return result


def _reading_order(blocks: list[_PdfBlock], width: float) -> list[_PdfBlock]:
    """Read columns independently when both sides have substantial body text.

    Full-width title, abstract, figure or section blocks divide the page into
    vertical bands. This keeps a full-width heading between the text above and
    below it instead of moving it to the end of a column.
    """
    if not blocks:
        return []
    midpoint = width / 2
    narrow = [b for b in blocks if b.x1 - b.x0 < width * 0.55]
    left = [b for b in narrow if (b.x0 + b.x1) / 2 < midpoint and b.x1 < midpoint + width * 0.04]
    right = [b for b in narrow if (b.x0 + b.x1) / 2 >= midpoint and b.x0 > midpoint - width * 0.04]
    if len(left) < 2 or len(right) < 2:
        return sorted(blocks, key=lambda b: (b.y0, b.x0))
    overlap = min(max(b.y1 for b in left), max(b.y1 for b in right)) - max(
        min(b.y0 for b in left), min(b.y0 for b in right))
    if overlap < 0.1 * max(b.y1 for b in blocks):
        return sorted(blocks, key=lambda b: (b.y0, b.x0))

    wide = sorted((b for b in blocks if b not in left and b not in right),
                  key=lambda b: (b.y0, b.x0))
    ordered: list[_PdfBlock] = []
    remaining = list(blocks)
    for separator in wide:
        above = [b for b in remaining if b is not separator and b.y1 <= separator.y0]
        ordered.extend(sorted((b for b in above if b in left), key=lambda b: (b.y0, b.x0)))
        ordered.extend(sorted((b for b in above if b in right), key=lambda b: (b.y0, b.x0)))
        for b in above:
            remaining.remove(b)
        if separator in remaining:
            ordered.append(separator)
            remaining.remove(separator)
    ordered.extend(sorted((b for b in remaining if b in left), key=lambda b: (b.y0, b.x0)))
    ordered.extend(sorted((b for b in remaining if b in right), key=lambda b: (b.y0, b.x0)))
    return ordered


@lru_cache(maxsize=1)
def _ocr_engine():
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Scanned PDF detected. Install the OCR dependencies from requirements.txt.") from exc
    return RapidOCR()


def _ocr_page(page) -> list[_PdfBlock]:
    """OCR a page image, preserving the text boxes for column ordering."""
    import fitz

    scale = 2.5  # 180 dpi; readable body text without an excessive bitmap
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    result = _ocr_engine()(pix.tobytes("png"))
    lines: list[_PdfBlock] = []
    for text, box, confidence in zip(
            result.txts if result.txts is not None else (),
            result.boxes if result.boxes is not None else (),
            result.scores if result.scores is not None else ()):
        if confidence < 0.45 or not text.strip():
            continue
        xs = [float(point[0]) / scale for point in box]
        ys = [float(point[1]) / scale for point in box]
        lines.append(_PdfBlock([_normalize_chars(text.strip())], min(xs), min(ys), max(xs), max(ys)))
    ordered = _reading_order(lines, page.rect.width)
    grouped: list[_PdfBlock] = []
    last_line: _PdfBlock | None = None
    for line in ordered:
        if grouped and last_line is not None:
            prev = grouped[-1]
            height = max(last_line.y1 - last_line.y0, line.y1 - line.y0)
            same_column = abs(line.x0 - last_line.x0) < page.rect.width * 0.08
            close = -height * 0.25 <= line.y0 - last_line.y1 <= height * 1.4
            if same_column and close:
                prev.lines.extend(line.lines)
                prev.x0, prev.y0 = min(prev.x0, line.x0), min(prev.y0, line.y0)
                prev.x1, prev.y1 = max(prev.x1, line.x1), max(prev.y1, line.y1)
                last_line = line
                continue
        grouped.append(line)
        last_line = line
    return grouped


def _normalize_chars(text: str) -> str:
    text = text.replace(" ", " ").replace("­", "")      # nbsp, soft hyphen
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _join_wrapped(lines: list[str]) -> str:
    """Merge the physical lines of one PDF block into a single logical line."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            # "exami-\nnation" is a broken word, while "English-\nto-German"
            # and "state-of-\nthe-art" retain their compound hyphen.
            out = (out if "-" in line.split()[0] else out[:-1]) + line
        else:
            out = f"{out} {line}"
    return re.sub(r"\s+", " ", out).strip()


_LINE_START_MARKER = re.compile(
    r"^\s*(?:\(?[a-z]{1,3}\)|\(?\d{1,3}[.)]|\d+(?:\.\d+)+|[•\-\*–]\s|"
    r"(?:section|article|chapter|clause|part)\s+\w+)", re.IGNORECASE)


def extract_pdf(data: bytes, header_footer_ratio: float = 0.5) -> list[RawLine]:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[tuple[float, list[_PdfBlock], bool]] = []
    for page in doc:
        blocks: list[_PdfBlock] = []
        for b in page.get_text("dict", sort=False)["blocks"]:
            if b["type"] != 0:                # skip image blocks
                continue
            blocks.extend(_split_pdf_block(b, page.rect.width))
        readable = sum(ch.isalpha() for block in blocks for line in block.lines for ch in line)
        used_ocr = readable < 25 and bool(page.get_images(full=True))
        if used_ocr:
            blocks = _ocr_page(page)
        else:
            blocks = _reading_order(blocks, page.rect.width)
        pages.append((page.rect.height, blocks, used_ocr))
    doc.close()

    # Running headers/footers: lines (digits masked) that recur on many pages.
    n_pages = len(pages)
    recurring: set[str] = set()
    if n_pages >= 3:
        counts = Counter()
        for height, blocks, _ in pages:
            seen = {_DIGITS_RE.sub("#", ln.strip().lower()) for blk in blocks
                    if blk.y0 < height * 0.12 or blk.y1 > height * 0.88
                    for ln in blk.lines}
            counts.update(seen)
        recurring = {k for k, c in counts.items() if c / n_pages >= header_footer_ratio}

    out: list[RawLine] = []
    block_id = 0
    for page_no, (height, blocks, used_ocr) in enumerate(pages, start=1):
        for blk in blocks:
            margin = blk.y0 < height * 0.12 or blk.y1 > height * 0.88
            kept = [ln for ln in blk.lines
                    if not (margin and PAGE_NUMBER_RE.match(ln))
                    and not (margin and _DIGITS_RE.sub("#", ln.strip().lower()) in recurring)]
            if not kept:
                continue
            # A block may still contain several list items / headings on separate
            # lines; start a new logical line whenever a line opens with a marker.
            group: list[str] = []
            for ln in kept:
                if group and _LINE_START_MARKER.match(ln):
                    out.append(RawLine(_join_wrapped(group), page_no, block_id, True, used_ocr))
                    group = []
                group.append(ln)
            if group:
                out.append(RawLine(_join_wrapped(group), page_no, block_id, True, used_ocr))
            block_id += 1
    if not out:
        raise ValueError("No text could be extracted from this PDF, including OCR of image-only pages.")
    return out


def extract_txt(text: str) -> list[RawLine]:
    """Plain text: one RawLine per non-empty line; blank lines separate blocks."""
    text = _normalize_chars(text)
    out: list[RawLine] = []
    block_id, blank = 0, True
    for line in text.split("\n"):
        if not line.strip():
            if not blank:
                block_id += 1
            blank = True
            continue
        out.append(RawLine(re.sub(r"\s+", " ", line).strip(), None, block_id, blank))
        blank = False
    return out


def extract(source: Union[str, Path, bytes], filename: Optional[str] = None,
            header_footer_ratio: float = 0.5) -> list[RawLine]:
    """Extract logical lines from a path, raw bytes (with ``filename``) or a text string.

    A ``str`` that is not an existing path is treated as the document text itself.
    """
    if isinstance(source, (str, Path)) and not isinstance(source, bytes):
        p = Path(source)
        if len(str(source)) < 260 and p.suffix and p.exists():
            filename = filename or p.name
            source = p.read_bytes()
        else:
            return extract_txt(str(source))
    name = (filename or "").lower()
    if name.endswith(".pdf") or source[:4] == b"%PDF":
        return extract_pdf(source, header_footer_ratio)
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return extract_txt(source.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file")

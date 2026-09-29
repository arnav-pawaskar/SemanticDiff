"""Text extraction from PDF / TXT into clean logical lines.

PDF handling (PyMuPDF):
  * lines are read block by block in reading order,
  * lines repeated on many pages (running headers / footers) and bare page numbers are dropped,
  * words hyphenated across a line break are re-joined,
  * wrapped lines inside a PDF block are merged into one logical line.
Scanned (image-only) PDFs and tables are out of scope.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
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
            out = out[:-1] + line                      # de-hyphenate "exami-\nnation"
        else:
            out = f"{out} {line}"
    return re.sub(r"\s+", " ", out).strip()


_LINE_START_MARKER = re.compile(
    r"^\s*(?:\(?[a-z]{1,3}\)|\(?\d{1,3}[.)]|\d+(?:\.\d+)+|[•\-\*–]\s|"
    r"(?:section|article|chapter|clause|part)\s+\w+)", re.IGNORECASE)


def extract_pdf(data: bytes, header_footer_ratio: float = 0.5) -> list[RawLine]:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[list[list[str]]] = []      # page -> block -> physical lines
    for page in doc:
        blocks = []
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0:                    # skip image blocks
                continue
            text = _normalize_chars(b[4])
            blocks.append([ln for ln in text.split("\n") if ln.strip()])
        pages.append(blocks)

    # Running headers/footers: lines (digits masked) that recur on many pages.
    n_pages = len(pages)
    recurring: set[str] = set()
    if n_pages >= 3:
        counts = Counter()
        for blocks in pages:
            seen = {_DIGITS_RE.sub("#", ln.strip().lower()) for blk in blocks for ln in blk}
            counts.update(seen)
        recurring = {k for k, c in counts.items() if c / n_pages >= header_footer_ratio}

    out: list[RawLine] = []
    block_id = 0
    for page_no, blocks in enumerate(pages, start=1):
        for blk in blocks:
            kept = [ln for ln in blk
                    if not PAGE_NUMBER_RE.match(ln)
                    and _DIGITS_RE.sub("#", ln.strip().lower()) not in recurring]
            if not kept:
                continue
            # A block may still contain several list items / headings on separate
            # lines; start a new logical line whenever a line opens with a marker.
            group: list[str] = []
            for ln in kept:
                if group and _LINE_START_MARKER.match(ln):
                    out.append(RawLine(_join_wrapped(group), page_no, block_id, True))
                    group = []
                group.append(ln)
            if group:
                out.append(RawLine(_join_wrapped(group), page_no, block_id, True))
            block_id += 1
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

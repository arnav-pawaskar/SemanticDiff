"""Recover document structure (sections, headings, list items) from logical lines.

Produces ``Block`` objects: runs of body text that share a section path. Headings
update the section path but are not themselves compared as provisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from semanticdiff.ingest.extract import RawLine

_KEYWORD_HEADING = re.compile(
    r"^(?P<kw>section|article|chapter|part|clause|rule|regulation)\s+"
    r"(?P<num>[0-9]+[A-Za-z]?(?:\.[0-9]+)*|[IVXLC]+)\s*[.:)\-–—]?\s*(?P<rest>.*)$",
    re.IGNORECASE)
_NUMBERED = re.compile(r"^(?P<num>\d{1,3}(?:\.\d{1,3})*)(?:[.)]|\s)\s*(?P<rest>\S.*)$")
_LIST_ITEM = re.compile(r"^(?P<label>\((?:[a-z]{1,3}|\d{1,2})\)|(?:[a-z]|[ivx]{1,4})\)|[•\-\*–])\s+(?P<rest>.+)$")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<rest>.+)$")
_TERMINAL = (".", ";", ":", "?", "!")
_PAPER_SINGLE_HEADINGS = {"abstract", "introduction", "background", "methods", "methodology",
                          "experiments", "results", "discussion", "conclusion", "conclusions",
                          "limitations", "references", "acknowledgements", "acknowledgments",
                          "appendix", "appendices"}
_ABSTRACT_LINE = re.compile(r"^abstract(?:\s*[:—–]\s*(?P<body>.*))?$", re.IGNORECASE)


@dataclass
class Block:
    text: str
    section_path: list[str] = field(default_factory=list)
    section_title: Optional[str] = None
    page: Optional[int] = None
    is_list_item: bool = False
    list_label: Optional[str] = None


def _looks_like_title(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 10 or text.endswith(_TERMINAL[:2]):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(c.isupper() for c in letters) / len(letters) > 0.8:   # ALL CAPS
        return True
    # Title Case with no terminal punctuation, e.g. "Attendance Requirements"
    caps = sum(w[0].isupper() for w in words if w[0].isalpha() and len(w) > 3)
    long_words = sum(1 for w in words if w[0].isalpha() and len(w) > 3)
    return long_words > 0 and caps == long_words and not text.endswith(_TERMINAL)


def _paper_heading(text: str) -> bool:
    """Conservative paper heading check; equations and table cells stay as body."""
    t = text.strip()
    words = t.split()
    if not words or len(words) > 8 or len(t) > 85:
        return False
    if any(ch in t for ch in "=∈∑√≤≥×÷|{}[]@") or re.search(r"\d", t):
        return False
    if t.endswith(_TERMINAL) or t.lower().startswith(("figure ", "table ", "algorithm ")):
        return False
    if len(words) == 1:
        return t.lower() in _PAPER_SINGLE_HEADINGS
    if not any(ch.islower() for ch in t):
        return False                 # acronym-heavy table headers, e.g. "EN-FR EN-DE"
    return _looks_like_title(t)


def _is_body(text: str) -> bool:
    return len(text.split()) > 10 or text.rstrip().endswith((".", ";", ":"))


class _SectionStack:
    def __init__(self) -> None:
        self.path: list[tuple[int, str]] = []
        self.title: Optional[str] = None

    def push(self, level: int, label: str, title: Optional[str]) -> None:
        self.path = [(lv, lb) for lv, lb in self.path if lv < level] + [(level, label)]
        self.title = title

    @property
    def labels(self) -> list[str]:
        return [lb for _, lb in self.path]


def build_blocks(lines: list[RawLine], paper_mode: bool = False) -> list[Block]:
    if paper_mode:
        # Research PDFs often put the title, authors and affiliations in
        # separate text boxes. They are metadata, not nested prose headings.
        abstract = next((i for i, line in enumerate(lines[:40])
                         if _ABSTRACT_LINE.fullmatch(line.text.strip())), None)
        if abstract is not None:
            lines = lines[abstract:]
            m = _ABSTRACT_LINE.fullmatch(lines[0].text.strip())
            if m and m.group("body"):
                first = lines[0]
                lines = [RawLine("Abstract", first.page, first.block, first.blank_before),
                         RawLine(m.group("body"), first.page, first.block + 1, True), *lines[1:]]
        # MuPDF sometimes separates a section number from its title.
        joined: list[RawLine] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (i + 1 < len(lines) and re.fullmatch(r"\d{1,2}(?:\.\d{1,2})*", line.text.strip())
                    and lines[i + 1].page == line.page and _paper_heading(lines[i + 1].text)):
                joined.append(RawLine(f"{line.text.strip()} {lines[i + 1].text.strip()}",
                                      line.page, line.block, line.blank_before))
                i += 2
            else:
                joined.append(line)
                i += 1
        lines = joined
    stack = _SectionStack()
    blocks: list[Block] = []
    current: Optional[Block] = None
    current_block_id: Optional[int] = None

    def flush() -> None:
        nonlocal current
        if current and current.text.strip():
            blocks.append(current)
        current = None

    def start(text: str, page, is_list=False, label=None) -> None:
        nonlocal current
        flush()
        current = Block(text=text, section_path=stack.labels, section_title=stack.title,
                        page=page, is_list_item=is_list, list_label=label)

    for line in lines:
        text = line.text.strip()
        if not text:
            continue

        m = _MARKDOWN_HEADING.match(text)
        if m:
            flush()
            stack.push(1, m["rest"].strip(), m["rest"].strip())
            current_block_id = line.block
            continue

        m = _KEYWORD_HEADING.match(text)
        if m:
            label = f"{m['kw'].title()} {m['num']}"
            level = m["num"].count(".") + 1
            rest = m["rest"].strip()
            if rest and _is_body(rest):
                stack.push(level, label, None)
                start(rest, line.page)
            else:
                flush()
                stack.push(level, label, rest or None)
            current_block_id = line.block
            continue

        m = _NUMBERED.match(text)
        if paper_mode and m and (int(m["num"].split(".")[0]) > 15 or
                                 not (_paper_heading(m["rest"]) or
                                      (len(m["rest"].split()) == 1 and m["rest"].isalpha()
                                       and m["rest"][0].isupper()))):
            m = None
        if m and not re.match(r"^\d[\d,]*\s*(%|percent|days?|hours?)", text, re.IGNORECASE):
            num, rest = m["num"], m["rest"].strip()
            level = num.count(".") + 1
            if _is_body(rest):
                stack.push(level, num, stack.title if level > 1 else None)
                start(rest, line.page)
            else:
                flush()
                stack.push(level, num, rest)
            current_block_id = line.block
            continue

        m = _LIST_ITEM.match(text)
        if m:
            start(m["rest"].strip(), line.page, True, m["label"])
            current_block_id = line.block
            continue

        # Font information is unreliable in extracted PDFs. An arbitrary
        # Title Case line is often a table header, so paper sections without
        # numbering are limited to conventional standalone headings.
        paper_standalone = text.lower() in _PAPER_SINGLE_HEADINGS
        if (paper_standalone if paper_mode else _looks_like_title(text)) and (line.blank_before or current is None):
            flush()
            stack.push(1 if paper_mode or not stack.path else stack.path[-1][0] + 1, text, text)
            current_block_id = line.block
            continue

        # Continuation of the current paragraph or a new paragraph.
        if (current is not None and
                ((line.block == current_block_id and
                  (not line.blank_before or
                   (paper_mode and not current.text.rstrip().endswith(_TERMINAL)))) or
                 (paper_mode and (text[:1].islower() or text[:1] in "[(")
                  and not current.text.rstrip().endswith(_TERMINAL)
                  and not current.is_list_item))):
            current.text = f"{current.text} {text}"
        else:
            start(text, line.page)
        current_block_id = line.block

    flush()
    return blocks

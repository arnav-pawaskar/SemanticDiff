"""Recognise numeric scholarly references such as [12] and [4, 7-9]."""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[(?:\s*\d+\s*(?:[,;–-]\s*\d+\s*)*)\]")
_AUTHOR_YEAR = re.compile(r"\bet\s+al\.,?\s*(?:19|20)\d{2}[a-z]?\b", re.I)


def _all_citations(text: str):
    return sorted([*_CITATION.finditer(text), *_AUTHOR_YEAR.finditer(text)],
                  key=lambda match: match.start())


def citation_char_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _all_citations(text)]


def citation_token_indices(doc) -> set[int]:
    spans = citation_char_spans(doc.text)
    return {tok.i for tok in doc
            if any(tok.idx < end and tok.idx + len(tok.text) > start for start, end in spans)}


def citation_strings(text: str) -> list[str]:
    return [m.group() for m in _all_citations(text)]


def without_citations(text: str) -> str:
    """Remove reference labels before NLI; their numbers are not claim values."""
    text = _CITATION.sub("", text)
    text = _AUTHOR_YEAR.sub("", text)
    text = re.sub(r"^\d+(?=[A-Z][a-z])", "", text)  # PDF footnote labels: 3We -> We
    return re.sub(r"\s+", " ", text).strip()

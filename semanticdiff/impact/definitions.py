"""Defined-term and cross-reference extraction.

Definitions are recognised by explicit definitional patterns:
    "Eligible Student" means ...          (quoted term, preferred)
    Eligible Student shall mean ...       (Capitalised term at sentence start)
    The term "X" refers to / is defined as ...
References are occurrences of a defined term elsewhere (plural-aware, case-insensitive)
and explicit section references ("Section 4", "clause 2.1").
"""
from __future__ import annotations

import re
from typing import Optional

from semanticdiff.lexicons import DEFINITION_VERBS

_VERBS = "|".join(re.escape(v) for v in sorted(DEFINITION_VERBS, key=len, reverse=True))
_QUOTED = re.compile(rf'["“\'](?P<term>[A-Za-z][^"”\']{{1,60}}?)["”\']\s*,?\s*(?:{_VERBS})\b')
_CAPITALISED = re.compile(
    rf"^(?:the\s+term\s+)?(?P<term>[A-Z][\w-]*(?:\s+[A-Z][\w-]*){{0,4}})\s+(?:{_VERBS})\b")
SECTION_REF = re.compile(r"\b(?:section|clause|article|rule|paragraph)\s+(?P<num>\d+(?:\.\d+)*[A-Za-z]?)",
                         re.IGNORECASE)


def extract_definition(text: str) -> Optional[str]:
    m = _QUOTED.search(text) or _CAPITALISED.match(text.strip())
    if not m:
        return None
    term = m.group("term").strip()
    if term.lower() in ("this", "it", "the", "these regulations", "this agreement"):
        return None
    return term


def term_pattern(term: str) -> re.Pattern:
    """Regex matching a term with an optional plural on its last word."""
    words = term.split()
    body = r"\s+".join(re.escape(w) for w in words[:-1])
    last = re.escape(words[-1])
    plural = rf"{last}(?:s|es)?" if not words[-1].endswith("s") else last
    full = rf"{body}\s+{plural}" if body else plural
    return re.compile(rf"\b{full}\b", re.IGNORECASE)


def find_term_mentions(text: str, term: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in term_pattern(term).finditer(text)]


def find_section_refs(text: str) -> list[str]:
    return [m.group("num") for m in SECTION_REF.finditer(text)]

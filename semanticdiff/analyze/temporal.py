"""Calendar dates and clock times (durations are handled by quantities.py).

Uses spaCy DATE / TIME entities; any entity that overlaps a duration quantity
("within 30 days", "every two weeks") is skipped. Remaining mentions are normalised
with dateutil when they contain an explicit calendar anchor (month name, dd/mm/yyyy,
a year, or a clock time); otherwise the lower-cased text is kept for comparison
("weekends", "working days").
"""
from __future__ import annotations

import re
from datetime import datetime

from dateutil import parser as dateparser

from semanticdiff.models import DateMention, Quantity

_MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_ANCHOR = re.compile(rf"\b{_MONTHS}\b|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\b(19|20)\d{{2}}\b|"
                     r"\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)|\bnoon\b|\bmidnight\b", re.IGNORECASE)
_DEFAULT = datetime(1900, 1, 1)


def _normalise(text: str) -> str | None:
    if not _ANCHOR.search(text):
        return None
    cleaned = re.sub(r"\b(on|by|from|until|till|before|after|the|of)\b", " ", text, flags=re.IGNORECASE)
    try:
        dt = dateparser.parse(cleaned, default=_DEFAULT, dayfirst=True, fuzzy=True)
    except (ValueError, OverflowError):
        return None
    has_time = bool(re.search(r"\d\s*(am|pm|a\.m|p\.m)|:\d{2}|noon|midnight", text, re.IGNORECASE))
    has_date = bool(re.search(rf"{_MONTHS}|\d{{1,2}}[/-]\d{{1,2}}|\b(19|20)\d{{2}}\b", text, re.IGNORECASE))
    if has_date and has_time:
        return dt.strftime("%Y-%m-%dT%H:%M")
    if has_time:
        return dt.strftime("T%H:%M")
    year = dt.strftime("%Y") if dt.year != 1900 else "----"
    return f"{year}-{dt.strftime('%m-%d')}"


_CLOCK = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)(?=\W|$)|\bnoon\b|\bmidnight\b", re.IGNORECASE)
_EXPLICIT_DATE = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\b(?:,?\s+\d{{4}})?|\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)


def extract_dates(doc, quantities: list[Quantity]) -> list[DateMention]:
    """Clock times and explicit calendar dates are found by pattern (NER misses many of
    them, e.g. "before 2 PM"); remaining spaCy DATE/TIME entities that are not durations
    are kept as relative dates ("weekends", "the same day")."""
    q_spans = [q.span for q in quantities if q.dimension in ("time", "business_time", "age")]
    out: list[DateMention] = []
    taken: list[tuple[int, int]] = []
    overlaps = lambda s, e, spans: any(a < e and b > s for a, b in spans)
    for rx in (_EXPLICIT_DATE, _CLOCK):
        for m in rx.finditer(doc.text):
            s, e = m.span()
            if overlaps(s, e, taken) or overlaps(s, e, q_spans):
                continue
            text = m.group(0)
            out.append(DateMention(raw=text, normalized=_normalise(text) or text.lower(), span=(s, e)))
            taken.append((s, e))
    for ent in doc.ents:
        if ent.label_ not in ("DATE", "TIME"):
            continue
        s, e = ent.start_char, ent.end_char
        if overlaps(s, e, q_spans) or overlaps(s, e, taken):
            continue
        root = ent.root
        if root.dep_ in ("amod", "compound", "nmod") and not (ent.start <= root.head.i < ent.end) \
                and root.head.pos_ in ("NOUN", "PROPN"):
            continue                 # "final-year students", "annual leave": a modifier, not a date
        text = ent.text.strip()
        out.append(DateMention(raw=text, normalized=_normalise(text) or text.lower(), span=(s, e)))
    out.sort(key=lambda d: d.span[0])
    return out

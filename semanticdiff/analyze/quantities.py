"""Numeric and quantity extraction with unit normalisation.

Handles digits ("5,000", "7.5", "99.9%"), number words ("seven", "twenty-five",
"one and a half"), fractions ("half a second"), implicit one ("a week",
"every fortnight"), currencies (Rs./INR/$/EUR ...), and units with a dimension and a
conversion factor to the dimension's base unit. Each quantity also records a comparator
("at least", "within", "not exceed" ...) found immediately before it.

Two quantities with equal ``si_value`` in the same dimension are *equivalent*
("500 milliseconds" == "half a second"), which is what lets SemanticDiff call a
rewrite meaning-preserving on a principled basis.
"""
from __future__ import annotations

import re
from typing import Optional

from semanticdiff.lexicons import COMPARATORS
from semanticdiff.models import Quantity

# unit surface form (lowercase) -> (canonical unit, dimension, factor to base unit, approximate?)
_TIME = {
    "millisecond": 0.001, "ms": 0.001, "msec": 0.001,
    "second": 1, "sec": 1, "secs": 1, "s": 1,
    "minute": 60, "min": 60, "mins": 60,
    "hour": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
    "day": 86400, "week": 604800, "fortnight": 1209600,
    "month": 2629800, "year": 31557600, "yr": 31557600, "decade": 315576000,
}
_APPROX_TIME = {"month", "year", "yr", "decade"}
UNITS: dict[str, tuple[str, str, float, bool]] = {}
for _u, _f in _TIME.items():
    canon = {"ms": "millisecond", "msec": "millisecond", "sec": "second", "secs": "second",
             "s": "second", "min": "minute", "mins": "minute", "hr": "hour", "hrs": "hour",
             "h": "hour", "yr": "year"}.get(_u, _u)
    UNITS[_u] = (canon, "time", float(_f), _u in _APPROX_TIME)
for _u in ("%", "percent", "percentage", "pc"):
    UNITS[_u] = ("percent", "percent", 1.0, False)
for _u, _f, _c in [("mm", 0.001, "metre"), ("cm", 0.01, "metre"), ("km", 1000.0, "metre"),
                   ("metre", 1.0, "metre"), ("meter", 1.0, "metre"), ("kilometre", 1000.0, "metre"),
                   ("kilometer", 1000.0, "metre"), ("mile", 1609.34, "metre"), ("foot", 0.3048, "metre"),
                   ("feet", 0.3048, "metre"), ("inch", 0.0254, "metre")]:
    UNITS[_u] = (_c, "length", _f, False)
for _u, _f in [("mg", 1e-6), ("g", 1e-3), ("gram", 1e-3), ("kg", 1.0), ("kilogram", 1.0), ("tonne", 1000.0)]:
    UNITS[_u] = ("kilogram", "mass", _f, False)
for _u, _f in [("kb", 1e3), ("mb", 1e6), ("gb", 1e9), ("tb", 1e12), ("byte", 1.0)]:
    UNITS[_u] = ("byte", "data", _f, False)

# Business time is deliberately a different dimension: 10 days != 10 working days.
BUSINESS_UNITS = {"working day": "working_day", "business day": "working_day",
                  "working hour": "working_hour", "business hour": "working_hour"}

CURRENCY_PREFIX = {"rs": "INR", "rs.": "INR", "inr": "INR", "₹": "INR", "$": "USD", "usd": "USD",
                   "us$": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP"}
CURRENCY_SUFFIX = {"rupee": "INR", "rupees": "INR", "inr": "INR", "dollar": "USD", "dollars": "USD",
                   "usd": "USD", "euro": "EUR", "euros": "EUR", "eur": "EUR"}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "dozen": 12,
}
MULTIPLIERS = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
               "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000, "billion": 1_000_000_000}
FRACTIONS = {"half": 0.5, "quarter": 0.25}
IMPLICIT_ONE = {"a", "an", "every", "each"}
REFERENCE_WORDS = {"section", "clause", "article", "rule", "regulation", "chapter", "part",
                   "paragraph", "schedule", "annexure", "appendix", "sec", "sec.", "no", "no."}
AGE_WORDS = {"old", "age"}

_NUMERIC_RE = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?$")
_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")
_SORTED_COMPARATORS = sorted(COMPARATORS, key=lambda s: -len(s.split()))
_EXTRA_COMPARATORS = {"not exceed": "at_most", "shall not exceed": "at_most", "exceed": "more_than",
                      "not to exceed": "at_most"}


def _num_value(tok_text: str) -> Optional[float]:
    t = tok_text.lower().replace(",", "")
    if _NUMERIC_RE.match(t):
        try:
            return float(t)
        except ValueError:
            return None
    m = _FRACTION_RE.match(t)
    if m and int(m[2]) != 0:
        return int(m[1]) / int(m[2])
    if t in NUMBER_WORDS:
        return float(NUMBER_WORDS[t])
    if "-" in t:                       # twenty-five
        parts = t.split("-")
        if all(p in NUMBER_WORDS for p in parts):
            return float(sum(NUMBER_WORDS[p] for p in parts))
    return None


def _unit_at(doc, j: int) -> tuple[Optional[tuple[str, str, float, bool]], int]:
    """Unit starting at token j -> ((canon, dim, factor, approx), tokens consumed)."""
    if j >= len(doc):
        return None, 0
    if j + 1 < len(doc):
        two = f"{doc[j].lower_} {doc[j + 1].lemma_.lower()}"
        if two in BUSINESS_UNITS:
            return (BUSINESS_UNITS[two], "business_time", 1.0, False), 2
        if two in ("calendar day", "calendar month", "calendar year"):
            return UNITS[doc[j + 1].lemma_.lower()], 2
        if two in ("per cent", "percentage point"):
            return UNITS["%"], 2
    low, lem = doc[j].lower_, doc[j].lemma_.lower()
    for key in (low, lem):
        if key in UNITS:
            # single letters are units only right after a digit ("5 s", "10 h")
            if len(key) == 1 and key != "%" and not (j > 0 and doc[j - 1].like_num):
                return None, 0
            return UNITS[key], 1
    return None, 0


def _comparator_before(doc, start: int) -> Optional[str]:
    """Comparator phrase ending right before token ``start`` (longest match),
    or 'minimum/maximum <noun>* of' within four tokens."""
    left = [t.lower_ for t in doc[max(0, start - 5):start]]
    text = " ".join(left)
    for phrase, canon in list(_EXTRA_COMPARATORS.items()):
        if text.endswith(phrase) or text.endswith(phrase + "ing"):
            return canon
    for phrase in _SORTED_COMPARATORS:
        if text == phrase or text.endswith(" " + phrase):
            return COMPARATORS[phrase]
    if left and left[-1] == "of":
        for w in reversed(left[:-1][-3:]):
            if w in ("minimum", "min"):
                return "at_least"
            if w in ("maximum", "max"):
                return "at_most"
    return None


def _parse_number(doc, i: int) -> tuple[Optional[float], int]:
    """Parse a number expression starting at token i -> (value, next index)."""
    toks = doc
    n = len(toks)
    low = toks[i].lower_
    # fractions: "half a second", "half an hour", "one and a half hours"
    if low in FRACTIONS:
        j = i + 1
        if j < n and toks[j].lower_ in ("a", "an"):
            j += 1
        return FRACTIONS[low], j
    if low in IMPLICIT_ONE:
        unit, _ = _unit_at(doc, i + 1)
        if unit and unit[1] in ("time", "business_time"):
            return 1.0, i + 1
        return None, i + 1
    v = _num_value(toks[i].text)
    if v is None:
        return None, i + 1
    total, current, j = 0.0, v, i + 1
    # word-number continuation: "twenty five", "one hundred", "5 lakh", "one and a half"
    while j < n:
        w = toks[j].lower_
        if w in MULTIPLIERS:
            current *= MULTIPLIERS[w]
            if MULTIPLIERS[w] >= 1000:
                total, current = total + current, 0.0
            j += 1
        elif w in NUMBER_WORDS and toks[j - 1].lower_ in NUMBER_WORDS | set(MULTIPLIERS):
            current += NUMBER_WORDS[w]
            j += 1
        elif w == "and" and j + 2 < n and toks[j + 1].lower_ == "a" and toks[j + 2].lower_ in FRACTIONS:
            current += FRACTIONS[toks[j + 2].lower_]
            j += 3
        else:
            break
    return total + current, j


def extract_quantities(doc) -> list[Quantity]:
    out: list[Quantity] = []
    n = len(doc)
    i = 0
    date_tokens = {t.i for ent in doc.ents if ent.label_ in ("DATE", "TIME") for t in ent}
    while i < n:
        tok = doc[i]
        low = tok.lower_
        is_candidate = (tok.like_num or low in NUMBER_WORDS or low in FRACTIONS
                        or (low in IMPLICIT_ONE and _unit_at(doc, i + 1)[0] is not None))
        if not is_candidate or (i > 0 and doc[i - 1].lower_ in REFERENCE_WORDS):
            i += 1
            continue
        value, j = _parse_number(doc, i)
        if value is None:
            i = max(j, i + 1)
            continue
        start = i
        # currency prefix: "Rs. 500", "$500", "INR 500"
        currency = None
        if i > 0 and doc[i - 1].lower_ in CURRENCY_PREFIX:
            currency, start = CURRENCY_PREFIX[doc[i - 1].lower_], i - 1
        elif i > 1 and doc[i - 1].text == "." and doc[i - 2].lower_ in CURRENCY_PREFIX:
            currency, start = CURRENCY_PREFIX[doc[i - 2].lower_], i - 2

        unit, used = _unit_at(doc, j)
        end = j + used
        if currency is None and j < n and doc[j].lower_ in CURRENCY_SUFFIX:
            currency, end = CURRENCY_SUFFIX[doc[j].lower_], j + 1
            unit = None
        # dependency fallback: "20 annual leave days" -> nummod of a unit noun further right
        if unit is None and currency is None and tok.dep_ == "nummod":
            head_unit, _ = _unit_at(doc, tok.head.i)
            if head_unit and tok.head.i > i:
                unit = head_unit
                if any(c.lower_ in ("working", "business") for c in tok.head.children):
                    unit = ("working_day", "business_time", 1.0, False)
                end = tok.head.i + 1

        approximate = False
        if currency:
            canon, dim, si = currency, f"currency", value
        elif unit:
            canon, dim, factor, approximate = unit
            si = value * factor
            if dim == "time" and end < n and doc[end].lower_ in AGE_WORDS:
                dim = "age"
        else:
            # bare number inside a calendar date / clock time is handled by temporal.py
            if i in date_tokens and not (low in NUMBER_WORDS and value < 100):
                i = j
                continue
            if low in IMPLICIT_ONE:
                i = j
                continue
            canon, dim, si = None, "count", value

        last = doc[end - 1] if end > start else doc[start]
        head_tok = last.head if last.head.i != last.i else last
        while (head_tok.is_punct or head_tok.lower_ in CURRENCY_PREFIX) and head_tok.head.i != head_tok.i:
            head_tok = head_tok.head
        span = (doc[start].idx, last.idx + len(last.text))
        out.append(Quantity(
            raw=doc.text[span[0]:span[1]], value=value, unit=canon, dimension=dim,
            si_value=round(si, 9), span=span, comparator=_comparator_before(doc, start),
            head=head_tok.lemma_.lower(), approximate=approximate))
        i = max(end, j)
    return out


def dimension_family(dim: str) -> str:
    if dim in ("time", "business_time"):
        return "time"
    return dim


def is_temporal(q: Quantity) -> bool:
    return q.dimension in ("time", "business_time")


def equivalent(a: Quantity, b: Quantity) -> bool:
    """Same dimension and same value after conversion to the base unit."""
    if a.dimension != b.dimension:
        return False
    if a.dimension in ("currency", "business_time") and a.unit != b.unit:
        return False
    return abs(a.si_value - b.si_value) <= 1e-9 * max(1.0, abs(a.si_value))


def describe(q: Quantity) -> str:
    comp = {"at_least": "≥", "at_most": "≤", "more_than": ">", "less_than": "<",
            "within": "≤", "up_to": "≤", "exactly": "="}.get(q.comparator or "", "")
    return f"{comp}{q.raw}".strip()

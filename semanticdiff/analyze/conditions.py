"""Condition extraction.

Three sources, all span-based:
  1. adverbial clauses (dep ``advcl``) introduced by a conditional marker
     (``mark`` in if / unless / provided / when / once ...),
  2. multi-word condition openers found by pattern ("subject to", "provided that",
     "as long as", "except when" ...), running to the next clause boundary,
  3. condition prepositional phrases: a preposition (with / upon / after / without ...)
     whose object is a condition noun (approval, permission, certificate ...) and that
     is NOT attached inside the subject noun phrase. The attachment test is what
     separates "submit applications *with faculty approval*" (condition) from
     "students *with medical exemptions*" (scope modifier of the subject).
"""
from __future__ import annotations

import re

from semanticdiff.lexicons import (CONDITION_MARKS, CONDITION_NOUNS, CONDITION_PHRASES,
                                   CONDITION_PREPOSITIONS)
from semanticdiff.models import Condition, ScopeInfo

_PHRASE_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in sorted(CONDITION_PHRASES, key=len, reverse=True)) + r")\b",
                        re.IGNORECASE)
_BOUNDARY = re.compile(r"[,;:]|\.\s*$|\.$")


def _overlaps(span, spans) -> bool:
    return any(span[0] < e and span[1] > s for s, e in spans)


def _trim(doc, start: int, end: int) -> tuple[int, int]:
    text = doc.text
    while end > start and text[end - 1] in " .,;:":
        end -= 1
    while start < end and text[start] in " ,;:":
        start += 1
    return start, end


def extract_conditions(doc, agent: ScopeInfo | None = None) -> list[Condition]:
    out: list[Condition] = []
    spans: list[tuple[int, int]] = []

    # 2. multi-word openers (run first: "only if", "provided that", "except when" win over "if")
    lowered = doc.text
    for m in _PHRASE_RE.finditer(lowered):
        marker, polarity = CONDITION_PHRASES[m.group(1).lower()]
        rest = lowered[m.end():]
        b = _BOUNDARY.search(rest)
        end = m.end() + (b.start() if b else len(rest))
        s, e = _trim(doc, m.start(), end)
        if e - s > len(m.group(1)) and not _overlaps((s, e), spans):
            out.append(Condition(marker=marker, text=doc.text[s:e], span=(s, e), polarity=polarity))
            spans.append((s, e))

    # 1. adverbial clauses with a conditional subordinator
    for tok in doc:
        if tok.dep_ != "advcl":
            continue
        mark = next((c for c in tok.children if c.dep_ == "mark" and c.lower_ in CONDITION_MARKS), None)
        if mark is None:
            continue
        marker, polarity = CONDITION_MARKS[mark.lower_]
        sub = list(tok.subtree)
        s, e = _trim(doc, min(mark.idx, sub[0].idx), sub[-1].idx + len(sub[-1].text))
        if not _overlaps((s, e), spans):
            out.append(Condition(marker=marker, text=doc.text[s:e], span=(s, e), polarity=polarity))
            spans.append((s, e))

    # 3. condition prepositional phrases, not attached inside the subject NP
    for tok in doc:
        if tok.dep_ != "prep" or tok.lower_ not in CONDITION_PREPOSITIONS:
            continue
        pobj = next((c for c in tok.children if c.dep_ == "pobj"), None)
        if pobj is None or pobj.lemma_.lower() not in CONDITION_NOUNS:
            continue
        if agent is not None and agent.span[0] <= tok.idx < agent.span[1]:
            continue                                   # scope modifier, not a condition
        sub = list(tok.subtree)
        s, e = _trim(doc, sub[0].idx, sub[-1].idx + len(sub[-1].text))
        if not _overlaps((s, e), spans):
            polarity = "negative" if tok.lower_ == "without" else "positive"
            out.append(Condition(marker=tok.lower_, text=doc.text[s:e], span=(s, e), polarity=polarity))
            spans.append((s, e))

    out.sort(key=lambda c: c.span[0])
    return out

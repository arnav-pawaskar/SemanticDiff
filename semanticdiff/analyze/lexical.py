"""Layer 1 — lexical analysis: what tokens literally changed.

Produces token-level diff opcodes (difflib) aligned to spaCy tokens, a textual-change
score (1 - word-level similarity ratio) and the sets of changed token indices that the
span-ownership step later tries to explain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from semanticdiff.lexicons import STOP_EDIT_TOKENS
from semanticdiff.models import Severity


@dataclass
class DiffOp:
    op: str                 # equal | replace | delete | insert
    i1: int
    i2: int
    j1: int
    j2: int
    old: str
    new: str


@dataclass
class LexicalDiff:
    ops: list[DiffOp]
    textual_change: float
    changed_old: set[int] = field(default_factory=set)   # token indices in v1 doc
    changed_new: set[int] = field(default_factory=set)   # token indices in v2 doc

    def content_changed(self, doc1, doc2) -> tuple[set[int], set[int]]:
        """Changed token indices excluding punctuation and function words."""
        keep = lambda t: not t.is_punct and not t.is_space and t.lower_ not in STOP_EDIT_TOKENS
        return ({i for i in self.changed_old if keep(doc1[i])},
                {j for j in self.changed_new if keep(doc2[j])})


def _key(tok) -> str:
    return tok.lower_


def token_diff(doc1, doc2) -> LexicalDiff:
    a = [_key(t) for t in doc1]
    b = [_key(t) for t in doc2]
    sm = SequenceMatcher(None, a, b, autojunk=False)
    ops, ch_old, ch_new = [], set(), set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old = doc1[i1:i2].text_with_ws if i2 > i1 else ""
        new = doc2[j1:j2].text_with_ws if j2 > j1 else ""
        ops.append(DiffOp(tag, i1, i2, j1, j2, old, new))
        if tag != "equal":
            ch_old.update(range(i1, i2))
            ch_new.update(range(j1, j2))

    # Textual change on words only (punctuation-insensitive), in [0, 1].
    wa = [t.lower_ for t in doc1 if not t.is_punct and not t.is_space]
    wb = [t.lower_ for t in doc2 if not t.is_punct and not t.is_space]
    ratio = SequenceMatcher(None, wa, wb, autojunk=False).ratio() if (wa or wb) else 1.0
    return LexicalDiff(ops=ops, textual_change=round(1.0 - ratio, 4),
                       changed_old=ch_old, changed_new=ch_new)


def textual_level(score: float, cfg: dict) -> Severity:
    lv = cfg.get("textual_levels", {"medium": 0.2, "high": 0.5})
    if score >= lv["high"]:
        return Severity.HIGH
    if score >= lv["medium"]:
        return Severity.MEDIUM
    return Severity.LOW


def char_span_to_tokens(doc, span: tuple[int, int]) -> set[int]:
    s, e = span
    return {t.i for t in doc if t.idx < e and t.idx + len(t.text) > s}

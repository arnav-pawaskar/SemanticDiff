"""Baselines for Task A (did meaning change?) and Task B (what changed?).

Task A — progressively stronger scorers; each gets a decision threshold tuned on dev
(maximising macro-F1 over both classes):
  1. lexical      : word-level edit dissimilarity (1 - difflib ratio)
  2. tfidf        : 1 - TF-IDF cosine
  3. embedding    : 1 - SBERT cosine
  4. embed+nli    : material unless (cosine >= t1 and bidirectional entailment >= t2)
  5. semanticdiff : the full hybrid pipeline (no tuning of its decision here)

Task B — typed change detection:
  * keyword-diff  : token diff + lexicon lookup, *no* parsing, *no* normalisation
  * semanticdiff  : categories of the atomic changes produced by the pipeline
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from semanticdiff.lexicons import (ANTONYMS, CONDITION_MARKS, CONDITION_PHRASES, MODAL_AUX,
                                   NEGATION_TOKENS, QUANTIFIERS)

CATEGORIES = ["NUMERIC", "TEMPORAL", "ENTITY", "MODALITY", "NEGATION", "SCOPE", "CONDITION", "REVERSAL"]
_TIME_WORDS = {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
               "minute", "minutes", "second", "seconds", "millisecond", "milliseconds", "am", "pm",
               "january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december", "working"}
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d[\d,.]*%?|%")


def words(s: str) -> list[str]:
    return _WORD.findall(s)


# ----------------------------------------------------------------------------- Task A scores
def lexical_dissimilarity(a: str, b: str) -> float:
    wa, wb = [w.lower() for w in words(a)], [w.lower() for w in words(b)]
    return 1.0 - SequenceMatcher(None, wa, wb, autojunk=False).ratio()


def tfidf_dissimilarity(pairs: list[tuple[str, str]], corpus: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(corpus)
    A = vec.transform([a for a, _ in pairs])
    B = vec.transform([b for _, b in pairs])
    return 1.0 - np.asarray(A.multiply(B).sum(axis=1)).ravel()


def _macro_f1(gold, pred) -> float:
    """Macro-F1 over BOTH classes. Plain F1 on a 77%-positive set is maximised by
    'always material', which would make every threshold baseline degenerate."""
    from sklearn.metrics import f1_score
    return f1_score(gold, pred, average="macro", zero_division=0)


def tune_threshold(scores: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    """Threshold t maximising macro-F1 of the rule 'material iff score >= t'."""
    best_t, best_f = 0.5, -1.0
    for t in np.unique(np.concatenate([scores, [scores.min() - 1e-6, scores.max() + 1e-6]])):
        f = _macro_f1(gold, scores >= t)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def tune_embed_nli(cos: np.ndarray, ent: np.ndarray, gold: np.ndarray) -> tuple[float, float, float]:
    best = (0.0, 0.5, -1.0)
    for t1 in np.linspace(0.0, 1.0, 41):
        for t2 in np.linspace(0.05, 0.95, 19):
            pred = ~((cos >= t1) & (ent >= t2))
            f = _macro_f1(gold, pred)
            if f > best[2]:
                best = (float(t1), float(t2), f)
    return best


# ----------------------------------------------------------------------------- Task B keyword baseline
_MODAL_WORDS = set(MODAL_AUX) | {"required", "permitted", "allowed", "entitled", "obliged", "optional",
                                 "mandatory", "compulsory", "recommended", "prohibited", "forbidden"}
_COND_WORDS = set(CONDITION_MARKS) | {w for p in CONDITION_PHRASES for w in p.split()} | {"approval", "consent"}


def keyword_diff_labels(a: str, b: str) -> set[str]:
    """Surface-level typing: look at which *words* were inserted/deleted/replaced."""
    wa, wb = words(a), words(b)
    la, lb = [w.lower() for w in wa], [w.lower() for w in wb]
    sm = SequenceMatcher(None, la, lb, autojunk=False)
    out: set[str] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old, new = la[i1:i2], lb[j1:j2]
        changed = old + new
        ctx = set(la[max(0, i1 - 1):i2 + 2]) | set(lb[max(0, j1 - 1):j2 + 2])
        if any(w in NEGATION_TOKENS for w in changed) or any(w in ("cannot",) for w in changed):
            out.add("NEGATION")
        if any(w in _MODAL_WORDS for w in changed):
            out.add("MODALITY")
        if any(re.match(r"\d", w) or w in ("one", "two", "three", "half", "seven", "ten") for w in changed):
            out.add("TEMPORAL" if ctx & _TIME_WORDS else "NUMERIC")
        elif any(w in _TIME_WORDS for w in changed):
            out.add("TEMPORAL")
        if any(w in _COND_WORDS for w in new) or any(w in _COND_WORDS for w in old):
            out.add("CONDITION")
        if any(w in QUANTIFIERS for w in changed) or (tag == "insert" and i1 < 3) or (tag == "delete" and i1 < 3):
            out.add("SCOPE")
        if any(wa[k][:1].isupper() for k in range(i1, i2) if k > 0) or \
                any(wb[k][:1].isupper() for k in range(j1, j2) if k > 0):
            out.add("ENTITY")
        if any(n in ANTONYMS.get(o, ()) or _prefix_negated(o, n) for o in old for n in new):
            out.add("REVERSAL")
    return out


def _prefix_negated(x: str, y: str) -> bool:
    """'valid' vs 'invalid', 'refundable' vs 'non-refundable', 'available' vs 'unavailable'."""
    forms = lambda w: {f"un{w}", f"in{w}", f"non-{w}", f"non{w}", f"dis{w}"}
    return y in forms(x) or x in forms(y)

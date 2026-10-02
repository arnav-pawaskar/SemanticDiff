"""Frame diff: turn two ProvisionFrames into atomic semantic changes.

Span ownership
--------------
Every atomic change *claims* the diff tokens it explains. Analysers run in a fixed
precedence order —

    deontic (modality/negation) -> quantities -> conditions -> dates -> scope
    -> entities -> antonym reversal

— and a token that is already claimed cannot give rise to a second change. This is what
keeps "30 days -> 15 days" inside a condition from being reported both as a temporal
change and as a modified condition, and what lets us report how much of the lexical
diff is *explained*. Changed content tokens that nobody claims are the residual that
the verdict step (changes/verdict.py) reasons about with NLI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

from rapidfuzz import fuzz

from semanticdiff.analyze.deontic import POLARITY
from semanticdiff.analyze.entities import NUMERIC_LABELS, entity_span_for_token
from semanticdiff.analyze.lexical import LexicalDiff, char_span_to_tokens
from semanticdiff.analyze.quantities import describe, dimension_family, equivalent, is_temporal
from semanticdiff.analyze.scope import compare_scope, modifier_lemma_set
from semanticdiff.lexicons import ANTONYMS, LOWER_BOUND, MODAL_STRENGTH, UPPER_BOUND, comparator_class
from semanticdiff.models import (AtomicChange, ChangeCategory as C, Condition, DeonticInfo,
                                 ProvisionFrame, Quantity, ScopeInfo)

_STATE_LABEL = {
    "OBLIGATION": "Obligation", "PROHIBITION": "Prohibition", "RECOMMENDATION": "Recommendation",
    "DISCOURAGED": "Discouraged", "PERMISSION": "Permission", "NO_OBLIGATION": "No obligation",
    "ASSERTION": "Assertion", "NEGATED_ASSERTION": "Negated assertion",
}


@dataclass
class Ownership:
    doc1: object
    doc2: object
    lex: LexicalDiff
    old: set[int] = field(default_factory=set)
    new: set[int] = field(default_factory=set)

    def _tok(self, doc, span):
        return char_span_to_tokens(doc, span) if span else set()

    def changed_in(self, span1=None, span2=None) -> tuple[set[int], set[int]]:
        return (self._tok(self.doc1, span1) & self.lex.changed_old,
                self._tok(self.doc2, span2) & self.lex.changed_new)

    def claim(self, span1=None, span2=None, only_changed: bool = True) -> None:
        t1, t2 = self._tok(self.doc1, span1), self._tok(self.doc2, span2)
        if only_changed:
            t1, t2 = t1 & self.lex.changed_old, t2 & self.lex.changed_new
        self.old |= t1
        self.new |= t2

    def is_free(self, span1=None, span2=None) -> bool:
        """True unless every changed token in the spans is already claimed."""
        c1, c2 = self.changed_in(span1, span2)
        if not c1 and not c2:
            return True
        return bool((c1 - self.old) or (c2 - self.new))


@dataclass
class FrameDiff:
    changes: list[AtomicChange]
    notes: list[str]
    residual_old: list[str]
    residual_new: list[str]
    explained_ratio: float


def _text(doc, span) -> Optional[str]:
    return doc.text[span[0]:span[1]] if span else None


def _union(*spans) -> Optional[tuple[int, int]]:
    spans = [s for s in spans if s]
    return (min(s[0] for s in spans), max(s[1] for s in spans)) if spans else None


# ================================================================== deontic
def _pair_predicates(p1: list[DeonticInfo], p2: list[DeonticInfo]):
    pairs, used2 = [], set()
    for a in p1:
        for k, b in enumerate(p2):
            if k not in used2 and a.predicate == b.predicate:
                pairs.append((a, b)); used2.add(k); break
    left1 = [a for a in p1 if all(a is not x for x, _ in pairs)]
    left2 = [b for k, b in enumerate(p2) if k not in used2]
    # Fallback: pair the remaining predicates positionally (main clause <-> main clause),
    # e.g. "must respond" <-> "shall return a response".
    for a, b in zip(left1, left2):
        pairs.append((a, b))
    return pairs


def _deontic_changes(f1, f2, own: Ownership, new_id, notes) -> list[AtomicChange]:
    out = []
    for a, b in _pair_predicates(f1.predicates, f2.predicates):
        span_a = _union(a.trigger_span, a.predicate_span)
        span_b = _union(b.trigger_span, b.predicate_span)
        if a.state == b.state:
            if a.trigger and b.trigger and a.trigger.lower() != b.trigger.lower():
                notes.append(f"'{a.trigger}' ≡ '{b.trigger}' (both {_STATE_LABEL[a.state.value]})")
            own.claim(a.trigger_span, b.trigger_span)
            continue
        s1, s2 = a.state.value, b.state.value
        transition = f"{_STATE_LABEL[s1]} → {_STATE_LABEL[s2]}"
        ev = {"old_state": s1, "new_state": s2, "old_trigger": a.trigger, "new_trigger": b.trigger,
              "predicate": b.predicate, "transition": f"{a.modal_class.value}->{b.modal_class.value}"}
        if a.negated != b.negated:
            sub = "INTRODUCED" if b.negated else "REMOVED"
            cue = b.negation_cue if b.negated else a.negation_cue
            out.append(AtomicChange(
                id=new_id(), category=C.NEGATION, subtype=sub,
                description=f"Negation {sub.lower()}: {transition}",
                old=_text(own.doc1, span_a), new=_text(own.doc2, span_b),
                old_span=span_a, new_span=span_b, evidence={**ev, "negation_cue": cue}))
        elif POLARITY[a.state] * POLARITY[b.state] == -1:
            out.append(AtomicChange(
                id=new_id(), category=C.REVERSAL, subtype=f"{s1}->{s2}",
                description=f"Meaning reversed: {transition}",
                old=_text(own.doc1, span_a), new=_text(own.doc2, span_b),
                old_span=span_a, new_span=span_b, evidence=ev))
        else:
            c1, c2 = a.modal_class, b.modal_class
            if c1 in MODAL_STRENGTH and c2 in MODAL_STRENGTH:
                sub = "STRENGTHENED" if MODAL_STRENGTH[c2] > MODAL_STRENGTH[c1] else "WEAKENED"
                what = "Prohibition" if a.negated else "Obligation"
                desc = f"{what} {sub.lower()}: {transition}"
            elif c1.value == "NONE":
                sub, desc = "INTRODUCED", f"Modality introduced: {transition}"
            elif c2.value == "NONE":
                sub, desc = "REMOVED", f"Modality removed: {transition}"
            else:
                sub, desc = "CHANGED", f"Modality changed: {transition}"
            out.append(AtomicChange(
                id=new_id(), category=C.MODALITY, subtype=sub, description=desc,
                old=a.trigger or None, new=b.trigger or None,
                old_span=a.trigger_span, new_span=b.trigger_span, evidence=ev))
        own.claim(a.trigger_span, b.trigger_span)
        if a.negated != b.negated:
            own.claim(span_a, span_b)
    return out


# ================================================================== conditions
def _mask_condition(cond: Condition, quantities: list[Quantity]) -> str:
    text, offset = cond.text, cond.span[0]
    for q in sorted(quantities, key=lambda q: -q.span[0]):
        if cond.span[0] <= q.span[0] and q.span[1] <= cond.span[1]:
            s, e = q.span[0] - offset, q.span[1] - offset
            text = text[:s] + "<Q>" + text[e:]
    text = re.sub(r"\b(the|a|an|their|his|her|its)\b", " ", text.lower())
    return re.sub(r"[^\w<>%]+", " ", text).strip()


def _pair_conditions(f1: ProvisionFrame, f2: ProvisionFrame):
    m1 = [_mask_condition(c, f1.quantities) for c in f1.conditions]
    m2 = [_mask_condition(c, f2.quantities) for c in f2.conditions]
    pairs, used2 = [], set()
    for i, c in enumerate(f1.conditions):                      # identical after masking
        for j, d in enumerate(f2.conditions):
            if j not in used2 and m1[i] == m2[j]:
                pairs.append((i, j)); used2.add(j); break
    done1 = {i for i, _ in pairs}
    for i, c in enumerate(f1.conditions):                      # similar enough to be "the same" condition
        if i in done1:
            continue
        best, best_j = 0.0, None
        for j, d in enumerate(f2.conditions):
            if j in used2:
                continue
            score = fuzz.token_set_ratio(m1[i], m2[j]) + (15 if c.marker == d.marker else 0)
            if score > best:
                best, best_j = score, j
        if best_j is not None and best >= 60:
            pairs.append((i, best_j)); used2.add(best_j); done1.add(i)
    removed = [i for i in range(len(f1.conditions)) if i not in done1]
    added = [j for j in range(len(f2.conditions)) if j not in used2]
    return pairs, removed, added, m1, m2


def _inside(span, conds: list[Condition], idx: list[int]) -> bool:
    return any(conds[i].span[0] <= span[0] and span[1] <= conds[i].span[1] for i in idx)


# ================================================================== quantities
def _pair_quantities(q1: list[Quantity], q2: list[Quantity], own: Ownership, equivalent=equivalent):
    pairs, used2 = [], set()
    for i, a in enumerate(q1):                                  # equivalent values first
        for j, b in enumerate(q2):
            if j not in used2 and equivalent(a, b) and comparator_class(a.comparator) == comparator_class(b.comparator):
                pairs.append((i, j)); used2.add(j); break
    done1 = {i for i, _ in pairs}
    for i, a in enumerate(q1):                                  # then same dimension, same head preferred
        if i in done1:
            continue
        cands = [j for j, b in enumerate(q2)
                 if j not in used2 and dimension_family(b.dimension) == dimension_family(a.dimension)]
        if not cands:
            continue
        cands.sort(key=lambda j: (q2[j].head != a.head, abs(j - i)))
        j = cands[0]
        pairs.append((i, j)); used2.add(j); done1.add(i)
    return pairs, [i for i in range(len(q1)) if i not in done1], [j for j in range(len(q2)) if j not in used2]


def _direction_note(comp: Optional[str], old: float, new: float) -> Optional[str]:
    """A lower bound that rises, or an upper bound that falls, is harder to satisfy."""
    if comp in LOWER_BOUND:
        return "bound tightened" if new > old else "bound loosened"
    if comp in UPPER_BOUND:
        return "bound tightened" if new < old else "bound loosened"
    return None


def _quantity_changes(f1, f2, own: Ownership, new_id, notes, cond_removed, cond_added,
                      equivalent=equivalent) -> list[AtomicChange]:
    out = []
    pairs, rem, add = _pair_quantities(f1.quantities, f2.quantities, own, equivalent)
    for i, j in pairs:
        a, b = f1.quantities[i], f2.quantities[j]
        cat = C.TEMPORAL if (is_temporal(a) or is_temporal(b)) else C.NUMERIC
        wa, wb = _widen_left(own.doc1, a.span), _widen_left(own.doc2, b.span)   # include "at least", "within" ...
        if equivalent(a, b):
            if comparator_class(a.comparator) == comparator_class(b.comparator):
                if a.raw.lower() != b.raw.lower():
                    approx = " (approx.)" if a.approximate or b.approximate else ""
                    notes.append(f"'{a.raw}' ≡ '{b.raw}'{approx}")
                own.claim(a.span, b.span)
                continue
        elif a.dimension != b.dimension or (a.dimension == "currency" and a.unit != b.unit):
            out.append(AtomicChange(
                id=new_id(), category=cat, subtype="UNIT_CHANGED",
                description=f"Unit changed: {a.unit} → {b.unit}", old=a.raw, new=b.raw,
                old_span=a.span, new_span=b.span,
                evidence={"old_unit": a.unit, "new_unit": b.unit, "old_dimension": a.dimension,
                          "new_dimension": b.dimension}))
            own.claim(a.span, b.span)
            continue
        else:
            inc = b.si_value > a.si_value
            rel = abs(b.si_value - a.si_value) / abs(a.si_value) if a.si_value else float("inf")
            comp = b.comparator or a.comparator
            direction = _direction_note(comp, a.si_value, b.si_value)
            if cat == C.TEMPORAL:
                noun = "Time limit" if comp in UPPER_BOUND else "Duration"
                verb = ("extended" if inc else "shortened") if comp in UPPER_BOUND else ("increased" if inc else "decreased")
            elif a.dimension == "currency":
                noun, verb = "Amount", ("increased" if inc else "decreased")
            else:
                noun = "Threshold" if comp else "Value"
                verb = ("raised" if inc else "lowered") if comp else ("increased" if inc else "decreased")
            desc = f"{noun} {verb}" + (f" ({direction})" if direction else "")
            out.append(AtomicChange(
                id=new_id(), category=cat, subtype="INCREASED" if inc else "DECREASED",
                description=desc, old=describe(a), new=describe(b), old_span=wa, new_span=wb,
                evidence={"old_value": a.value, "new_value": b.value, "unit": b.unit,
                          "dimension": b.dimension, "relative_change": round(rel, 4),
                          "absolute_change": round(b.si_value - a.si_value, 6),
                          "comparator": comp, "direction": direction, "head": b.head}))
        if comparator_class(a.comparator) != comparator_class(b.comparator):
            out.append(AtomicChange(
                id=new_id(), category=cat, subtype="COMPARATOR_CHANGED",
                description=f"Comparator changed: {a.comparator or 'exact'} → {b.comparator or 'exact'}",
                old=describe(a), new=describe(b), old_span=wa, new_span=wb,
                evidence={"old_comparator": a.comparator, "new_comparator": b.comparator}))
        # comparator words sit just left of the quantity; claim them with it
        own.claim(wa, wb)

    for i in rem:
        a = f1.quantities[i]
        if _inside(a.span, f1.conditions, cond_removed):
            continue
        cat = C.TEMPORAL if is_temporal(a) else C.NUMERIC
        out.append(AtomicChange(id=new_id(), category=cat, subtype="REMOVED",
                                description=f"{'Time limit' if cat == C.TEMPORAL else 'Quantity'} removed",
                                old=describe(a), old_span=a.span, evidence={"dimension": a.dimension}))
        own.claim(_widen_left(own.doc1, a.span), None)
    for j in add:
        b = f2.quantities[j]
        if _inside(b.span, f2.conditions, cond_added):
            continue
        cat = C.TEMPORAL if is_temporal(b) else C.NUMERIC
        out.append(AtomicChange(id=new_id(), category=cat, subtype="ADDED",
                                description=f"{'Time limit' if cat == C.TEMPORAL else 'Quantity'} added",
                                new=describe(b), new_span=b.span, evidence={"dimension": b.dimension}))
        own.claim(None, _widen_left(own.doc2, b.span))
    return out


def _widen_left(doc, span, max_tokens: int = 3):
    """Extend a quantity span leftwards over comparator words ('at least', 'within')."""
    if not span:
        return span
    toks = sorted(char_span_to_tokens(doc, span))
    if not toks:
        return span
    first = toks[0]
    k = first
    while k > 0 and first - k < max_tokens and doc[k - 1].pos_ in ("ADP", "ADV", "ADJ", "PART", "SCONJ") \
            and doc[k - 1].lower_ in {"at", "least", "most", "within", "up", "to", "more", "less", "than",
                                       "not", "no", "over", "under", "above", "below", "exactly",
                                       "fewer", "greater", "minimum", "maximum"}:
        k -= 1
    return (doc[k].idx, span[1])


def _condition_changes(f1, f2, own, new_id, pairs, removed, added, m1, m2) -> list[AtomicChange]:
    out = []
    for i in removed:
        c = f1.conditions[i]
        out.append(AtomicChange(id=new_id(), category=C.CONDITION, subtype="REMOVED",
                                description=f"Condition removed ({c.marker})", old=c.text, old_span=c.span,
                                evidence={"marker": c.marker, "polarity": c.polarity}))
        own.claim(c.span, None)
    for j in added:
        c = f2.conditions[j]
        desc = "Exception added" if c.polarity == "negative" else "Condition added"
        out.append(AtomicChange(id=new_id(), category=C.CONDITION, subtype="ADDED",
                                description=f"{desc} ({c.marker})", new=c.text, new_span=c.span,
                                evidence={"marker": c.marker, "polarity": c.polarity}))
        own.claim(None, c.span)
    for i, j in pairs:
        a, b = f1.conditions[i], f2.conditions[j]
        if m1[i] == m2[j] and a.polarity == b.polarity:
            continue
        if not own.is_free(a.span, b.span):
            continue
        flip = a.polarity != b.polarity
        out.append(AtomicChange(
            id=new_id(), category=C.CONDITION, subtype="MODIFIED",
            description="Condition polarity flipped" if flip else "Condition modified",
            old=a.text, new=b.text, old_span=a.span, new_span=b.span,
            evidence={"old_marker": a.marker, "new_marker": b.marker, "polarity_flip": flip}))
        own.claim(a.span, b.span)
    return out


# ================================================================== scope
def _scope_changes(f1, f2, own, new_id) -> list[AtomicChange]:
    out = []
    ct1 = [c.text for c in f1.conditions]
    ct2 = [c.text for c in f2.conditions]
    for role in ("agent", "patient"):
        a: Optional[ScopeInfo] = getattr(f1, role)
        b: Optional[ScopeInfo] = getattr(f2, role)
        if a is None or b is None:
            continue
        who = "Subject" if role == "agent" else "Object"
        if a.head != b.head:
            if role == "agent" and own.is_free(a.span, b.span):
                out.append(AtomicChange(
                    id=new_id(), category=C.ENTITY, subtype="REPLACED",
                    description=f"{who} changed: {a.head} → {b.head}", old=a.text, new=b.text,
                    old_span=a.span, new_span=b.span, evidence={"role": role}))
                own.claim(a.span, b.span)
            continue
        verdict = compare_scope(a, b, ct1, ct2)
        if verdict is None:
            own.claim(a.span, b.span)          # e.g. "All students" ≡ "Students": explained, no change
            continue
        if not own.is_free(a.span, b.span):
            continue
        if verdict == "CHANGED" and (role == "patient" or a.has_proper_noun_modifier or b.has_proper_noun_modifier):
            # A replaced modifier naming someone ("constituted by the Dean" -> "... the Registrar")
            # is an entity change; an arbitrary object rewrite is left to the residual + NLI.
            continue
        ma, mb = modifier_lemma_set(a, ct1), modifier_lemma_set(b, ct2)
        out.append(AtomicChange(
            id=new_id(), category=C.SCOPE, subtype=verdict,
            description=f"{who} scope {verdict.lower()}", old=a.text, new=b.text,
            old_span=a.span, new_span=b.span,
            evidence={"role": role, "old_quantifier": a.quantifier, "new_quantifier": b.quantifier,
                      "added_modifiers": sorted(mb - ma), "removed_modifiers": sorted(ma - mb),
                      "added_exceptions": sorted(set(b.exceptions) - set(a.exceptions)),
                      "removed_exceptions": sorted(set(a.exceptions) - set(b.exceptions))}))
        own.claim(a.span, b.span)
    return out


# ================================================================== dates
def _date_changes(f1, f2, own, new_id, cond_removed, cond_added) -> list[AtomicChange]:
    out = []
    d1 = [d for d in f1.dates if own.is_free(d.span, None)]
    d2 = [d for d in f2.dates if own.is_free(None, d.span)]
    used1, used2 = set(), set()
    for ia, a in enumerate(d1):
        match = next((k for k, b in enumerate(d2) if k not in used2 and b.normalized == a.normalized), None)
        if match is not None:
            used1.add(ia)
            used2.add(match)
            own.claim(a.span, d2[match].span)
    unmatched1 = [a for ia, a in enumerate(d1) if ia not in used1]
    unmatched2 = [b for k, b in enumerate(d2) if k not in used2]
    for a, b in zip(unmatched1, unmatched2):
        out.append(AtomicChange(id=new_id(), category=C.TEMPORAL, subtype="CHANGED",
                                description="Date/time changed", old=a.raw, new=b.raw,
                                old_span=a.span, new_span=b.span,
                                evidence={"old_normalized": a.normalized, "new_normalized": b.normalized}))
        own.claim(a.span, b.span)
    for a in unmatched1[len(unmatched2):]:
        if not _inside(a.span, f1.conditions, cond_removed):
            out.append(AtomicChange(id=new_id(), category=C.TEMPORAL, subtype="REMOVED",
                                    description="Date/time removed", old=a.raw, old_span=a.span))
            own.claim(a.span, None)
    for b in unmatched2[len(unmatched1):]:
        if not _inside(b.span, f2.conditions, cond_added):
            out.append(AtomicChange(id=new_id(), category=C.TEMPORAL, subtype="ADDED",
                                    description="Date/time added", new=b.raw, new_span=b.span))
            own.claim(None, b.span)
    return out


# ================================================================== entities & antonyms
def _is_entityish(tok) -> bool:
    return tok.pos_ == "PROPN" or (tok.ent_type_ and tok.ent_type_ not in NUMERIC_LABELS)


def _entity_changes(own: Ownership, new_id) -> list[AtomicChange]:
    out = []
    d1, d2 = own.doc1, own.doc2
    for op in own.lex.ops:
        if op.op == "equal":
            continue
        t1 = [i for i in range(op.i1, op.i2) if i not in own.old and _is_entityish(d1[i])]
        t2 = [j for j in range(op.j1, op.j2) if j not in own.new and _is_entityish(d2[j])]
        if not t1 and not t2:
            continue
        s1 = _union(*(entity_span_for_token(d1, i) for i in t1)) if t1 else None
        s2 = _union(*(entity_span_for_token(d2, j) for j in t2)) if t2 else None
        if s1 and s2:
            sub, desc = "REPLACED", "Entity changed"
        elif s2:
            sub, desc = "ADDED", "Entity added"
        else:
            sub, desc = "REMOVED", "Entity removed"
        out.append(AtomicChange(id=new_id(), category=C.ENTITY, subtype=sub, description=desc,
                                old=_text(d1, s1), new=_text(d2, s2), old_span=s1, new_span=s2,
                                evidence={"source": "token-diff + NER/PROPN"}))
        own.claim(s1, s2)
        if sub == "REPLACED":                 # the whole replace op is explained by the entity swap
            own.old.update(range(op.i1, op.i2))
            own.new.update(range(op.j1, op.j2))
    return out


def _antonym_changes(own: Ownership, new_id) -> list[AtomicChange]:
    out = []
    d1, d2 = own.doc1, own.doc2
    for op in own.lex.ops:
        if op.op != "replace":
            continue
        for i in range(op.i1, op.i2):
            if i in own.old:
                continue
            l1 = d1[i].lemma_.lower()
            for j in range(op.j1, op.j2):
                if j in own.new:
                    continue
                l2 = d2[j].lemma_.lower()
                if l2 in ANTONYMS.get(l1, ()) or d2[j].lower_ in ANTONYMS.get(d1[i].lower_, ()):
                    s1 = (d1[i].idx, d1[i].idx + len(d1[i].text))
                    s2 = (d2[j].idx, d2[j].idx + len(d2[j].text))
                    out.append(AtomicChange(id=new_id(), category=C.REVERSAL, subtype="ANTONYM",
                                            description=f"Meaning reversed: '{d1[i].text}' → '{d2[j].text}'",
                                            old=d1[i].text, new=d2[j].text, old_span=s1, new_span=s2,
                                            evidence={"lemma_pair": [l1, l2]}))
                    own.old.add(i); own.new.add(j)
    return out


# ================================================================== entry point
def _surface_equal(a: Quantity, b: Quantity) -> bool:
    """Ablation: quantities are equal only if written identically (no unit normalisation)."""
    return a.raw.lower() == b.raw.lower()


def diff_frames(doc1, doc2, f1: ProvisionFrame, f2: ProvisionFrame, lex: LexicalDiff,
                ablations: frozenset = frozenset()) -> FrameDiff:
    own = Ownership(doc1, doc2, lex)
    counter = count(1)
    new_id = lambda: f"c{next(counter)}"
    notes: list[str] = []
    changes: list[AtomicChange] = []

    changes += _deontic_changes(f1, f2, own, new_id, notes)
    c_pairs, c_removed, c_added, m1, m2 = _pair_conditions(f1, f2)
    eq = _surface_equal if "no_quantity_normalization" in ablations else equivalent
    changes += _quantity_changes(f1, f2, own, new_id, notes, c_removed, c_added, eq)
    changes += _condition_changes(f1, f2, own, new_id, c_pairs, c_removed, c_added, m1, m2)
    changes += _date_changes(f1, f2, own, new_id, c_removed, c_added)
    changes += _scope_changes(f1, f2, own, new_id)
    changes += _entity_changes(own, new_id)
    changes += _antonym_changes(own, new_id)

    # Grounding rule: a change must be anchored in at least one token that actually differs.
    # Extractors run on independent parses, so a parse difference alone (e.g. a PP that
    # attaches to the noun in one version and to the verb in the other) is not a change.
    changes = [c for c in changes if any(own.changed_in(c.old_span, c.new_span))]

    content_old, content_new = lex.content_changed(doc1, doc2)
    res_old = sorted(content_old - own.old)
    res_new = sorted(content_new - own.new)
    total = len(content_old) + len(content_new)
    explained = 1.0 if total == 0 else 1.0 - (len(res_old) + len(res_new)) / total
    return FrameDiff(changes=changes, notes=notes,
                     residual_old=[doc1[i].text for i in res_old],
                     residual_new=[doc2[j].text for j in res_new],
                     explained_ratio=round(explained, 3))

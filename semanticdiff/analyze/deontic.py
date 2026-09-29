"""Deontic analysis: modality and negation as ONE dimension.

For each modal expression we find (a) its modal class (OBLIGATION / RECOMMENDATION /
PERMISSION / lexical PROHIBITION), (b) whether it is negated, via spaCy ``neg``
dependencies on the modal, its governing verb or a negative determiner on the subject,
and (c) the *action* it governs (the verb the rule is about). Class + negation are
combined into a deontic state: "may not" -> PROHIBITION, "need not" -> NO_OBLIGATION,
"should not" -> DISCOURAGED. Comparing states (not surface words) is what prevents
"may appear -> may not appear" from being double-counted as modality + negation, and
what makes "may not" and "must not" count as equivalent.
"""
from __future__ import annotations

from typing import Optional

from semanticdiff.lexicons import (AUX_NEGATED_STATE, MODAL_AUX, MULTIWORD_MODALS,
                                   NEGATIVE_DETERMINERS, POSITIVE_STATE)
from semanticdiff.models import DeonticInfo, DeonticState, ModalClass

_SKIPPABLE_DEPS = {"neg", "advmod"}
_PREDICATE_POS = {"VERB", "AUX", "ADJ"}

POLARITY = {
    DeonticState.OBLIGATION: 1, DeonticState.RECOMMENDATION: 1, DeonticState.PERMISSION: 1,
    DeonticState.ASSERTION: 1, DeonticState.PROHIBITION: -1, DeonticState.DISCOURAGED: -1,
    DeonticState.NEGATED_ASSERTION: -1, DeonticState.NO_OBLIGATION: 0,
}


def _match_multiword(doc, start: int, pattern: tuple[str, ...]) -> Optional[list[int]]:
    """Match a lemma sequence at ``start``, skipping negation/adverbs in between."""
    idx, k = [], 0
    i = start
    while i < len(doc) and k < len(pattern):
        tok = doc[i]
        if tok.lemma_.lower() == pattern[k] or tok.lower_ == pattern[k]:
            idx.append(i)
            k += 1
        elif idx and (tok.dep_ in _SKIPPABLE_DEPS or tok.lower_ == "not"):
            pass
        else:
            return None
        i += 1
    return idx if k == len(pattern) else None


def _action_of(anchor) -> object:
    """The verb a modal expression is about: xcomp / pcomp of the anchor if present."""
    for child in anchor.children:
        if child.dep_ == "xcomp" and child.pos_ in ("VERB", "AUX"):
            return child
    for child in anchor.children:                   # "prohibited from entering"
        if child.dep_ == "prep" and child.lower_ in ("from", "to", "for"):
            for g in child.children:
                if g.dep_ == "pcomp" and g.pos_ == "VERB":
                    return g
    return anchor


def _negation(anchor, action, extra=()) -> Optional[str]:
    for tok in (anchor, action, *extra):
        for child in tok.children:
            if child.dep_ == "neg" or child.lower_ in ("not", "n't", "never"):
                return child.text
    # negative determiner on the subject: "No student may ..."
    for tok in (anchor, action):
        for child in tok.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                for d in child.children:
                    if d.dep_ == "det" and d.lower_ in NEGATIVE_DETERMINERS:
                        return d.text
    return None


def _make(doc, anchor, action, modal_class: ModalClass, trigger_idx: list[int],
          negated_state: Optional[DeonticState]) -> DeonticInfo:
    extra = [doc[i] for i in trigger_idx]
    neg = _negation(anchor, action, extra)
    if neg:
        state = negated_state or AUX_NEGATED_STATE[modal_class]
    else:
        state = POSITIVE_STATE[modal_class]
    toks = sorted(set(trigger_idx) | ({t.i for t in anchor.children if t.dep_ == "neg"} if neg else set()))
    trig_start, trig_end = doc[toks[0]].idx, doc[toks[-1]].idx + len(doc[toks[-1]].text)
    return DeonticInfo(
        predicate=action.lemma_.lower(), predicate_span=(action.idx, action.idx + len(action.text)),
        modal_class=modal_class, negated=bool(neg), state=state,
        trigger=doc.text[trig_start:trig_end], trigger_span=(trig_start, trig_end), negation_cue=neg)


def extract_deontic(doc) -> list[DeonticInfo]:
    found: list[DeonticInfo] = []
    covered_actions: set[int] = set()
    used_tokens: set[int] = set()

    # 1. multi-word expressions ("are required to", "is prohibited from")
    for start in range(len(doc)):
        if start in used_tokens:
            continue
        for pattern, cls, neg_state in MULTIWORD_MODALS:
            idx = _match_multiword(doc, start, pattern)
            if not idx:
                continue
            # anchor = the content word of the expression (require / allow / prohibit / eligible)
            content = [doc[i] for i in idx if doc[i].lemma_.lower() not in ("be", "to", "from", "for", "the")]
            anchor = content[0] if content else doc[idx[0]]
            action = _action_of(anchor)
            if action.i in covered_actions:
                break
            found.append(_make(doc, anchor, action, cls, idx, neg_state))
            covered_actions.add(action.i)
            used_tokens.update(idx)
            break

    # 2. single-token modal auxiliaries ("must", "may", "shall", "cannot")
    for tok in doc:
        if tok.i in used_tokens or tok.lower_ not in MODAL_AUX or tok.dep_ not in ("aux", "auxpass", "ROOT"):
            continue
        anchor = tok.head if tok.dep_ != "ROOT" else tok
        # "shall be required to ..." — let the multiword expression win
        if anchor.i in covered_actions or _action_of(anchor).i in covered_actions:
            continue
        action = _action_of(anchor)
        found.append(_make(doc, anchor, action, MODAL_AUX[tok.lower_], [tok.i], None))
        covered_actions.add(action.i)

    # 3. no modal at all: a plain assertion about the ROOT predicate
    if not found:
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is not None and root.pos_ in _PREDICATE_POS | {"NOUN"}:
            neg = _negation(root, root)
            neg_tok = next((c for c in root.children if c.dep_ == "neg"), None)
            found.append(DeonticInfo(
                predicate=root.lemma_.lower(), predicate_span=(root.idx, root.idx + len(root.text)),
                modal_class=ModalClass.NONE, negated=bool(neg),
                state=DeonticState.NEGATED_ASSERTION if neg else DeonticState.ASSERTION,
                trigger=neg or "",
                trigger_span=(neg_tok.idx, neg_tok.idx + len(neg_tok.text)) if neg_tok is not None else None,
                negation_cue=neg))
    found.sort(key=lambda d: d.predicate_span[0])
    return found


def main_predicate(infos: list[DeonticInfo]) -> Optional[DeonticInfo]:
    return infos[0] if infos else None

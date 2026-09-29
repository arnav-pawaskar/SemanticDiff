"""Participants and scope: who a provision applies to.

Extracts the logical *agent* and *patient* noun phrases with voice normalisation:
  * active   "Applicants must submit applications"  -> agent=applicants, patient=applications
  * passive  "Applications shall be submitted"      -> agent=None,       patient=applications
  * control  "Applicants are required to submit ..." -> the passive subject of a modal
             predicate (required/allowed/prohibited ...) is the agent of the action.
For each noun phrase we record the head lemma, quantifier, modifiers and exceptions.
Scope comparison (same head, different quantifier/modifiers) is done in changes/diff.py.
"""
from __future__ import annotations

import re
from typing import Optional

from semanticdiff.lexicons import EXCEPTION_MARKERS, QUANTIFIERS
from semanticdiff.models import ScopeInfo

_MODAL_CONTENT = {"require", "allow", "permit", "entitle", "oblige", "obligate", "prohibit",
                  "forbid", "expect", "encourage", "advise", "authorize", "authorise", "bar", "ban",
                  "recommend"}
_MODIFIER_DEPS = {"amod", "compound", "nmod", "prep", "relcl", "acl", "npadvmod", "nummod", "appos"}
_IGNORED_DETS = {"the", "a", "an", "this", "that", "these", "those", "their", "its", "his", "her", "our",
                 "your", "such"}
_EXCEPTION_RE = re.compile(r"\b(" + "|".join(re.escape(m) for m in EXCEPTION_MARKERS) + r")\b", re.IGNORECASE)


def _span_text(tokens) -> str:
    toks = sorted(tokens, key=lambda t: t.i)
    if not toks:
        return ""
    doc = toks[0].doc
    return doc.text[toks[0].idx: toks[-1].idx + len(toks[-1].text)]


def _content_lemmas(tokens) -> str:
    """Order-free content signature: 'for re-evaluation' and 're-evaluation' -> 'evaluation re'."""
    lem = {t.lemma_.lower() for t in tokens
           if not (t.is_punct or t.is_stop or t.pos_ in ("ADP", "DET", "PRON", "CCONJ", "PART"))}
    return " ".join(sorted(lem))


def noun_phrase(head, role: str) -> ScopeInfo:
    quantifier: Optional[str] = None
    modifiers: list[str] = []
    terms: list[str] = []
    proper = False
    exceptions: list[str] = []
    for child in head.children:
        low = child.lower_
        if low in QUANTIFIERS and child.dep_ in ("det", "predet", "advmod", "amod", "quantmod", "nummod"):
            quantifier = low
            continue
        if child.dep_ in ("det", "poss", "punct", "cc") or low in _IGNORED_DETS:
            continue
        if child.dep_ == "prep" and _EXCEPTION_RE.fullmatch(low):
            exceptions.append(_span_text(child.subtree).lower())
            continue
        if child.dep_ in _MODIFIER_DEPS:
            sub = list(child.subtree)
            modifiers.append(_span_text(sub).lower())
            terms.append(_content_lemmas(sub))
            proper = proper or any(t.pos_ == "PROPN" or (t.ent_type_ and t.ent_type_ not in
                                   ("DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"))
                                   for t in sub)
    # "Only X" is often attached above the noun phrase
    if quantifier is None and head.left_edge.i > 0 and head.doc[head.left_edge.i - 1].lower_ == "only":
        quantifier = "only"
    subtree = list(head.subtree)
    start, end = subtree[0].idx, subtree[-1].idx + len(subtree[-1].text)
    return ScopeInfo(text=head.doc.text[start:end], head=head.lemma_.lower(), span=(start, end),
                     role=role, quantifier=quantifier, modifiers=modifiers, modifier_terms=terms,
                     has_proper_noun_modifier=proper, exceptions=sorted(set(exceptions)))


def _find_child(tok, deps: set[str]):
    return next((c for c in tok.children if c.dep_ in deps), None)


def extract_roles(doc) -> tuple[Optional[ScopeInfo], Optional[ScopeInfo], str]:
    """Return (agent, patient, voice) for the main clause."""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return None, None, "active"
    subj = _find_child(root, {"nsubj", "nsubjpass", "csubj"})
    action = root
    for c in root.children:
        if c.dep_ == "xcomp" and c.pos_ in ("VERB", "AUX"):
            action = c
            break
        if c.dep_ == "prep" and c.lower_ in ("from", "to"):
            pc = _find_child(c, {"pcomp"})
            if pc is not None and root.lemma_.lower() in _MODAL_CONTENT:
                action = pc
                break

    agent = patient = None
    voice = "active"
    action_passive = action is not root and any(c.dep_ == "auxpass" for c in action.children)
    if subj is not None and subj.dep_ == "nsubjpass":
        if root.lemma_.lower() in _MODAL_CONTENT and action is not root and action_passive:
            # "Work is required to be approved by the Head": the subject is the patient
            voice = "passive"
            patient = noun_phrase(subj, "patient")
            by = next((c for c in action.children if c.dep_ == "agent"), None)
            pobj = _find_child(by, {"pobj"}) if by is not None else None
            if pobj is not None:
                agent = noun_phrase(pobj, "agent")
        elif root.lemma_.lower() in _MODAL_CONTENT and action is not root:
            agent = noun_phrase(subj, "agent")            # control: "Applicants are required to submit"
        else:
            voice = "passive"
            patient = noun_phrase(subj, "patient")
            by = next((c for c in root.children if c.dep_ == "agent"), None)
            if by is not None:
                pobj = _find_child(by, {"pobj"})
                if pobj is not None:
                    agent = noun_phrase(pobj, "agent")
    elif subj is not None and subj.pos_ in ("NOUN", "PROPN", "PRON", "ADJ"):
        agent = noun_phrase(subj, "agent")

    if patient is None and not action_passive:
        obj = _find_child(action, {"dobj", "attr"}) or _find_child(root, {"dobj"})
        if obj is not None and obj.pos_ in ("NOUN", "PROPN"):
            patient = noun_phrase(obj, "patient")
    return agent, patient, voice


# ------------------------------------------------------------------ comparison helpers
def quantifier_class(q: Optional[str]) -> str:
    if q in (None, "all", "every", "each", "any", "everyone", "everybody", "no", "none"):
        return "universal"            # bare plurals are generic; "no" is handled as negation
    return "restricted"


def modifier_lemma_set(info: ScopeInfo, drop_texts: list[str] = ()) -> set[str]:
    """Flattened content lemmas of all modifiers, skipping modifiers that are really
    conditions (their text lies inside a detected condition) and masking digits
    (numbers are compared by the quantity analyser, not as scope)."""
    out: set[str] = set()
    for text, terms in zip(info.modifiers, info.modifier_terms):
        if any(text in d.lower() or d.lower() in text for d in drop_texts):
            continue
        out.update(re.sub(r"\d[\d,.]*", "#", w) for w in terms.split())
    out.discard("#")
    return out


def compare_scope(a: ScopeInfo, b: ScopeInfo, drop_a: list[str] = (), drop_b: list[str] = ()) -> Optional[str]:
    """NARROWED / BROADENED / CHANGED / None for two noun phrases with the same head."""
    qa, qb = quantifier_class(a.quantifier), quantifier_class(b.quantifier)
    ma, mb = modifier_lemma_set(a, drop_a), modifier_lemma_set(b, drop_b)
    ea, eb = set(a.exceptions), set(b.exceptions)
    if qa == qb and ma == mb and ea == eb:
        return None
    narrower = (mb >= ma and eb >= ea and (qa == qb or (qa == "universal" and qb == "restricted")))
    broader = (ma >= mb and ea >= eb and (qa == qb or (qa == "restricted" and qb == "universal")))
    if narrower and not broader:
        return "NARROWED"
    if broader and not narrower:
        return "BROADENED"
    return "CHANGED"

"""Controlled perturbation generators for the SemanticDiff benchmark.

Each generator takes a parsed seed provision and returns a perturbed sentence (or None
if the seed offers no suitable site). Generators are split-aware: the dev and test
splits draw on *different* lexical variants (adjectives, condition templates, entity
swaps, paraphrase transforms) so that thresholds tuned on dev are tested on unseen
surface forms. Gold labels follow data/eval/LABELING.md.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Optional

from semanticdiff.analyze.conditions import extract_conditions
from semanticdiff.analyze.quantities import extract_quantities

# --------------------------------------------------------------------------- helpers
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
          9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}


def _replace(text: str, start: int, end: int, new: str) -> str:
    return text[:start] + new + text[end:]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _decap(doc) -> str:
    """Lower-case the first letter unless the sentence starts with a proper noun or acronym."""
    first = doc[0]
    if first.pos_ == "PROPN" or first.text.isupper():
        return doc.text
    return doc.text[:1].lower() + doc.text[1:]


def _subject(doc):
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return None, None
    subj = next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None)
    return root, subj


def _plural(tok) -> bool:
    return tok is not None and (tok.tag_ in ("NNS", "NNPS") or tok.lower_.endswith("s") and tok.tag_ != "NN")


def _fmt_number(v: float, like: str) -> str:
    if float(v).is_integer():
        v = int(v)
        return f"{v:,}" if "," in like else str(v)
    return f"{v:.1f}" if "." in like else f"{v:.1f}"


# --------------------------------------------------------------------------- paraphrase
_UNIT_REWRITES = {
    "dev": [("7 days", "one week"), ("24 hours", "one day"), ("60 days", "two months"),
            ("4 hours", "240 minutes"), ("12 months", "one year"), ("2 years", "24 months"),
            ("8 hours", "480 minutes"), ("90 days", "ninety days"), ("200 milliseconds", "0.2 seconds"),
            ("48 hours", "two days"), ("26 weeks", "182 days"), ("3 months", "three months")],
    "test": [("14 days", "two weeks"), ("72 hours", "three days"), ("6 months", "half a year"),
             ("8 weeks", "56 days"), ("7 years", "seven years"), ("30 days", "thirty days"),
             ("10 days", "ten days"), ("15 days", "fifteen days"), ("21 days", "three weeks"),
             ("45 days", "forty-five days"), ("2 days", "48 hours"), ("4 hours", "four hours")],
}
_PREFIX = {"dev": ["Under this policy, ", "As a rule, "], "test": ["As per these rules, ", "According to this policy, "]}


def _t_modal_synonym(doc, rng, split) -> Optional[str]:
    root, subj = _subject(doc)
    for t in doc:
        if t.dep_ not in ("aux", "auxpass") or t.i + 1 >= len(doc) or doc[t.i + 1].lower_ == "not":
            continue
        low = t.lower_
        if split == "dev":
            swap = {"must": "shall", "shall": "must", "may": "can"}.get(low)
        else:
            be = "are" if _plural(subj) else "is"
            swap = {"must": f"{be} required to", "shall": f"{be} required to",
                    "may": f"{be} permitted to"}.get(low)
        if swap:
            return _replace(doc.text, t.idx, t.idx + len(t.text), swap)
    return None


def _t_units(doc, rng, split) -> Optional[str]:
    text = doc.text
    for old, new in _UNIT_REWRITES[split]:
        m = re.search(rf"\b{re.escape(old)}\b", text)
        if m:
            return _replace(text, m.start(), m.end(), new)
    if split == "dev":           # small integers -> words
        m = re.search(r"\b([1-9]|1[0-9]|20) (days|hours|weeks|months|years|books|credits)\b", text)
        if m:
            return _replace(text, m.start(1), m.end(1), _WORDS[int(m.group(1))])
    return None


def _t_prefix(doc, rng, split) -> Optional[str]:
    if doc.text.startswith(("Under", "As ", "According")):
        return None
    return rng.choice(_PREFIX[split]) + _decap(doc)


def _t_drop_all(doc, rng, split) -> Optional[str]:
    m = re.match(r"^All (\w)", doc.text)
    if m:
        return m.group(1).upper() + doc.text[len("All ") + 1:]
    return None


PARAPHRASE_TRANSFORMS = [_t_modal_synonym, _t_units, _t_prefix, _t_drop_all]


def paraphrase(doc, rng, split, nlp) -> Optional[str]:
    ts = PARAPHRASE_TRANSFORMS[:]
    rng.shuffle(ts)
    text, applied = doc.text, 0
    for t in ts:
        out = t(nlp(text), rng, split)
        if out and out != text:
            text, applied = out, applied + 1
            if applied >= rng.choice([1, 2]):
                break
    return text if applied else None


# --------------------------------------------------------------------------- numeric / temporal
_FACTORS = [0.5, 0.6, 0.8, 1.2, 1.25, 1.5, 2.0]


def _change_quantity(doc, rng, temporal: bool) -> Optional[str]:
    qs = [q for q in extract_quantities(doc)
          if (q.dimension in ("time", "business_time")) == temporal and q.dimension != "age"]
    qs = [q for q in qs if re.search(r"\d", q.raw)]
    if not qs:
        return None
    q = rng.choice(qs)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", q.raw)
    old = float(m.group(0).replace(",", ""))
    for _ in range(10):
        new = old * rng.choice(_FACTORS)
        if q.dimension == "percent" and old > 90:
            new = round(old - rng.choice([0.4, 0.5, 1.0, 2.0]), 1)
        elif old >= 1000:
            new = round(new / 100) * 100
        elif "." in m.group(0):
            new = round(new, 1)
        else:
            new = max(1, round(new))
        if new != old:
            break
    else:
        return None
    new_raw = q.raw[:m.start()] + _fmt_number(new, m.group(0)) + q.raw[m.end():]
    if new == 1:
        new_raw = re.sub(r"(day|hour|week|month|year|minute)s", r"", new_raw)
    return _replace(doc.text, q.span[0], q.span[1], new_raw)


def numeric(doc, rng, split, nlp) -> Optional[str]:
    return _change_quantity(doc, rng, temporal=False)


_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]


def temporal(doc, rng, split, nlp) -> Optional[str]:
    out = _change_quantity(doc, rng, temporal=True)
    if out:
        return out
    m = re.search(r"\b(" + "|".join(_MONTHS) + r")\b", doc.text)
    if m:
        new = rng.choice([x for x in _MONTHS if x != m.group(1)])
        return _replace(doc.text, m.start(), m.end(), new)
    m = re.search(r"\b(\d{1,2}) (AM|PM)\b", doc.text)
    if m:
        h = int(m.group(1))
        new = h + rng.choice([-2, -1, 1, 2])
        if 1 <= new <= 12:
            return _replace(doc.text, m.start(1), m.end(1), str(new))
    return None


# --------------------------------------------------------------------------- entity
_ENTITY_SWAPS = {
    "dev": {"Registrar": "Controller of Examinations", "Dean of Academics": "Vice-Chancellor",
            "HR Manager": "Finance Manager", "Finance Department": "Legal Department",
            "Procurement Committee": "Board of Directors", "Department Head": "Project Director",
            "Data Protection Officer": "Chief Technology Officer", "Mumbai": "Pune",
            "Bengaluru": "Chennai", "Hostel Warden": "Chief Proctor", "Examination Committee": "Academic Council",
            "Chief Medical Officer": "Hospital Administrator", "IT Department": "Purchase Department",
            "Accounts Office": "Student Welfare Office", "Board of Governors": "Vice-Chancellor",
            "Legal Department": "Compliance Team", "Placement Cell": "Alumni Office",
            "Dean of Students": "Head of Department"},
    "test": {"Registrar": "Dean of Students", "Dean of Academics": "Academic Council",
             "HR Manager": "Chief Executive Officer", "Finance Department": "Audit Committee",
             "Procurement Committee": "Managing Director", "Department Head": "Plant Manager",
             "Data Protection Officer": "Legal Counsel", "Mumbai": "Delhi",
             "Bengaluru": "Hyderabad", "Hostel Warden": "Security Officer", "Examination Committee": "Controller of Examinations",
             "Chief Medical Officer": "Senior Surgeon", "IT Department": "Finance Department",
             "Accounts Office": "Registrar", "Board of Governors": "Academic Council",
             "Legal Department": "Managing Director", "Placement Cell": "Training Department",
             "Dean of Students": "Registrar"},
}


def entity(doc, rng, split, nlp) -> Optional[str]:
    for old, new in _ENTITY_SWAPS[split].items():
        m = re.search(rf"\b{re.escape(old)}\b", doc.text)
        if m:
            return _replace(doc.text, m.start(), m.end(), new)
    return None


# --------------------------------------------------------------------------- modality
_MODAL_SWAPS = {"must": ["should", "may"], "shall": ["may", "should"], "may": ["must", "should"],
                "should": ["must", "may"]}


def modality(doc, rng, split, nlp) -> Optional[str]:
    for t in doc:
        if t.lower_ in _MODAL_SWAPS and t.dep_ in ("aux", "auxpass") and \
                not (t.i + 1 < len(doc) and doc[t.i + 1].lower_ == "not"):
            new = rng.choice(_MODAL_SWAPS[t.lower_])
            return _replace(doc.text, t.idx, t.idx + len(t.text), _cap(new) if t.i == 0 else new)
    m = re.search(r"\bare (allowed|permitted|entitled|eligible) (to|for)\b", doc.text)
    if m:
        return _replace(doc.text, m.start(), m.end(), f"are required {'to' if m.group(2) == 'to' else 'to apply for'}")
    root, subj = _subject(doc)
    if root is not None and not any(c.dep_ in ("aux", "auxpass", "neg") for c in root.children):
        if root.tag_ == "VBP":                                  # "Employees receive" -> "Employees may receive"
            return _replace(doc.text, root.idx, root.idx, rng.choice(["may ", "should "]))
        if root.tag_ == "VBZ" and root.lemma_ == "be":        # "The fee is" -> "The fee may be"
            return _replace(doc.text, root.idx, root.idx + len(root.text), rng.choice(["may be", "should be"]))
    return None


# --------------------------------------------------------------------------- negation
def negation(doc, rng, split, nlp) -> Optional[str]:
    for t in doc:                                              # remove an existing negation
        if t.dep_ == "neg" and t.lower_ == "not":
            end = t.idx + len(t.text)
            return _replace(doc.text, t.idx - 1 if doc.text[t.idx - 1] == " " else t.idx, end, "")
    for t in doc:
        if t.lower_ in ("must", "shall", "may", "should", "will") and t.dep_ in ("aux", "auxpass"):
            return _replace(doc.text, t.idx + len(t.text), t.idx + len(t.text), " not")
        if t.lower_ == "can" and t.dep_ == "aux":
            return _replace(doc.text, t.idx, t.idx + 3, "cannot")
    root, subj = _subject(doc)
    if root is None:
        return None
    aux = next((c for c in root.children if c.dep_ in ("auxpass", "aux") and c.lemma_ == "be"), None)
    if aux is not None:
        return _replace(doc.text, aux.idx + len(aux.text), aux.idx + len(aux.text), " not")
    if root.lemma_ == "be":
        return _replace(doc.text, root.idx + len(root.text), root.idx + len(root.text), " not")
    if root.tag_ == "VBP":
        return _replace(doc.text, root.idx, root.idx, "do not ")
    if root.tag_ == "VBZ":
        return _replace(doc.text, root.idx, root.idx + len(root.text), f"does not {root.lemma_}")
    return None


# --------------------------------------------------------------------------- scope
_SCOPE_ADJ = {"dev": ["final-year", "international", "full-time", "senior"],
              "test": ["first-year", "part-time", "visiting", "probationary"]}
# scope perturbations only make sense on person-denoting subjects
_PERSON_NOUNS = {"student", "employee", "patient", "visitor", "customer", "user", "vendor", "guest",
                 "applicant", "candidate", "contractor", "nurse", "resident", "tenant", "borrower",
                 "scholar", "member", "holder", "staff", "researcher"}
_BROADEN = re.compile(r"^(Final-year|Permanent|Research|International|Undergraduate) (\w+)")


def scope(doc, rng, split, nlp) -> Optional[str]:
    m = _BROADEN.match(doc.text)
    if m and rng.random() < 0.5:
        return "All " + m.group(2) + doc.text[m.end():]
    root, subj = _subject(doc)
    if subj is None or subj.pos_ != "NOUN" or not _plural(subj) or subj.lemma_.lower() not in _PERSON_NOUNS:
        return None
    adj = rng.choice(_SCOPE_ADJ[split])
    first = subj.left_edge
    if first.lower_ == "all":
        # "All students must ..." -> "Final-year students must ..."
        return _cap(adj) + doc.text[first.idx + len(first.text):]
    if first.i == subj.i:
        if subj.i == 0:
            return _cap(adj) + " " + _decap(doc)
        return _replace(doc.text, subj.idx, subj.idx, adj + " ")
    return None


# --------------------------------------------------------------------------- condition
_AUTH = ["Registrar", "Head of Department", "Finance Department", "Dean", "competent authority"]
_COND_TEMPLATES = {
    "dev": ["if approved by the {a}", "with prior approval from the {a}", "unless exempted by the {a}",
            "subject to verification by the {a}"],
    "test": ["provided that the {a} gives written consent", "only if the {a} agrees",
             "upon receipt of written approval from the {a}", "except when the {a} decides otherwise"],
}


def condition(doc, rng, split, nlp) -> Optional[str]:
    conds = extract_conditions(doc)
    if conds and rng.random() < 0.5:
        c = rng.choice(conds)
        s, e = c.span
        while s > 0 and doc.text[s - 1] in " ,":
            s -= 1
        return _replace(doc.text, s, e, "")
    text = doc.text.rstrip()
    end = len(text) - 1 if text.endswith(".") else len(text)
    clause = rng.choice(_COND_TEMPLATES[split]).format(a=rng.choice(_AUTH))
    return text[:end] + " " + clause + text[end:]


# --------------------------------------------------------------------------- reversal
_ANTONYM_SUBS = [
    ("non-refundable", "refundable"), ("refundable", "non-refundable"), ("are allowed to", "are forbidden to"),
    ("are prohibited", "are allowed"), ("is restricted to", "is open to"), ("increase", "decrease"),
    ("valid", "invalid"), ("free of charge", "for a fee"), ("destroyed", "preserved"),
    ("available to", "unavailable to"), ("reject", "accept"), ("encrypted", "unencrypted"),
    ("is available", "is unavailable"), ("are accepted", "are rejected"), ("are approved", "are rejected"), ("is prohibited", "is allowed"),
    ("are permitted", "are forbidden"), ("open to", "closed to"), ("are included", "are excluded"),
]


def reversal(doc, rng, split, nlp) -> Optional[str]:
    for old, new in _ANTONYM_SUBS:
        m = re.search(rf"\b{re.escape(old)}\b", doc.text)
        if m:
            return _replace(doc.text, m.start(), m.end(), new)
    return None


GENERATORS: dict[str, Callable] = {
    "NUMERIC": numeric, "TEMPORAL": temporal, "ENTITY": entity, "MODALITY": modality,
    "NEGATION": negation, "SCOPE": scope, "CONDITION": condition, "REVERSAL": reversal,
}
# combinations excluded from compound pairs because their gold label would be ambiguous
_INCOMPATIBLE = {frozenset({"MODALITY", "NEGATION"}), frozenset({"REVERSAL", "NEGATION"}),
                 frozenset({"REVERSAL", "MODALITY"}), frozenset({"NUMERIC", "TEMPORAL"})}


def compound(doc, rng, split, nlp, k: int = 2) -> Optional[tuple[str, list[str]]]:
    cats = [c for c in GENERATORS if c != "CONDITION"]
    rng.shuffle(cats)
    cats.append("CONDITION")        # add conditions last, so later edits never land inside them
    text, used = doc.text, []
    for cat in cats:
        if any(frozenset({cat, u}) in _INCOMPATIBLE for u in used):
            continue
        out = GENERATORS[cat](nlp(text), rng, split, nlp)
        if out and out != text:
            text, used = out, used + [cat]
            if len(used) == k:
                return text, sorted(used)
    return None

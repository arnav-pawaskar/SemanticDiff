"""Named-entity mentions (non-numeric spaCy labels).

spaCy's NER is weak on policy text (roles like "Registrar" are often missed), so entity
*changes* are also derived from token-diff replacements involving proper nouns — see
changes/diff.py. This module only lists what NER finds.
"""
from __future__ import annotations

from semanticdiff.models import EntityMention

NUMERIC_LABELS = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}


def extract_entities(doc) -> list[EntityMention]:
    return [EntityMention(text=e.text, label=e.label_, span=(e.start_char, e.end_char))
            for e in doc.ents if e.label_ not in NUMERIC_LABELS]


def entity_span_for_token(doc, i: int) -> tuple[int, int] | None:
    """Character span of the entity or proper-noun run containing token i."""
    tok = doc[i]
    if tok.ent_type_ and tok.ent_type_ not in NUMERIC_LABELS:
        for e in doc.ents:
            if e.start <= i < e.end:
                return e.start_char, e.end_char
    if tok.pos_ == "PROPN":
        s = e = i
        while s > 0 and doc[s - 1].pos_ == "PROPN":
            s -= 1
        while e + 1 < len(doc) and doc[e + 1].pos_ == "PROPN":
            e += 1
        return doc[s].idx, doc[e].idx + len(doc[e].text)
    return None

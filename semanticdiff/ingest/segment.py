"""Split structural blocks into provisions (the unit of alignment).

A provision is a sentence, a list item, or a top-level ';'-separated segment that
contains its own finite verb. Clause structure *inside* a provision is left to the
dependency-based analysers; we never cut sentences into fragments here.
"""
from __future__ import annotations

import re
from typing import Iterable

from semanticdiff.ingest.extract import RawLine, extract
from semanticdiff.ingest.structure import Block, build_blocks
from semanticdiff.models import Document, Provision, ProvisionKind

# A sentence ending with one of these is almost certainly a false split.
_ABBREVIATIONS = ("rs.", "no.", "nos.", "sec.", "e.g.", "i.e.", "etc.", "dr.", "mr.", "mrs.",
                  "ms.", "prof.", "vs.", "approx.", "art.", "cl.", "min.", "max.", "st.",
                  "govt.", "dept.", "inc.", "ltd.", "co.", "u.s.", "fig.", "p.", "pp.")
_OPENING_QUOTE_ONLY = re.compile(r"^[\"')\]]+$")


def _needs_merge(prev: str, nxt: str) -> bool:
    p = prev.rstrip().lower()
    if p.endswith(_ABBREVIATIONS):
        return True
    if re.search(r"\b[a-z]\.$", p):          # single-letter initials "J."
        return True
    n = nxt.lstrip()
    return bool(n) and (n[0].islower() or n[0].isdigit() or _OPENING_QUOTE_ONLY.match(n))


def split_sentences(text: str, nlp) -> list[str]:
    doc = nlp(text)
    sents: list[str] = []
    for s in doc.sents:
        t = s.text.strip()
        if not t:
            continue
        if sents and _needs_merge(sents[-1], t):
            sents[-1] = f"{sents[-1]} {t}"
        else:
            sents.append(t)
    return sents


def split_semicolons(sentence: str, nlp) -> list[str]:
    """Split on top-level ';' only if every part has its own finite verb."""
    if ";" not in sentence:
        return [sentence]
    parts = [p.strip() for p in sentence.split(";") if p.strip()]
    if len(parts) < 2:
        return [sentence]
    for part in parts:
        d = nlp(part)
        if not any(t.pos_ in ("VERB", "AUX") for t in d):
            return [sentence]
    fixed = []
    for i, part in enumerate(parts):
        part = re.sub(r"^(and|or)\s+", "", part, flags=re.IGNORECASE)
        part = part[0].upper() + part[1:]
        if not part.endswith((".", "?", "!")):
            part += "."
        fixed.append(part)
    return fixed


def normalize_for_match(text: str) -> str:
    """Canonical form used for exact-match detection (case, whitespace, quotes, trailing dot)."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    return t.rstrip(" .;")


def segment_blocks(blocks: Iterable[Block], version: str, nlp,
                   split_semi: bool = True, min_chars: int = 3) -> list[Provision]:
    provisions: list[Provision] = []
    for block in blocks:
        kind = ProvisionKind.LIST_ITEM if block.is_list_item else ProvisionKind.SENTENCE
        pieces = split_sentences(block.text, nlp)
        if split_semi:
            pieces = [p for s in pieces for p in split_semicolons(s, nlp)]
        for piece in pieces:
            if len(piece) < min_chars:
                continue
            idx = len(provisions)
            provisions.append(Provision(
                id=f"{version}-{idx:03d}", version=version, index=idx, text=piece, kind=kind,
                section_path=list(block.section_path), section_title=block.section_title,
                page=block.page))
    # Parse every provision on its own so that analysis sees exactly the text being compared.
    for prov, doc in zip(provisions, nlp.pipe([p.text for p in provisions], batch_size=64)):
        prov.doc = doc
    return provisions


def load_document(source, version: str, nlp, cfg: dict, name: str | None = None,
                  filename: str | None = None) -> Document:
    ing = cfg.get("ingest", {})
    lines: list[RawLine] = extract(source, filename=filename,
                                   header_footer_ratio=ing.get("header_footer_min_page_ratio", 0.5))
    blocks = build_blocks(lines)
    provisions = segment_blocks(blocks, version, nlp,
                                split_semi=ing.get("split_semicolons", True),
                                min_chars=ing.get("min_provision_chars", 3))
    raw = "\n".join(l.text for l in lines)
    return Document(name=name or filename or version, version=version, raw_text=raw,
                    provisions=provisions)

"""The decision table: fuse deterministic changes with NLI into a pair verdict.

    M = atomic changes found by the deterministic extractors
    E = bidirectional entailment >= tau_e        C = max contradiction >= tau_c
    S = embedding cosine                          R = unexplained content-word edits

    rule  M    NLI / other                      verdict              semantic level
    V0    -    texts identical                  IDENTICAL            LOW
    V1    ∅    E                                MEANING_PRESERVING   LOW
    V2    ∅    C                                MATERIAL (REVERSAL, NLI-only, flagged unexplained)
    V3    ∅    R == 0                           MEANING_PRESERVING   LOW   (only form/function words changed)
    V4    ∅    not E, not C, S >= tau_s         MEANING_PRESERVING   LOW   (low confidence)
    V5    ∅    otherwise                        UNCLASSIFIED (UNEXPLAINED change)   MEDIUM
    V6    ≠∅   —                                MATERIAL             max severity of M
          (+ "extractors and NLI disagree" note when E also holds)
"""
from __future__ import annotations

from typing import Optional

from semanticdiff.models import (AtomicChange, ChangeCategory as C, NLIResult, Severity, Verdict)


def decide(changes: list[AtomicChange], nli: Optional[NLIResult], sim_embed: Optional[float],
           residual_count: int, cfg: dict, identical: bool = False,
           residual_desc: str = "") -> tuple[Verdict, str, list[AtomicChange], list[str], float]:
    """Return (verdict, rule_id, extra_changes, notes, confidence)."""
    v = cfg["verdict"]
    notes: list[str] = []
    if identical:
        return Verdict.IDENTICAL, "V0:identical", [], notes, 1.0

    ent = nli.bidirectional_entailment if nli else 0.0
    con = nli.max_contradiction if nli else 0.0
    E, Cn = ent >= v["entail"], con >= v["contradiction"]

    if changes:
        if E:
            notes.append(f"Extractors and NLI disagree: NLI finds the versions mutually entailing "
                         f"(p={ent:.2f}) — review the detected changes.")
        return Verdict.MATERIAL, "V6:extractor-changes", [], notes, 1.0

    if E:
        return Verdict.MEANING_PRESERVING, "V1:no-extractor-change+bidirectional-entailment", [], notes, ent
    if Cn:
        extra = AtomicChange(
            id="nli1", category=C.REVERSAL, subtype="NLI_CONTRADICTION",
            description="Meaning reversed (detected by NLI only; no deterministic explanation)",
            confidence=con, evidence={"contradiction": con, "unexplained_tokens": residual_desc})
        return Verdict.MATERIAL, "V2:nli-contradiction-unexplained", [extra], notes, con
    if residual_count < v.get("unexplained_min_tokens", 1):
        return Verdict.MEANING_PRESERVING, "V3:no-content-word-change", [], notes, 0.9
    if sim_embed is not None and sim_embed >= v["preserve_similarity"]:
        notes.append("Low-confidence: NLI neutral, but embeddings very similar and no rule fired.")
        return Verdict.MEANING_PRESERVING, "V4:high-similarity-no-rule", [], notes, 0.6
    extra = AtomicChange(
        id="u1", category=C.UNEXPLAINED, subtype="LEXICAL",
        description="Wording changed in a way no rule explains and NLI does not confirm as equivalent",
        confidence=0.5, evidence={"unexplained_tokens": residual_desc, "entailment": ent})
    return Verdict.UNCLASSIFIED, "V5:unexplained-change", [extra], notes, 0.5


def semantic_level(verdict: Verdict, changes: list[AtomicChange]) -> Severity:
    if verdict in (Verdict.IDENTICAL, Verdict.MEANING_PRESERVING) or not changes:
        return Severity.LOW
    return max((c.severity for c in changes), key=lambda s: s.rank)

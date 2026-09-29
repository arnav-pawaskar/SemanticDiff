"""Configurable severity heuristic (config/severity.yaml).

Severity is SemanticDiff's *policy*, not ground truth. Every change records the id of
the rule that set its level so the UI can always answer "why is this HIGH?".
"""
from __future__ import annotations

from semanticdiff.models import AtomicChange, ChangeCategory as C, Severity

_UP = {Severity.LOW: Severity.MEDIUM, Severity.MEDIUM: Severity.HIGH, Severity.HIGH: Severity.HIGH}


def assign_severity(change: AtomicChange, cfg: dict, in_definition: bool = False,
                    normative: bool = False) -> None:
    sev = cfg["severity"]
    cat = change.category.value
    level = Severity(sev["category_defaults"].get(cat, "MEDIUM"))
    rule = f"category:{cat}"

    overrides = sev.get("subtype_overrides", {}).get(cat, {}) or {}
    for key in (change.subtype, change.evidence.get("transition")):
        if key and key in overrides:
            level, rule = Severity(overrides[key]), f"subtype:{cat}/{key}"

    ev = change.evidence
    if change.category in (C.NUMERIC, C.TEMPORAL) and change.subtype in ("INCREASED", "DECREASED"):
        conf = sev["numeric"] if change.category == C.NUMERIC else sev["temporal"]
        rel = ev.get("relative_change", 0.0)
        if rel >= conf.get("high_relative_change", 0.1):
            level, rule = Severity.HIGH, f"{cat.lower()}:relative_change>={conf['high_relative_change']}"
        elif ev.get("dimension") == "percent" and abs(ev.get("absolute_change", 0)) >= conf.get("high_percent_points", 1.0):
            level, rule = Severity.HIGH, f"{cat.lower()}:percentage_points>={conf.get('high_percent_points', 1.0)}"
        elif ev.get("comparator") and conf.get("threshold_is_high", True):
            level, rule = Severity.HIGH, f"{cat.lower()}:threshold_with_comparator"
    if change.subtype == "COMPARATOR_CHANGED":
        level, rule = Severity(sev["numeric"].get("comparator_change", "HIGH")), "numeric:comparator_changed"
    if change.category == C.TEMPORAL and change.subtype == "CHANGED":
        level, rule = Severity(sev["temporal"].get("date_change", "HIGH")), "temporal:date_changed"
    if change.category == C.STRUCTURE and change.subtype in ("ADDED", "REMOVED") and normative \
            and sev.get("structure", {}).get("normative_escalates", True):
        level, rule = Severity.HIGH, "structure:normative_provision"

    if in_definition and sev.get("escalate_in_definition", True) and change.category != C.STRUCTURE:
        if _UP[level] != level:
            level, rule = _UP[level], rule + "+definition"

    change.severity = level
    change.severity_rule = rule

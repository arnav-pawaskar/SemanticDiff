"""Evidence-linked revision brief and portable human review decisions."""
from __future__ import annotations

import hashlib
import json

from semanticdiff.models import ChangeCategory, DiffReport, PairStatus, Severity, Verdict
from semanticdiff.report.serialize import report_to_json

_CATEGORY_ORDER = {name: index for index, name in enumerate((
    ChangeCategory.NUMERIC, ChangeCategory.TEMPORAL, ChangeCategory.NEGATION,
    ChangeCategory.MODALITY, ChangeCategory.REVERSAL, ChangeCategory.CONDITION,
    ChangeCategory.SCOPE, ChangeCategory.ENTITY, ChangeCategory.UNEXPLAINED,
    ChangeCategory.STRUCTURE))}
_DECISIONS = {"Confirmed", "Corrected", "Dismissed"}
_CORRECTED_VERDICTS = {"MATERIAL", "MEANING_PRESERVING", "UNCLASSIFIED"}


def document_fingerprint(old: bytes, new: bytes) -> str:
    digest = hashlib.sha256()
    for content in (old, new):
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def pair_key(pair) -> str:
    identity = [pair.pair.v1_id, pair.pair.v2_id,
                pair.pair.v1_ids, pair.pair.v2_ids,
                pair.v1_text, pair.v2_text, pair.pair.status.value]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:20]


def revision_brief(report: DiffReport, limit: int = 5) -> tuple[str, list[dict]]:
    pairs = report.pairs
    material = sum(p.verdict == Verdict.MATERIAL and p.pair.status != PairStatus.UNCHANGED
                   for p in pairs)
    uncertain = sum(p.verdict == Verdict.UNCLASSIFIED and p.pair.status != PairStatus.UNCHANGED
                    for p in pairs)
    headline = (f"{material} changes flagged as material; {uncertain} need human review. "
                f"{report.summary.get('added', 0)} passages added and "
                f"{report.summary.get('removed', 0)} removed.")
    ranked = []
    for index, pair in enumerate(pairs):
        if pair.pair.status == PairStatus.UNCHANGED:
            continue
        changes = [change for change in pair.changes if change.category != ChangeCategory.STRUCTURE]
        if changes:
            change = min(changes, key=lambda item: (-item.severity.rank,
                                                     _CATEGORY_ORDER.get(item.category, 99)))
            old = (change.old or "∅").strip()
            new = (change.new or "∅").strip()
            detail = f"{change.description}: {old[:70]} → {new[:70]}"
            priority = (0 if pair.verdict == Verdict.MATERIAL else 1,
                        -change.severity.rank, _CATEGORY_ORDER.get(change.category, 99), index)
        else:
            detail = ("Passage added" if pair.pair.status == PairStatus.ADDED else
                      "Passage removed" if pair.pair.status == PairStatus.REMOVED else
                      "Wording changed; interpretation needs review")
            priority = (2 if pair.verdict == Verdict.UNCLASSIFIED else 3, 0, 99, index)
        ranked.append((priority, {"pair_index": index, "detail": detail,
                                  "section": pair.v2_section or pair.v1_section,
                                  "old_page": pair.v1_page, "new_page": pair.v2_page,
                                  "verdict": pair.verdict.value}))
    return headline, [item for _, item in sorted(ranked, key=lambda row: row[0])[:limit]]


def review_candidates(report: DiffReport) -> list[int]:
    indices = [index for index, pair in enumerate(report.pairs)
               if pair.pair.status != PairStatus.UNCHANGED]
    return sorted(indices, key=lambda index: (
        0 if report.pairs[index].verdict == Verdict.UNCLASSIFIED else
        1 if report.pairs[index].verdict == Verdict.MATERIAL else 2,
        -max((change.severity.rank for change in report.pairs[index].changes), default=0),
        index))


def reviewed_report_json(report: DiffReport, fingerprint: str, decisions: dict[str, dict]) -> str:
    exported = json.loads(report_to_json(report))
    keys = {pair_key(pair): index for index, pair in enumerate(report.pairs)}
    exported["human_review"] = {
        "schema_version": 1, "document_sha256": fingerprint,
        "decisions": [{"pair_key": key, "pair_index": keys[key], **decision}
                      for key, decision in decisions.items() if key in keys],
    }
    return json.dumps(exported, indent=2, ensure_ascii=False)


def import_decisions(data: bytes, report: DiffReport, fingerprint: str) -> dict[str, dict]:
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("This JSON is not a reviewed SemanticDiff report.")
    review = payload.get("human_review")
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ValueError("This JSON does not contain a supported human review.")
    if review.get("document_sha256") != fingerprint:
        raise ValueError("This review belongs to a different pair of documents.")
    known = {pair_key(pair) for pair in report.pairs}
    decisions = {}
    items = review.get("decisions")
    if not isinstance(items, list):
        raise ValueError("Review JSON has an invalid decisions list.")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Review JSON has an invalid decision.")
        key = item.get("pair_key")
        if key not in known:
            continue
        decision = item.get("decision")
        verdict = item.get("corrected_verdict")
        note = str(item.get("note", ""))[:2000]
        if decision not in _DECISIONS:
            raise ValueError("Review JSON contains an unknown decision.")
        if decision == "Corrected" and verdict not in _CORRECTED_VERDICTS:
            raise ValueError("A corrected finding needs a valid verdict.")
        decisions[key] = {"decision": decision,
                          "corrected_verdict": verdict if decision == "Corrected" else None,
                          "note": note}
    return decisions

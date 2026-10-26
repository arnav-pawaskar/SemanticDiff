import json

import fitz
import pytest

from app.review_workflow import (document_fingerprint, import_decisions, pair_key,
                                 reviewed_report_json, revision_brief)
from app.source_preview import render_source_page
from semanticdiff.models import (AlignedPair, AtomicChange, ChangeCategory,
                                 DiffReport, PairAnalysis, PairStatus, Severity, Verdict)


def _report():
    pair = PairAnalysis(AlignedPair("v1-000", "v2-000", PairStatus.MODIFIED),
                        "BLEU score was 41.0.", "BLEU score was 41.2.",
                        v1_page=1, v2_page=1, verdict=Verdict.MATERIAL,
                        changes=[AtomicChange("c1", ChangeCategory.NUMERIC, "INCREASED",
                                              "BLEU score increased", "41.0", "41.2",
                                              severity=Severity.HIGH)])
    return DiffReport("old.pdf", "new.pdf", [pair],
                      summary={"added": 0, "removed": 0})


def test_revision_brief_and_review_roundtrip():
    report = _report()
    headline, items = revision_brief(report)
    assert "1 changes flagged as material" in headline
    assert items[0]["pair_index"] == 0
    assert "41.0 → 41.2" in items[0]["detail"]
    fingerprint = document_fingerprint(b"old", b"new")
    key = pair_key(report.pairs[0])
    exported = reviewed_report_json(report, fingerprint, {
        key: {"decision": "Corrected", "corrected_verdict": "UNCLASSIFIED",
              "note": "Check source table."}})
    assert json.loads(exported)["pairs"][0]["verdict"] == "MATERIAL"
    assert import_decisions(exported.encode(), report, fingerprint)[key]["note"] == "Check source table."
    with pytest.raises(ValueError, match="different pair"):
        import_decisions(exported.encode(), report, document_fingerprint(b"other", b"new"))
    with pytest.raises(ValueError, match="not a reviewed"):
        import_decisions(b"[]", report, fingerprint)
    malformed = json.loads(exported)
    malformed["human_review"]["decisions"] = [None]
    with pytest.raises(ValueError, match="invalid decision"):
        import_decisions(json.dumps(malformed).encode(), report, fingerprint)


def test_source_preview_marks_pdf_text_region():
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((50, 120), "The BLEU score was 41.2.")
    data = doc.tobytes()
    doc.close()
    full, focused, label = render_source_page(data, 1, "The BLEU score was 41.2.",
                                              (45, 100, 260, 130))
    assert full.startswith(b"\x89PNG")
    assert focused and focused.startswith(b"\x89PNG")
    assert label == "Exact text match"

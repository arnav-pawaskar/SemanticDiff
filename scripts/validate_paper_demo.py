"""Check a few manually verified anchors in the public arXiv revision pair.

Run after comparing attention_v1.pdf and attention_v2.pdf with --json. This is a
small real-document regression check, not a claim of full paper-level accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    abstract = [p for p in report["pairs"] if p["v2_section"] == "Abstract"]
    assert report["meta"]["document_types"] == {"v1": "paper", "v2": "paper"}
    if "bert" in path.stem.lower():
        score = next(p for p in abstract if "80.5%" in (p["v2_text"] or ""))
        assert "80.4%" in score["v1_text"]
        assert ("NUMERIC", "80.4%", "80.5%") in {
            (c["category"], c["old"], c["new"]) for c in score["changes"]}
        assert not any(c["category"] == "SCOPE" for c in score["changes"])
        assert score["verdict"] == "MATERIAL"
        assert any(p["pair"]["moved"] and p["v2_section"] == "Section C"
                   for p in report["pairs"])
        return
    score = next(p for p in abstract if "BLEU score of 41.2" in (p["v2_text"] or ""))
    assert "BLEU score of 41.0" in score["v1_text"]
    assert {("NUMERIC", "41.0", "41.2"), ("TEMPORAL", "3.5 days", "4.5 days")} <= {
        (c["category"], c["old"], c["new"]) for c in score["changes"]}
    assert score["verdict"] == "MATERIAL"
    assert report["summary"]["paper_review"] > 0
    assert report["summary"]["moved"] <= 5, "PDF reflow appears to be mislabeled as moves"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    validate(args.report)
    print("Real paper anchors passed.")

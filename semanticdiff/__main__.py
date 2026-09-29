"""Command-line interface.

    python -m semanticdiff data/demo/academic_regulations_v1.txt data/demo/academic_regulations_v2.txt
    python -m semanticdiff v1.pdf v2.pdf --json report.json --all
    python -m semanticdiff --pair "Students may access the lab." "Students may not access the lab."
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

_SYM = {"UNCHANGED": "=", "MODIFIED": "~", "ADDED": "+", "REMOVED": "-"}


def _print_pair(pa, show_all: bool) -> None:
    st = pa.pair.status.value
    if st == "UNCHANGED" and not pa.pair.moved and not pa.affected_by and not show_all:
        return
    sym = _SYM[st] + ("↕" if pa.pair.moved else " ")
    sec = pa.v2_section or pa.v1_section
    print(f"\n{sym} [{sec}] {pa.verdict.value}  text:{pa.textual_level.value}  meaning:{pa.semantic_level.value}"
          f"  ({pa.verdict_rule})")
    if pa.v1_text and st != "ADDED":
        print(f"    OLD: {pa.v1_text}")
    if pa.v2_text and st != "REMOVED":
        print(f"    NEW: {pa.v2_text}")
    for ch in pa.changes:
        arrow = f"   {ch.old!s} → {ch.new!s}" if (ch.old or ch.new) and ch.category.value != "STRUCTURE" else ""
        print(f"      [{ch.severity.value:<6}] {ch.category.value}/{ch.subtype}: {ch.description}{arrow}")
    for n in pa.notes:
        print(f"      note: {n}")
    if pa.nli:
        print(f"      NLI: entail {pa.nli.entail_12:.2f}/{pa.nli.entail_21:.2f}  "
              f"contra {pa.nli.contra_12:.2f}/{pa.nli.contra_21:.2f}   emb-cos {pa.pair.sim_embed:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="semanticdiff", description="Meaning-aware document comparison")
    ap.add_argument("v1", nargs="?")
    ap.add_argument("v2", nargs="?")
    ap.add_argument("--pair", nargs=2, metavar=("OLD", "NEW"), help="compare two sentences")
    ap.add_argument("--json", help="write the full DiffReport as JSON")
    ap.add_argument("--all", action="store_true", help="also print unchanged clauses")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    from semanticdiff.pipeline import SemanticDiff
    from semanticdiff.report.serialize import save_report

    engine = SemanticDiff()
    if args.pair:
        _print_pair(engine.analyze_pair(*args.pair), True)
        return 0
    if not (args.v1 and args.v2):
        ap.error("give two documents or --pair")
    report = engine.compare(args.v1, args.v2, os.path.basename(args.v1), os.path.basename(args.v2))
    for pa in report.pairs:
        _print_pair(pa, args.all)
    print("\nImpacts:")
    for imp in report.impacts:
        print(f"  {imp.affected_provision_id}: {imp.reason}  path={' → '.join(imp.path)}")
    s = report.summary
    print(f"\nSummary: {s}\nDocument similarity: {report.doc_similarity}\nTimings: {report.meta['timings_s']}")
    if args.json:
        save_report(report, args.json)
        print(f"Report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Calibrate SemanticDiff's verdict thresholds on the DEV split only.

    python -m eval.calibrate

Grid-searches tau_e (bidirectional entailment), tau_c (contradiction) and tau_s
(embedding similarity for rule V4) to maximise Task-A macro-F1 on dev — the same
objective the baselines are tuned with — and writes config/calibrated.yaml, which
load_config() merges automatically. Deterministic extractor output is computed once;
only the decision table (changes/verdict.py) is re-evaluated per grid point.
"""
from __future__ import annotations

import itertools
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import yaml
from sklearn.metrics import f1_score

from semanticdiff.changes.verdict import decide
from semanticdiff.config import CONFIG_DIR, load_config
from semanticdiff.models import PairStatus, Verdict
from semanticdiff.pipeline import SemanticDiff

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "eval" / "generated" / "benchmark.jsonl"
_EXTRA_IDS = {"nli1", "u1"}          # changes added by the decision table itself


def main() -> None:
    calibrated = CONFIG_DIR / "calibrated.yaml"
    if calibrated.exists():
        calibrated.unlink()          # start from the defaults
    items = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = [it for it in items if it["split"] == "dev"]
    cfg = load_config()
    engine = SemanticDiff(cfg)
    pas = engine.analyze_pairs([(it["v1"], it["v2"]) for it in dev])
    gold = [bool(it["material"]) for it in dev]
    base = [(pa, [c for c in pa.changes if c.id not in _EXTRA_IDS],
             len(pa.frames.get("residual", {}).get("old", [])) + len(pa.frames.get("residual", {}).get("new", [])))
            for pa in pas]

    grid = {"entail": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "contradiction": [0.5, 0.6, 0.7, 0.8, 0.9],
            "preserve_similarity": [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 1.01]}
    results = []
    for te, tc, ts in itertools.product(*grid.values()):
        c = {**cfg, "verdict": {**cfg["verdict"], "entail": te, "contradiction": tc, "preserve_similarity": ts}}
        pred = []
        for pa, changes, residual in base:
            v, *_ = decide(changes, pa.nli, pa.pair.sim_embed, residual, c,
                           identical=pa.pair.status == PairStatus.UNCHANGED)
            pred.append(v in (Verdict.MATERIAL, Verdict.UNCLASSIFIED))
        results.append((f1_score(gold, pred, average="macro"), -abs(te - 0.6), te, tc, ts))
    results.sort(reverse=True)
    best = results[0]
    out = {"verdict": {"entail": best[2], "contradiction": best[3], "preserve_similarity": best[4]}}
    calibrated.write_text("# written by eval/calibrate.py (dev split only)\n" + yaml.safe_dump(out), encoding="utf-8")
    print(f"dev macro-F1 {best[0]:.3f} with {out['verdict']}  ->  {calibrated}")
    print("top 5:", [(round(r[0], 3), r[2:]) for r in results[:5]])


if __name__ == "__main__":
    main()

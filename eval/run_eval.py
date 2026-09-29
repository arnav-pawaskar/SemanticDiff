"""Run the SemanticDiff evaluation (Tasks A and B, ablations, figures).

    python -m eval.run_eval            # uses data/eval/generated/benchmark.jsonl (+ natural set if present)

Outputs in results/: eval_summary.md, taskA_*.csv, taskB_*.csv, predictions.jsonl, fig_*.png
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, hamming_loss,
                             precision_recall_fscore_support)
from sklearn.preprocessing import MultiLabelBinarizer

from eval.baselines import (CATEGORIES, keyword_diff_labels, lexical_dissimilarity, tfidf_dissimilarity,
                            tune_embed_nli, tune_threshold)
from semanticdiff.align.similarity import embed
from semanticdiff.config import load_config
from semanticdiff.models import ChangeCategory, Verdict
from semanticdiff.pipeline import SemanticDiff

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "eval" / "generated" / "benchmark.jsonl"
NATURAL = ROOT / "data" / "eval" / "natural" / "pairs.jsonl"
OUT = ROOT / "results"

SYSTEMS_A = ["lexical", "tfidf", "embedding", "embed+nli", "semanticdiff"]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _categories(pa) -> set[str]:
    return {c.category.value for c in pa.changes
            if c.category not in (ChangeCategory.STRUCTURE, ChangeCategory.UNEXPLAINED)}


def run_system(engine: SemanticDiff, items: list[dict]) -> list[dict]:
    pas = engine.analyze_pairs([(it["v1"], it["v2"]) for it in items])
    out = []
    for pa in pas:
        out.append({
            "material": pa.verdict in (Verdict.MATERIAL, Verdict.UNCLASSIFIED),
            "labels": sorted(_categories(pa)), "verdict": pa.verdict.value, "rule": pa.verdict_rule,
            "textual_change": pa.textual_change, "cos": pa.pair.sim_embed,
            "entail": pa.nli.bidirectional_entailment if pa.nli else 1.0,
            "contra": pa.nli.max_contradiction if pa.nli else 0.0,
            "changes": [f"{c.category.value}/{c.subtype}: {c.old} -> {c.new}" for c in pa.changes],
        })
    return out


def binary_metrics(gold, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(gold, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(gold, pred), "precision": p, "recall": r, "f1": f,
            "macro_f1": f1_score(gold, pred, average="macro", zero_division=0)}


def mcnemar(gold, a, b) -> float:
    """Exact McNemar test on the discordant pairs of two classifiers."""
    ca, cb = (np.asarray(a) == gold), (np.asarray(b) == gold)
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    if n01 + n10 == 0:
        return 1.0
    return binomtest(n01, n01 + n10, 0.5).pvalue


def main() -> None:
    OUT.mkdir(exist_ok=True)
    items = load(BENCH)
    if NATURAL.exists():
        nat = load(NATURAL)
        for it in nat:
            it.update(split="natural", source="natural", task_b=True)
        items += nat
        print(f"natural set: {len(nat)} pairs")
    print(f"benchmark: {len(items)} pairs")

    cfg = load_config()
    full = SemanticDiff(cfg)
    sd = run_system(full, items)
    no_nli = run_system(SemanticDiff(cfg, use_nli=False), items)
    no_qn = run_system(SemanticDiff(load_config({"ablations": ["no_quantity_normalization"]})), items)

    df = pd.DataFrame(items)
    df["gold"] = df["material"].astype(bool)
    df["lexical"] = [lexical_dissimilarity(a, b) for a, b in zip(df.v1, df.v2)]
    corpus = list(df.v1) + list(df.v2)
    df["tfidf"] = tfidf_dissimilarity(list(zip(df.v1, df.v2)), corpus)
    e1, e2 = embed(list(df.v1), full.encoder), embed(list(df.v2), full.encoder)
    df["cos"] = np.sum(e1 * e2, axis=1)
    df["embedding"] = 1.0 - df["cos"]
    df["entail"] = [r["entail"] for r in sd]
    df["sd_pred"] = [r["material"] for r in sd]
    df["sd_no_nli"] = [r["material"] for r in no_nli]
    df["sd_no_qn"] = [r["material"] for r in no_qn]

    dev, test_mask = df.split == "dev", df.split == "test"
    thresholds = {}
    for name in ("lexical", "tfidf", "embedding"):
        t, _ = tune_threshold(df.loc[dev, name].values, df.loc[dev, "gold"].values)
        thresholds[name] = t
        df[f"{name}_pred"] = df[name] >= t
    t1, t2, _ = tune_embed_nli(df.loc[dev, "cos"].values, df.loc[dev, "entail"].values, df.loc[dev, "gold"].values)
    thresholds["embed+nli"] = {"cos": t1, "entail": t2}
    df["embed+nli_pred"] = ~((df.cos >= t1) & (df.entail >= t2))
    df["semanticdiff_pred"] = df.sd_pred

    # ---------------------------------------------------------------- Task A
    rows = []
    eval_splits = ["test"] + (["natural"] if (df.split == "natural").any() else [])
    systems = SYSTEMS_A + ["semanticdiff −NLI", "semanticdiff −unit-norm"]
    pred_col = {s: f"{s}_pred" for s in SYSTEMS_A} | {"semanticdiff −NLI": "sd_no_nli",
                                                       "semanticdiff −unit-norm": "sd_no_qn"}
    for split in eval_splits:
        m = df.split == split
        for s in systems:
            rows.append({"split": split, "system": s, **binary_metrics(df.loc[m, "gold"], df.loc[m, pred_col[s]])})
    taskA = pd.DataFrame(rows)
    taskA.to_csv(OUT / "taskA_metrics.csv", index=False)

    # recall / specificity by source category (test)
    t = df[test_mask].copy()
    t["cat"] = t.source.str.replace("synthetic:", "", regex=False)
    by_cat = []
    for cat, g in t.groupby("cat"):
        row = {"category": cat, "n": len(g)}
        for s in systems:
            pred = g[pred_col[s]].astype(bool)
            # for meaning-preserving categories report specificity (correctly NOT flagged)
            row[s] = float((pred == g.gold).mean())
        by_cat.append(row)
    by_cat = pd.DataFrame(by_cat)
    by_cat.to_csv(OUT / "taskA_by_category.csv", index=False)

    # danger zones (test)
    lowtext_material = t[(t.lexical < 0.15) & t.gold]
    hightext_preserving = t[(t.lexical >= 0.25) & ~t.gold]
    danger = pd.DataFrame([{
        "system": s,
        f"recall | small edit, meaning changed (n={len(lowtext_material)})": float(lowtext_material[pred_col[s]].mean()),
        f"specificity | big edit, meaning kept (n={len(hightext_preserving)})":
            float((~hightext_preserving[pred_col[s]].astype(bool)).mean()) if len(hightext_preserving) else float("nan"),
    } for s in systems])
    danger.to_csv(OUT / "taskA_danger_zones.csv", index=False)

    sig = pd.DataFrame([{"comparison": f"semanticdiff vs {s}",
                         "p_value": mcnemar(t.gold.values, t.semanticdiff_pred.values, t[pred_col[s]].values)}
                        for s in SYSTEMS_A[:-1]])
    sig.to_csv(OUT / "taskA_mcnemar.csv", index=False)

    # ---------------------------------------------------------------- Task B
    df["sd_labels"] = [r["labels"] for r in sd]
    df["sd_no_nli_labels"] = [r["labels"] for r in no_nli]
    df["sd_no_qn_labels"] = [r["labels"] for r in no_qn]
    df["kw_labels"] = [sorted(keyword_diff_labels(a, b)) for a, b in zip(df.v1, df.v2)]
    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    b_systems = {"keyword-diff": "kw_labels", "semanticdiff": "sd_labels",
                 "semanticdiff −NLI": "sd_no_nli_labels", "semanticdiff −unit-norm": "sd_no_qn_labels"}
    rowsB, per_label = [], []
    for split in eval_splits:
        m = (df.split == split) & df.task_b.fillna(True).astype(bool)
        Y = mlb.fit_transform(df.loc[m, "labels"])
        for s, col in b_systems.items():
            P = mlb.transform(df.loc[m, col])
            rowsB.append({"split": split, "system": s,
                          "micro_f1": f1_score(Y, P, average="micro", zero_division=0),
                          "macro_f1": f1_score(Y, P, average="macro", zero_division=0),
                          "exact_match": accuracy_score(Y, P), "hamming_loss": hamming_loss(Y, P)})
            p, r, f, n = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
            for k, c in enumerate(CATEGORIES):
                per_label.append({"split": split, "system": s, "category": c, "precision": p[k],
                                  "recall": r[k], "f1": f[k], "support": int(n[k])})
    taskB = pd.DataFrame(rowsB)
    taskB.to_csv(OUT / "taskB_metrics.csv", index=False)
    perlab = pd.DataFrame(per_label)
    perlab.to_csv(OUT / "taskB_per_label.csv", index=False)

    # single-label confusion matrix for SemanticDiff (test)
    single = t[t.labels.apply(len) <= 1].copy()
    gold_lab = single.labels.apply(lambda l: l[0] if l else "NONE")
    pred_lab = df.loc[single.index, "sd_labels"].apply(lambda l: l[0] if len(l) == 1 else ("NONE" if not l else "MULTI"))
    cm_labels = CATEGORIES + ["NONE", "MULTI"]
    cm = confusion_matrix(gold_lab, pred_lab, labels=cm_labels)
    pd.DataFrame(cm, index=cm_labels, columns=cm_labels).to_csv(OUT / "taskB_confusion_semanticdiff.csv")

    # ---------------------------------------------------------------- predictions & report
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for (_, row), r in zip(df.iterrows(), sd):
            fh.write(json.dumps({"id": row.id, "split": row.split, "source": row.source, "v1": row.v1, "v2": row.v2,
                                 "gold_material": bool(row.gold), "gold_labels": row.labels,
                                 "pred_material": r["material"], "pred_labels": r["labels"],
                                 "rule": r["rule"], "changes": r["changes"],
                                 "lexical": round(row.lexical, 3), "cos": round(float(row.cos), 3),
                                 "entail": round(r["entail"], 3)}, ensure_ascii=False) + "\n")

    from eval.figures import make_figures
    make_figures(OUT, taskA, by_cat, danger, t, pred_col, cm, cm_labels, perlab)

    def md(d: pd.DataFrame) -> str:
        return d.to_markdown(index=False, floatfmt=".3f")

    counts = df.groupby(["source", "split"]).size().unstack(fill_value=0)
    report = [
        "# SemanticDiff evaluation", "",
        f"Pairs: {len(df)} (dev {int(dev.sum())}, test {int(test_mask.sum())}"
        + (f", natural {int((df.split == 'natural').sum())}" if (df.split == 'natural').any() else "") + ")", "",
        counts.to_markdown(), "",
        "Baseline thresholds (tuned on dev to maximise macro-F1):", "", f"`{json.dumps(thresholds)}`", "",
        "## Task A — did the meaning change? (binary)", "", md(taskA), "",
        "### Accuracy per source category (test) — paraphrase = specificity", "", md(by_cat), "",
        "### Danger zones (test)", "", md(danger), "",
        "### McNemar exact test (test): SemanticDiff vs baselines", "", md(sig), "",
        "## Task B — what changed? (multi-label, 8 categories)", "", md(taskB), "",
        "### Per-category F1 (test)", "",
        perlab[perlab.split == "test"].pivot(index="category", columns="system", values="f1").to_markdown(floatfmt=".3f"), "",
        "### SemanticDiff confusion matrix (single-label test pairs; rows = gold)", "",
        pd.DataFrame(cm, index=cm_labels, columns=cm_labels).to_markdown(), "",
    ]
    (OUT / "eval_summary.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

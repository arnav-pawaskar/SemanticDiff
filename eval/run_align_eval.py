"""Task C — provision alignment evaluation on synthetic document pairs.

    python -m eval.run_align_eval [--docs 40]

Each document is 12–18 seed provisions; version 2 applies scripted edits with known
gold links: delete 1–2, insert 1–2 unseen provisions, modify 2–4 (a category
perturbation or a paraphrase), and move 1 provision elsewhere. Documents are split
dev/test; the alignment threshold is tuned on dev (link F1) and written to
config/calibrated.yaml.
"""
from __future__ import annotations

import argparse
import os
import random
import warnings
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

from eval.build_benchmark import load_seeds
from eval.perturb import GENERATORS, paraphrase
from semanticdiff.align.aligner import align
from semanticdiff.align.similarity import embed
from semanticdiff.config import CONFIG_DIR, load_config
from semanticdiff.ingest.segment import normalize_for_match
from semanticdiff.models import Provision
from semanticdiff.resources import get_encoder, get_spacy

OUT = Path(__file__).resolve().parent.parent / "results"


def make_doc_pair(rng: random.Random, seeds: list[str], nlp) -> dict:
    n = rng.randint(12, 18)
    idx = rng.sample(range(len(seeds)), n + 2)
    v1 = [seeds[i] for i in idx[:n]]
    spare = [seeds[i] for i in idx[n:]]
    # v2 entries: (text, source index in v1 or None)
    v2 = [(t, i) for i, t in enumerate(v1)]
    for _ in range(rng.randint(1, 2)):                                   # deletions
        v2.pop(rng.randrange(len(v2)))
    for k in rng.sample(range(len(v2)), rng.randint(2, 4)):              # modifications
        text, src = v2[k]
        doc = nlp(text)
        gens = list(GENERATORS.values()) + [paraphrase]
        rng.shuffle(gens)
        for g in gens:
            out = g(doc, rng, "test", nlp)
            if out and normalize_for_match(out) != normalize_for_match(text):
                v2[k] = (out, src)
                break
    src_pos = rng.randrange(len(v2))                                     # one move, >= 2 slots away
    item = v2.pop(src_pos)
    choices = [p for p in range(len(v2) + 1) if abs(p - src_pos) >= 2] or list(range(len(v2) + 1))
    v2.insert(rng.choice(choices), item)
    for t in spare[: rng.randint(1, 2)]:                                 # insertions
        v2.insert(rng.randrange(len(v2) + 1), (t, None))
    links = {(src, j) for j, (_, src) in enumerate(v2) if src is not None}
    # gold moves = matched pairs off the LIS of the gold links (minimal move set)
    from semanticdiff.align.aligner import longest_increasing_subsequence
    ordered = sorted(links)
    keep = longest_increasing_subsequence([j for _, j in ordered])
    moved = {ordered[k] for k in range(len(ordered)) if k not in keep}
    return {"v1": v1, "v2": [t for t, _ in v2], "links": links, "moved": moved}


def _provs(texts, version):
    return [Provision(id=f"{version}-{k:03d}", version=version, index=k, text=t) for k, t in enumerate(texts)]


def sys_positional(d) -> tuple[set, set]:
    return {(k, k) for k in range(min(len(d["v1"]), len(d["v2"])))}, set()


def sys_difflib(d) -> tuple[set, set]:
    a = [normalize_for_match(t) for t in d["v1"]]
    b = [normalize_for_match(t) for t in d["v2"]]
    links = set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal" or tag == "replace":
            links |= {(i1 + k, j1 + k) for k in range(min(i2 - i1, j2 - j1))}
    return links, set()


def sys_semanticdiff(d, e1, e2, alpha, threshold) -> tuple[set, set]:
    p1, p2 = _provs(d["v1"], "v1"), _provs(d["v2"], "v2")
    pairs = align(p1, p2, e1 if alpha > 0 else None, e2 if alpha > 0 else None, alpha, threshold)
    links, moved = set(), set()
    for pr in pairs:
        if pr.v1_id and pr.v2_id:
            link = (int(pr.v1_id[3:]), int(pr.v2_id[3:]))
            links.add(link)
            if pr.moved:
                moved.add(link)
    return links, moved


def prf(gold: set, pred: set) -> tuple[float, float, float]:
    tp = len(gold & pred)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(docs, embs, system) -> dict:
    G_l, P_l, G_m, P_m = set(), set(), set(), set()
    for k, (d, (e1, e2)) in enumerate(zip(docs, embs)):
        links, moved = system(d, e1, e2)
        G_l |= {(k, *x) for x in d["links"]}
        P_l |= {(k, *x) for x in links}
        G_m |= {(k, *x) for x in d["moved"]}
        P_m |= {(k, *x) for x in moved}
    lp, lr, lf = prf(G_l, P_l)
    mp, mr, mf = prf(G_m, P_m)
    return {"link_P": lp, "link_R": lr, "link_F1": lf, "move_P": mp, "move_R": mr, "move_F1": mf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=40)
    args = ap.parse_args()
    rng = random.Random(7)
    nlp = get_spacy()
    cfg = load_config()
    enc = get_encoder(cfg["models"]["embedding"], cfg.get("device", "auto"))
    seeds = load_seeds()
    docs = [make_doc_pair(rng, seeds, nlp) for _ in range(args.docs)]
    embs = [(embed(d["v1"], enc), embed(d["v2"], enc)) for d in docs]
    dev = [k for k in range(len(docs)) if k % 2 == 0]
    test = [k for k in range(len(docs)) if k % 2 == 1]
    sub = lambda ks: ([docs[k] for k in ks], [embs[k] for k in ks])

    alpha = cfg["align"]["alpha"]
    sweep = []
    for th in np.round(np.arange(0.30, 0.86, 0.05), 2):
        r = evaluate(*sub(dev), lambda d, e1, e2: sys_semanticdiff(d, e1, e2, alpha, th))
        sweep.append({"threshold": float(th), **r})
    sweep = pd.DataFrame(sweep)
    best = float(sweep.sort_values(["link_F1", "threshold"], ascending=[False, True]).iloc[0].threshold)

    path = CONFIG_DIR / "calibrated.yaml"
    cal = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cal = cal or {}
    cal.setdefault("align", {})["threshold"] = best
    path.write_text("# written by eval/calibrate.py and eval/run_align_eval.py (dev split only)\n"
                    + yaml.safe_dump(cal), encoding="utf-8")

    systems = {
        "positional (i ↔ i)": lambda d, e1, e2: sys_positional(d),
        "difflib line diff": lambda d, e1, e2: sys_difflib(d),
        "SemanticDiff (token overlap only)": lambda d, e1, e2: sys_semanticdiff(d, e1, e2, 0.0, best),
        "SemanticDiff (embedding + token)": lambda d, e1, e2: sys_semanticdiff(d, e1, e2, alpha, best),
    }
    rows = [{"system": name, **evaluate(*sub(test), fn)} for name, fn in systems.items()]
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "taskC_alignment.csv", index=False)
    sweep.to_csv(OUT / "taskC_threshold_sweep_dev.csv", index=False)
    n_links = sum(len(docs[k]["links"]) for k in test)
    md = ["# Task C — provision alignment", "",
          f"{len(docs)} synthetic document pairs ({len(test)} test, {n_links} gold links); "
          f"threshold tuned on dev = {best}", "", res.to_markdown(index=False, floatfmt=".3f"), "",
          "Dev threshold sweep (SemanticDiff, embedding + token):", "",
          sweep.to_markdown(index=False, floatfmt=".3f")]
    (OUT / "taskC_alignment.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:6]))


if __name__ == "__main__":
    main()

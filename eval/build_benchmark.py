"""Build the controlled SemanticDiff benchmark.

    python -m eval.build_benchmark [--per-category 36] [--paraphrases 90] [--compound 40] [--paws 40]

Seeds are split 50/50 into dev/test *by seed*, so no seed sentence appears in both.
Output: data/eval/generated/benchmark.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from eval.perturb import GENERATORS, compound, paraphrase
from semanticdiff.ingest.segment import normalize_for_match
from semanticdiff.resources import get_spacy

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data" / "eval" / "seeds" / "seeds.txt"
OUT = ROOT / "data" / "eval" / "generated" / "benchmark.jsonl"


def load_seeds() -> list[str]:
    return [l.strip() for l in SEEDS.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def _paws(n: int, rng: random.Random) -> list[dict]:
    """Out-of-domain pairs from PAWS (optional; needs the HF datasets cache or network)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("paws", "labeled_final", split="test")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[paws] skipped: {exc}")
        return []
    pos = [r for r in ds if r["label"] == 1]
    neg = [r for r in ds if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = []
    for k, r in enumerate(pos[:n]):
        out.append({"v1": r["sentence1"], "v2": r["sentence2"], "labels": [], "material": False,
                    "source": "paws:paraphrase", "split": "dev" if k % 2 == 0 else "test", "task_b": False})
    for k, r in enumerate(neg[:n]):
        out.append({"v1": r["sentence1"], "v2": r["sentence2"], "labels": [], "material": True,
                    "source": "paws:non-paraphrase", "split": "dev" if k % 2 == 0 else "test", "task_b": False})
    return out


def build(per_category: int, n_para: int, n_compound: int, n_paws: int, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    nlp = get_spacy()
    seeds = load_seeds()
    order = list(range(len(seeds)))
    rng.shuffle(order)
    split_of = {i: ("dev" if k % 2 == 0 else "test") for k, i in enumerate(order)}
    docs = list(nlp.pipe(seeds))

    pools: dict[str, list[dict]] = {c: [] for c in [*GENERATORS, "PARAPHRASE", "COMPOUND"]}
    for i, (text, doc) in enumerate(zip(seeds, docs)):
        split = split_of[i]
        for cat, gen in GENERATORS.items():
            for attempt in range(3):                  # a few variants per seed
                out = gen(doc, random.Random(seed * 1000 + i * 17 + attempt * 5 + list(GENERATORS).index(cat) * 101), split, nlp)
                if out and normalize_for_match(out) != normalize_for_match(text):
                    pools[cat].append({"v1": text, "v2": out, "labels": [cat], "material": True,
                                       "source": f"synthetic:{cat.lower()}", "split": split, "seed_id": i})
        for attempt in range(2):
            out = paraphrase(doc, random.Random(seed * 7 + i * 31 + attempt), split, nlp)
            if out and normalize_for_match(out) != normalize_for_match(text):
                pools["PARAPHRASE"].append({"v1": text, "v2": out, "labels": [], "material": False,
                                            "source": "synthetic:paraphrase", "split": split, "seed_id": i})
        res = compound(doc, random.Random(seed * 3 + i), split, nlp)
        if res:
            pools["COMPOUND"].append({"v1": text, "v2": res[0], "labels": res[1], "material": True,
                                      "source": "synthetic:compound", "split": split, "seed_id": i})

    items: list[dict] = []
    quotas = {c: per_category for c in GENERATORS} | {"PARAPHRASE": n_para, "COMPOUND": n_compound}
    for cat, pool in pools.items():
        # de-duplicate, then sample evenly from both splits, at most one item per seed where possible
        uniq = {(p["v1"], p["v2"]): p for p in pool}
        pool = list(uniq.values())
        rng.shuffle(pool)
        pool.sort(key=lambda p: sum(1 for q in pool if q["seed_id"] == p["seed_id"]))
        chosen: list[dict] = []
        for split in ("dev", "test"):
            part = [p for p in pool if p["split"] == split]
            per_seed: Counter = Counter()
            for p in part:
                if len([c for c in chosen if c["split"] == split]) >= quotas[cat] // 2:
                    break
                if per_seed[p["seed_id"]] >= 1 and len(part) > quotas[cat]:
                    continue
                per_seed[p["seed_id"]] += 1
                chosen.append(p)
        items += chosen
    items += _paws(n_paws, rng) if n_paws else []
    for k, it in enumerate(items):
        it["id"] = f"b{k:04d}"
        it.setdefault("task_b", True)
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=36)
    ap.add_argument("--paraphrases", type=int, default=90)
    ap.add_argument("--compound", type=int, default=40)
    ap.add_argument("--paws", type=int, default=0, help="PAWS pairs per class (0 = skip)")
    args = ap.parse_args()
    items = build(args.per_category, args.paraphrases, args.compound, args.paws)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = Counter((it["source"], it["split"]) for it in items)
    for (src, split), n in sorted(c.items()):
        print(f"{src:<28} {split:<5} {n}")
    print(f"total {len(items)} -> {OUT}")


if __name__ == "__main__":
    main()

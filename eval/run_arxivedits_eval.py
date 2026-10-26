"""Evaluate paper sentence alignment on the authors' arXivEdits test split.

    python -m eval.run_arxivedits_eval [--limit-papers 5]

This evaluates clean, LaTeX-derived sentences, not PDF ingestion or semantic
change classification. The test data is cached locally and never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen

from semanticdiff.align.aligner import align, reconcile_split_merge
from semanticdiff.align.similarity import embed
from semanticdiff.config import load_config
from semanticdiff.ingest.segment import normalize_for_match
from semanticdiff.models import Provision
from semanticdiff.resources import get_encoder

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cache" / "arxivedits" / "test.json"
OUTPUT = ROOT / "results" / "arxivedits_scorecard.json"
SOURCE_REPO = "https://github.com/chaojiang06/arXivEdits"
SOURCE_COMMIT = "113913d68522c1c8428d8e191d5c9bed633da6ce"
SOURCE_SHA256 = "995f042d44e87c2d891429e2d30bf22c9f97a4dd9c3799854944325b31a18eb0"
SOURCE_URL = ("https://raw.githubusercontent.com/chaojiang06/arXivEdits/"
              f"{SOURCE_COMMIT}/data/sentence_alignment/test.json")


@dataclass
class Revision:
    paper_id: str
    old_version: str
    new_version: str
    old_ids: list[str]
    new_ids: list[str]
    old_texts: list[str]
    new_texts: list[str]
    gold: set[tuple[int, int]]


def load_test(path: Path = CACHE) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(SOURCE_URL, headers={"User-Agent": "SemanticDiff-benchmark/1.0"})
        with urlopen(request, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError("arXivEdits test download does not match the pinned checksum")
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"Cached arXivEdits test file has an unexpected checksum: {path}")
    return json.loads(data)


def revisions(payload: dict, limit_papers: int | None = None) -> list[Revision]:
    out = []
    for paper_id in sorted(payload)[:limit_papers]:
        paper = payload[paper_id]
        versions = sorted((key for key in paper if key.isdigit()), key=int)
        skipped = set(paper.get("skipped_alignment_indices", []))
        links = paper["alignment"]
        for old_v, new_v in zip(versions, versions[1:]):
            old = [(ident, text) for ident, text in paper[old_v].items()
                   if ident not in skipped and text.strip() and not text.lstrip().startswith("#")]
            new = [(ident, text) for ident, text in paper[new_v].items()
                   if ident not in skipped and text.strip() and not text.lstrip().startswith("#")]
            old_ids, new_ids = [ident for ident, _ in old], [ident for ident, _ in new]
            old_index = {ident: i for i, ident in enumerate(old_ids)}
            new_index = {ident: i for i, ident in enumerate(new_ids)}
            gold = {(old_index[a], new_index[b]) for a, b in links
                    if a in old_index and b in new_index}
            out.append(Revision(paper_id, old_v, new_v, old_ids, new_ids,
                                [text for _, text in old], [text for _, text in new], gold))
    return out


def positional(revision: Revision) -> set[tuple[int, int]]:
    return {(i, i) for i in range(min(len(revision.old_texts), len(revision.new_texts)))}


def difflib_links(revision: Revision) -> set[tuple[int, int]]:
    old = [normalize_for_match(text) for text in revision.old_texts]
    new = [normalize_for_match(text) for text in revision.new_texts]
    links = set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag in ("equal", "replace"):
            links.update((i1 + k, j1 + k) for k in range(min(i2 - i1, j2 - j1)))
    return links


def semanticdiff_links(revision: Revision, old_embeddings, new_embeddings,
                       alpha: float, threshold: float) -> set[tuple[int, int]]:
    old = [Provision(f"v1-{i:04d}", "v1", i, text)
           for i, text in enumerate(revision.old_texts)]
    new = [Provision(f"v2-{i:04d}", "v2", i, text)
           for i, text in enumerate(revision.new_texts)]
    paired = align(old, new, old_embeddings, new_embeddings,
                   alpha=alpha, threshold=threshold, paper_mode=True)
    paired = reconcile_split_merge(paired, old, new)
    result = set()
    for pair in paired:
        old_ids = pair.v1_ids or ([pair.v1_id] if pair.v1_id else [])
        new_ids = pair.v2_ids or ([pair.v2_id] if pair.v2_id else [])
        result.update((int(a.split("-")[-1]), int(b.split("-")[-1]))
                      for a in old_ids for b in new_ids)
    return result


def counts(gold: set, predicted: set) -> dict[str, int]:
    return {"tp": len(gold & predicted), "fp": len(predicted - gold),
            "fn": len(gold - predicted)}


def metrics(tallies: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = tallies["tp"], tallies["fp"], tallies["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**tallies, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def evaluate(payload: dict, encoder, alpha: float, threshold: float,
             limit_papers: int | None = None) -> dict:
    cases = revisions(payload, limit_papers)
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    edited_totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_revision = []
    embeddings = {}
    current_paper = None
    for case_number, case in enumerate(cases, start=1):
        if case.paper_id != current_paper:
            embeddings.clear()
            current_paper = case.paper_id
        for version, texts in ((case.old_version, case.old_texts),
                               (case.new_version, case.new_texts)):
            if version not in embeddings:
                embeddings[version] = embed(texts, encoder)
        predicted = {
            "Positional": positional(case),
            "difflib": difflib_links(case),
            "SemanticDiff": semanticdiff_links(
                case, embeddings[case.old_version], embeddings[case.new_version],
                alpha, threshold),
        }
        row = {"paper_id": case.paper_id,
               "versions": f"v{case.old_version}→v{case.new_version}",
               "old_sentences": len(case.old_texts), "new_sentences": len(case.new_texts),
               "gold_links": len(case.gold)}
        old_norm = [normalize_for_match(text) for text in case.old_texts]
        new_norm = [normalize_for_match(text) for text in case.new_texts]
        edited = lambda links: {(i, j) for i, j in links if old_norm[i] != new_norm[j]}
        gold_edited = edited(case.gold)
        row["gold_edited_links"] = len(gold_edited)
        for name, links in predicted.items():
            result = counts(case.gold, links)
            edited_result = counts(gold_edited, edited(links))
            for key, value in result.items():
                totals[name][key] += value
            for key, value in edited_result.items():
                edited_totals[name][key] += value
            row[name] = metrics(result)
            row[f"{name}_edited"] = metrics(edited_result)
        per_revision.append(row)
        if case_number % 20 == 0 or case_number == len(cases):
            print(f"Scored {case_number}/{len(cases)} revisions", flush=True)
    return {"benchmark": "arXivEdits sentence alignment",
            "scope": "Clean LaTeX-derived sentences; headings and dataset-skipped sentences excluded. "
                     "Measures alignment links only, not PDF extraction or meaning-change verdicts.",
            "split": "test", "source_repo": SOURCE_REPO,
            "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
            "paper_groups": len({case.paper_id for case in cases}),
            "revision_pairs": len(cases),
            "old_sentences": sum(len(case.old_texts) for case in cases),
            "new_sentences": sum(len(case.new_texts) for case in cases),
            "gold_links": sum(len(case.gold) for case in cases),
            "gold_edited_links": sum(row["gold_edited_links"] for row in per_revision),
            "alignment": {"alpha": alpha, "threshold": threshold,
                          "embedding_dimension": encoder.get_embedding_dimension()},
            "systems": {name: metrics(tally) for name, tally in totals.items()},
            "edited_systems": {name: metrics(tally) for name, tally in edited_totals.items()},
            "per_revision": per_revision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-papers", type=int, default=None,
                        help="Run a deterministic subset for development; default is the full test split")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.limit_papers is not None and args.limit_papers < 1:
        parser.error("--limit-papers must be positive")
    cfg = load_config()
    encoder = get_encoder(cfg["models"]["embedding"], cfg.get("device", "auto"))
    start = time.perf_counter()
    scorecard = evaluate(load_test(), encoder, cfg["align"]["alpha"],
                         cfg["align"]["threshold"], args.limit_papers)
    scorecard["alignment"]["encoder_name"] = cfg["models"]["embedding"]
    scorecard["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    print(f"arXivEdits test: {scorecard['paper_groups']} paper groups, "
          f"{scorecard['revision_pairs']} revisions, {scorecard['gold_links']} gold links")
    for name, result in scorecard["systems"].items():
        print(f"{name:13} P={result['precision']:.3f} R={result['recall']:.3f} "
              f"F1={result['f1']:.3f} | edited F1={scorecard['edited_systems'][name]['f1']:.3f}")
    print(f"Scorecard: {args.output}")


if __name__ == "__main__":
    main()

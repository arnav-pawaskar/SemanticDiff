"""Provision alignment for general documents and research papers.

1. Provisions whose normalised text is identical are paired first (UNCHANGED).
2. The rest are paired by maximum-weight bipartite matching (Hungarian algorithm,
   ``scipy.optimize.linear_sum_assignment``) on a fused similarity
       fused = alpha * cosine(SBERT) + (1 - alpha) * token_overlap.
   Pairs below ``threshold`` are rejected -> REMOVED / ADDED.
3. General-document moves: sort all matched pairs by V1 position and take the Longest Increasing
   Subsequence of their V2 positions. Those pairs kept their relative order; every
   matched pair *off* the LIS changed relative order and is flagged ``moved``. This is
   the minimal set of moves that explains the reordering (the idea behind patience diff).

Papers use section-local assignment and only mark a unique, distant exact
passage as moved. Nearby exact text remains matched across heading changes.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import replace
from typing import Optional

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from semanticdiff.align.similarity import cosine_matrix, fused_matrix, token_matrix
from semanticdiff.ingest.segment import normalize_for_match
from semanticdiff.models import AlignedPair, PairStatus, Provision, ProvisionKind


def longest_increasing_subsequence(seq: list[int]) -> set[int]:
    """Return the *positions* in ``seq`` that form one longest strictly increasing subsequence."""
    if not seq:
        return set()
    tails: list[int] = []        # tails[k] = smallest tail value of an increasing run of length k+1
    tails_pos: list[int] = []
    prev = [-1] * len(seq)
    for i, v in enumerate(seq):
        k = bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tails_pos.append(i)
        else:
            tails[k] = v
            tails_pos[k] = i
        prev[i] = tails_pos[k - 1] if k > 0 else -1
    out, i = set(), tails_pos[-1]
    while i != -1:
        out.add(i)
        i = prev[i]
    return out


def _paper_align(p1: list[Provision], p2: list[Provision],
                 emb1: Optional[np.ndarray], emb2: Optional[np.ndarray],
                 alpha: float, threshold: float) -> list[AlignedPair]:
    """Order-aware matching for revisions of a paper.

    Match within top-level sections so repeated scientific language cannot
    cross an entire paper. Local paragraph reflow is not called a move. Only
    unique, long exact passages unmatched across sections qualify as moves.
    """
    n, m = len(p1), len(p2)
    if not n or not m:
        return _assemble(p1, p2, {}, set())
    cos = cosine_matrix(emb1, emb2) if emb1 is not None and emb2 is not None else np.zeros((n, m), dtype=np.float32)
    tok = token_matrix([p.text for p in p1], [p.text for p in p2])
    fused = fused_matrix(cos, tok, alpha if emb1 is not None else 0.0)
    norm1 = [normalize_for_match(p.text) for p in p1]
    norm2 = [normalize_for_match(p.text) for p in p2]
    matches: dict[int, tuple[int, float, float, float]] = {}
    sections1: dict[str, list[int]] = {}
    sections2: dict[str, list[int]] = {}
    for i, p in enumerate(p1):
        sections1.setdefault(p.section_path[0] if p.section_path else "", []).append(i)
    for j, p in enumerate(p2):
        sections2.setdefault(p.section_path[0] if p.section_path else "", []).append(j)
    for section, old_ids in sections1.items():
        new_ids = sections2.get(section)
        if not new_ids:
            continue
        score = fused[np.ix_(old_ids, new_ids)].copy()
        for r, i in enumerate(old_ids):
            for c, j in enumerate(new_ids):
                a, b = p1[i], p2[j]
                if (a.kind == ProvisionKind.NON_PROSE) != (b.kind == ProvisionKind.NON_PROSE):
                    score[r, c] = -1.0
                    continue
                if norm1[i] == norm2[j]:
                    score[r, c] = 1.25
                    continue
                score[r, c] -= min(0.15, abs(i / n - j / m) * 0.3)
                score[r, c] += 0.06 if a.section_label == b.section_label else -0.05
                if a.kind == ProvisionKind.NON_PROSE:
                    score[r, c] -= 0.08
        rows, cols = linear_sum_assignment(-score)
        for r, c in zip(rows, cols):
            i, j = old_ids[r], new_ids[c]
            required = max(threshold, 0.72) if p1[i].kind == ProvisionKind.NON_PROSE else threshold
            if score[r, c] >= required:
                matches[i] = (j, float(cos[i, j]), float(tok[i, j]), float(fused[i, j]))

    # Recover unique exact text across section boundaries. PDF heading
    # extraction may label the same nearby paragraph differently; only a
    # distant, long passage is reported as a genuine relocation.
    used2 = {v[0] for v in matches.values()}
    old_unmatched = [i for i in range(n) if i not in matches]
    new_unmatched = [j for j in range(m) if j not in used2]
    old_counts = {key: norm1.count(key) for key in {norm1[i] for i in old_unmatched}}
    new_by_text: dict[str, list[int]] = {}
    for j in new_unmatched:
        new_by_text.setdefault(norm2[j], []).append(j)
    moved_i: set[int] = set()
    for i in old_unmatched:
        key = norm1[i]
        cands = new_by_text.get(key, [])
        if old_counts[key] != 1 or len(cands) != 1:
            continue
        j = cands[0]
        if (p1[i].kind == ProvisionKind.NON_PROSE) != (p2[j].kind == ProvisionKind.NON_PROSE):
            continue
        near = (abs(i / n - j / m) < 0.1 or
                (p1[i].page is not None and p2[j].page is not None and
                 abs(p1[i].page - p2[j].page) <= 1))
        move = (not near and p1[i].kind != ProvisionKind.NON_PROSE and
                len(key.split()) >= 10 and abs(i - j) > 8 and
                p1[i].section_path[:1] != p2[j].section_path[:1])
        if not near and not move:
            continue
        matches[i] = (j, float(cos[i, j]), float(tok[i, j]), 1.0)
        if move:
            moved_i.add(i)
        del new_by_text[key]
    return _assemble(p1, p2, matches, moved_i)


def _assemble(p1: list[Provision], p2: list[Provision],
              matches: dict[int, tuple[int, float, float, float]],
              moved_i: set[int]) -> list[AlignedPair]:
    n, m = len(p1), len(p2)
    pairs: list[tuple[tuple[float, float], AlignedPair]] = []
    matched_j = {v[0] for v in matches.values()}
    for i in sorted(matches):
        j, c, t, f = matches[i]
        same = normalize_for_match(p1[i].text) == normalize_for_match(p2[j].text)
        pairs.append(((j, 0.0), AlignedPair(
            v1_id=p1[i].id, v2_id=p2[j].id,
            status=PairStatus.UNCHANGED if same else PairStatus.MODIFIED,
            moved=i in moved_i, sim_embed=c, sim_token=t, sim_fused=f)))
    for j in range(m):
        if j not in matched_j:
            pairs.append(((j, 0.0), AlignedPair(v1_id=None, v2_id=p2[j].id, status=PairStatus.ADDED)))
    anchor = -1.0
    for i in range(n):
        if i in matches:
            if i not in moved_i:
                anchor = float(matches[i][0])
            continue
        pairs.append(((anchor + 0.5, i / 1000.0), AlignedPair(
            v1_id=p1[i].id, v2_id=None, status=PairStatus.REMOVED)))
    pairs.sort(key=lambda kp: kp[0])
    return [p for _, p in pairs]


def reconcile_split_merge(aligned: list[AlignedPair], p1: list[Provision],
                          p2: list[Provision]) -> list[AlignedPair]:
    """Rejoin one sentence split into two (or two merged into one) after alignment.

    A high text-coverage requirement keeps a genuine new nearby sentence from
    disappearing into its neighbour. Constituent IDs remain on the pair so the
    report can show one combined text diff and account for both provisions.
    """
    old = {p.id: p for p in p1}
    new = {p.id: p for p in p2}
    added = {ap.v2_id: k for k, ap in enumerate(aligned) if ap.status == PairStatus.ADDED}
    removed = {ap.v1_id: k for k, ap in enumerate(aligned) if ap.status == PairStatus.REMOVED}
    consumed: set[int] = set()
    replacements: dict[int, AlignedPair] = {}

    def close(a: Provision, b: Provision) -> bool:
        if a.page is not None and b.page is not None and abs(a.page - b.page) > 1:
            return False
        return not (a.section_path and b.section_path and a.section_path[0] != b.section_path[0])

    def coverage(left: str, right: str) -> float:
        left, right = normalize_for_match(left), normalize_for_match(right)
        if not left or not right or min(len(left), len(right)) / max(len(left), len(right)) < 0.82:
            return 0.0
        return fuzz.ratio(left, right) / 100.0

    for k, ap in enumerate(aligned):
        if ap.status not in (PairStatus.UNCHANGED, PairStatus.MODIFIED) or ap.moved:
            continue
        a, b = old[ap.v1_id], new[ap.v2_id]
        options: list[tuple[float, int, list[str], list[str]]] = []
        for offset in (-1, 1):
            j = b.index + offset
            if 0 <= j < len(p2):
                extra = p2[j]
                row = added.get(extra.id)
                if row is not None and row not in consumed and close(b, extra):
                    ids = sorted([b.id, extra.id], key=lambda ident: new[ident].index)
                    score = coverage(a.text, " ".join(new[ident].text for ident in ids))
                    if score >= 0.88:
                        options.append((score, row, [a.id], ids))
            i = a.index + offset
            if 0 <= i < len(p1):
                extra = p1[i]
                row = removed.get(extra.id)
                if row is not None and row not in consumed and close(a, extra):
                    ids = sorted([a.id, extra.id], key=lambda ident: old[ident].index)
                    score = coverage(" ".join(old[ident].text for ident in ids), b.text)
                    if score >= 0.88:
                        options.append((score, row, ids, [b.id]))
        if not options:
            continue
        score, row, old_ids, new_ids = max(options, key=lambda item: item[0])
        consumed.add(row)
        left = " ".join(old[ident].text for ident in old_ids)
        right = " ".join(new[ident].text for ident in new_ids)
        replacements[k] = replace(
            ap, status=PairStatus.UNCHANGED if normalize_for_match(left) == normalize_for_match(right)
            else PairStatus.MODIFIED, sim_fused=score, v1_ids=old_ids, v2_ids=new_ids)
    return [replacements.get(k, ap) for k, ap in enumerate(aligned) if k not in consumed]


def align(p1: list[Provision], p2: list[Provision],
          emb1: Optional[np.ndarray], emb2: Optional[np.ndarray],
          alpha: float = 0.7, threshold: float = 0.55,
          paper_mode: bool = False) -> list[AlignedPair]:
    if paper_mode:
        return _paper_align(p1, p2, emb1, emb2, alpha, threshold)
    n, m = len(p1), len(p2)
    matches: dict[int, tuple[int, float, float, float]] = {}   # i -> (j, cos, tok, fused)

    # ---- 1. exact matches (in document order, so duplicates pair up sequentially)
    by_text: dict[str, list[int]] = {}
    for j, p in enumerate(p2):
        by_text.setdefault(normalize_for_match(p.text), []).append(j)
    used2: set[int] = set()
    for i, p in enumerate(p1):
        cands = by_text.get(normalize_for_match(p.text))
        if cands:
            j = cands.pop(0)
            matches[i] = (j, 1.0, 1.0, 1.0)
            used2.add(j)

    # ---- 2. Hungarian assignment over the remaining provisions
    rest1 = [i for i in range(n) if i not in matches]
    rest2 = [j for j in range(m) if j not in used2]
    if rest1 and rest2:
        if emb1 is not None and emb2 is not None:
            cos = cosine_matrix(emb1[rest1], emb2[rest2])
        else:
            cos = np.zeros((len(rest1), len(rest2)), dtype=np.float32)
        tok = token_matrix([p1[i].text for i in rest1], [p2[j].text for j in rest2])
        fused = fused_matrix(cos, tok, alpha if emb1 is not None else 0.0)
        rows, cols = linear_sum_assignment(-fused)
        for r, c in zip(rows, cols):
            if fused[r, c] >= threshold:
                matches[rest1[r]] = (rest2[c], float(cos[r, c]), float(tok[r, c]), float(fused[r, c]))

    # ---- 3. move detection via LIS
    matched_i = sorted(matches)
    v2_seq = [matches[i][0] for i in matched_i]
    in_order = longest_increasing_subsequence(v2_seq)
    moved_i = {matched_i[k] for k in range(len(matched_i)) if k not in in_order}
    return _assemble(p1, p2, matches, moved_i)

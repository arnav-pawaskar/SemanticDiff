"""Provision alignment: exact match -> Hungarian assignment -> LIS move detection.

1. Provisions whose normalised text is identical are paired first (UNCHANGED).
2. The rest are paired by maximum-weight bipartite matching (Hungarian algorithm,
   ``scipy.optimize.linear_sum_assignment``) on a fused similarity
       fused = alpha * cosine(SBERT) + (1 - alpha) * token_overlap.
   Pairs below ``threshold`` are rejected -> REMOVED / ADDED.
3. Moves: sort all matched pairs by V1 position and take the Longest Increasing
   Subsequence of their V2 positions. Those pairs kept their relative order; every
   matched pair *off* the LIS changed relative order and is flagged ``moved``. This is
   the minimal set of moves that explains the reordering (the idea behind patience diff).
"""
from __future__ import annotations

from bisect import bisect_left
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from semanticdiff.align.similarity import cosine_matrix, fused_matrix, token_matrix
from semanticdiff.ingest.segment import normalize_for_match
from semanticdiff.models import AlignedPair, PairStatus, Provision


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


def align(p1: list[Provision], p2: list[Provision],
          emb1: Optional[np.ndarray], emb2: Optional[np.ndarray],
          alpha: float = 0.7, threshold: float = 0.55) -> list[AlignedPair]:
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

    pairs: list[tuple[tuple[float, float], AlignedPair]] = []
    matched_j = {v[0] for v in matches.values()}
    for i in matched_i:
        j, c, t, f = matches[i]
        same = normalize_for_match(p1[i].text) == normalize_for_match(p2[j].text)
        pairs.append(((j, 0.0), AlignedPair(
            v1_id=p1[i].id, v2_id=p2[j].id,
            status=PairStatus.UNCHANGED if same else PairStatus.MODIFIED,
            moved=i in moved_i, sim_embed=c, sim_token=t, sim_fused=f)))
    for j in range(m):
        if j not in matched_j:
            pairs.append(((j, 0.0), AlignedPair(v1_id=None, v2_id=p2[j].id, status=PairStatus.ADDED)))

    # Removed provisions are displayed right after the V2 position of their nearest
    # preceding V1 neighbour that survived (in the LIS), so the report reads like a diff.
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

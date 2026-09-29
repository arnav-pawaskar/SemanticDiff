"""Layer 2 — bidirectional Natural Language Inference.

NLI is directional, so every pair is scored twice: v1 => v2 and v2 => v1.
  * entailment both ways          -> the two provisions are (near-)equivalent
  * entailment one way only       -> one is more specific (e.g. scope narrowing)
  * contradiction in either way   -> reversal of meaning
NLI is *evidence*, never the sole arbiter: see changes/verdict.py.
"""
from __future__ import annotations

from semanticdiff.models import NLIResult


def nli_batch(pairs: list[tuple[str, str]], model, label_index: dict[str, int],
              batch_size: int = 32) -> list[NLIResult]:
    if not pairs:
        return []
    forward = [(a, b) for a, b in pairs]
    backward = [(b, a) for a, b in pairs]
    probs = model.predict(forward + backward, batch_size=batch_size, apply_softmax=True,
                          show_progress_bar=False)
    n = len(pairs)
    e, c, u = label_index["entailment"], label_index["contradiction"], label_index["neutral"]
    out = []
    for k in range(n):
        f, b = probs[k], probs[n + k]
        out.append(NLIResult(
            entail_12=round(float(f[e]), 4), contra_12=round(float(f[c]), 4), neutral_12=round(float(f[u]), 4),
            entail_21=round(float(b[e]), 4), contra_21=round(float(b[c]), 4), neutral_21=round(float(b[u]), 4)))
    return out

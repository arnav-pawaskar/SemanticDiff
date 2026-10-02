"""Cross-clause impact analysis over a definition / reference graph.

Nodes are defined terms, sections and provisions. Edges:
    defining provision --defines--> term --referenced by--> provision
    section --contains--> provision  (only for sections referenced as "Section N")
If a term's defining provision changed materially (or was added/removed), every
provision that references the term is flagged AFFECTED — even when its own text is
unchanged — with the reference path as the explanation. Propagation continues through
definitions that reference other definitions, up to ``max_hops``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from semanticdiff.impact.definitions import find_section_refs, find_term_mentions
from semanticdiff.models import Impact, PairAnalysis, PairStatus, Provision, Verdict

_CHANGED_VERDICTS = {Verdict.MATERIAL, Verdict.UNCLASSIFIED}


@dataclass
class _Node:
    pair_index: int
    provision: Provision


def _is_changed(pa: PairAnalysis) -> bool:
    return pa.pair.status in (PairStatus.ADDED, PairStatus.REMOVED) or pa.verdict in _CHANGED_VERDICTS


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _section_label(p: Provision) -> str:
    return p.section_path[-1] if p.section_path else p.id


def build_impact(pairs: list[PairAnalysis], prov1: dict[str, Provision], prov2: dict[str, Provision],
                 frames: dict[str, object], max_hops: int = 2) -> tuple[list[Impact], dict]:
    # current view of every aligned row: prefer the V2 provision, fall back to V1
    rows: list[_Node] = []
    for k, pa in enumerate(pairs):
        p = prov2.get(pa.pair.v2_id) if pa.pair.v2_id else prov1.get(pa.pair.v1_id)
        if p is not None:
            rows.append(_Node(k, p))

    # term -> defining row
    definitions: dict[str, tuple[str, int]] = {}
    for k, pa in enumerate(pairs):
        for pid in (pa.pair.v2_id, pa.pair.v1_id):
            f = frames.get(pid) if pid else None
            if f is not None and getattr(f, "defined_term", None):
                definitions.setdefault(f.defined_term.lower(), (f.defined_term, k))
                break

    # term -> rows that mention it (excluding its definition)
    references: dict[str, list[int]] = {}
    for key, (term, def_k) in definitions.items():
        for node in rows:
            if node.pair_index != def_k and find_term_mentions(node.provision.text, term):
                references.setdefault(key, []).append(node.pair_index)

    # section label -> rows; row -> referenced section numbers
    section_rows: dict[str, list[int]] = {}
    for node in rows:
        for label in node.provision.section_path:
            section_rows.setdefault(label.lower().replace("section ", ""), []).append(node.pair_index)
    section_refs = {node.pair_index: find_section_refs(node.provision.text) for node in rows}

    impacts: list[Impact] = []
    seen: set[tuple[int, int]] = set()
    frontier: list[tuple[int, str, list[str], int, int]] = []   # (changed row, term, path, hops, origin row)
    for key, (term, def_k) in definitions.items():
        if _is_changed(pairs[def_k]):
            frontier.append((def_k, key, [term], 1, def_k))
    for node in rows:                                            # explicit "Section N" references
        for num in section_refs.get(node.pair_index, []):
            for target in section_rows.get(num.lower(), []):
                if target != node.pair_index and _is_changed(pairs[target]):
                    frontier.append((target, f"§{num}", [f"Section {num}"], 1, target))

    while frontier:
        src, key, path, hops, origin = frontier.pop(0)
        targets = references.get(key, []) if not key.startswith("§") else [
            n.pair_index for n in rows if key[1:] in section_refs.get(n.pair_index, [])]
        for tgt in targets:
            if (origin, tgt) in seen or tgt == origin:
                continue
            seen.add((origin, tgt))
            pa = pairs[tgt]
            p = prov2.get(pa.pair.v2_id) if pa.pair.v2_id else prov1.get(pa.pair.v1_id)
            label = path[-1]
            reason = (f"References “{label}”, whose definition changed."
                      if not label.startswith("Section") else f"Refers to {label}, which changed.")
            if hops > 1:
                reason = f"Indirect ({hops} hops): " + " → ".join(path) + " → this clause."
            imp = Impact(id=f"i{len(impacts) + 1}", source_pair_index=origin, term=label,
                         affected_provision_id=p.id, affected_pair_index=tgt,
                         path=path + [_section_label(p)], reason=reason, hops=hops)
            impacts.append(imp)
            pa.affected_by.append(imp.id)
            # the affected clause is itself a definition -> propagate further
            f = frames.get(p.id)
            if hops < max_hops and f is not None and getattr(f, "defined_term", None):
                frontier.append((tgt, f.defined_term.lower(), path + [f.defined_term], hops + 1, origin))

    graph = _graph(pairs, rows, definitions, references, impacts)
    return impacts, graph


def _graph(pairs, rows, definitions, references, impacts) -> dict:
    by_index = {n.pair_index: n.provision for n in rows}
    nodes, edges = {}, []
    affected = {i.affected_pair_index for i in impacts}
    for key, (term, def_k) in definitions.items():
        changed = _is_changed(pairs[def_k])
        tid = f"term:{key}"
        nodes[tid] = {"id": tid, "label": term, "type": "term", "changed": changed}
        dp = by_index.get(def_k)
        if dp is not None:
            pid = f"row:{def_k}"
            nodes[pid] = {"id": pid, "label": f"{_section_label(dp)}: {_short(dp.text, 48)}",
                          "type": "definition", "changed": changed}
            edges.append({"source": pid, "target": tid, "label": "defines"})
        for r in references.get(key, []):
            p = by_index.get(r)
            if p is None:
                continue
            rid = f"row:{r}"
            nodes.setdefault(rid, {"id": rid, "label": f"{_section_label(p)}: {_short(p.text, 48)}",
                                   "type": "clause", "changed": _is_changed(pairs[r]),
                                   "affected": r in affected})
            edges.append({"source": tid, "target": rid, "label": "referenced by"})
    return {"nodes": list(nodes.values()), "edges": edges}


def to_dot(graph: dict, only_changed: bool = False) -> str:
    """Render the impact graph as Graphviz DOT (for st.graphviz_chart)."""
    keep_terms = {n["id"] for n in graph["nodes"] if n["type"] == "term" and (n["changed"] or not only_changed)}
    # colours from the app theme: ink = changed definition, peach = affected clause, white = context
    lines = ["digraph G {", '  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.9;',
             '  node [shape=box, style="rounded,filled", fontname="Inter", fontsize=10, color="#dcdcdf", '
             'fontcolor="#17191c", fillcolor="#ffffff", penwidth=1, margin="0.2,0.09"];',
             '  edge [color="#a3a6af", fontcolor="#6b6f7a", fontsize=8, fontname="Inter", arrowsize=0.6];']
    used = set()
    for e in graph["edges"]:
        term = e["target"] if e["target"].startswith("term:") else e["source"]
        if term not in keep_terms:
            continue
        used.update((e["source"], e["target"]))
        lines.append(f'  "{e["source"]}" -> "{e["target"]}" [label="{e["label"]}"];')
    for n in graph["nodes"]:
        if n["id"] not in used:
            continue
        if n["type"] == "term":
            attrs = ('shape=ellipse, fillcolor="#17191c", color="#17191c", fontcolor="#ffffff"' if n["changed"]
                     else 'shape=ellipse, fillcolor="#e6e6e9", color="#e6e6e9"')
        elif n.get("changed"):
            attrs = 'color="#17191c", penwidth=1.2'
        elif n.get("affected"):
            attrs = 'fillcolor="#fbe1d1", color="#fbe1d1", fontcolor="#5d2a1a"'
        else:
            attrs = 'fillcolor="#ffffff"'
        label = n["label"].replace('"', "'")
        lines.append(f'  "{n["id"]}" [label="{label}", {attrs}];')
    lines.append("}")
    return "\n".join(lines)

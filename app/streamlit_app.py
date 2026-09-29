"""SemanticDiff — Streamlit demo.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import CSS, e, frame_table, row_html
from semanticdiff.impact.graph import to_dot
from semanticdiff.models import ChangeCategory, PairStatus, Severity, Verdict
from semanticdiff.report.serialize import report_to_json

DEMO = ROOT / "data" / "demo"
DEMOS = {
    "University academic regulations (2025 → 2026)": ("academic_regulations_v1.txt", "academic_regulations_v2.txt"),
    "Cloud service level agreement": ("sla_v1.txt", "sla_v2.txt"),
}

st.set_page_config(page_title="SemanticDiff", page_icon="≠", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading spaCy, sentence-transformer and NLI models…")
def get_engine():
    from semanticdiff.pipeline import SemanticDiff
    return SemanticDiff()


@st.cache_data(show_spinner="Comparing documents…", max_entries=16)
def run_compare(b1: bytes, n1: str, b2: bytes, n2: str):
    return get_engine().compare(b1, b2, n1, n2, filename1=n1, filename2=n2)


# ------------------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.markdown("## ≠ SemanticDiff")
    st.caption("Meaning-aware document version comparison")
    source = st.radio("Documents", ["Demo pair", "Upload"], horizontal=True)
    if source == "Demo pair":
        demo = st.selectbox("Demo", list(DEMOS))
        f1, f2 = DEMOS[demo]
        b1, n1 = (DEMO / f1).read_bytes(), f1
        b2, n2 = (DEMO / f2).read_bytes(), f2
    else:
        u1 = st.file_uploader("Version 1 (PDF / TXT)", type=["pdf", "txt", "md"])
        u2 = st.file_uploader("Version 2 (PDF / TXT)", type=["pdf", "txt", "md"])
        if not (u1 and u2):
            st.info("Upload both versions to compare them.")
            st.stop()
        b1, n1, b2, n2 = u1.getvalue(), u1.name, u2.getvalue(), u2.name
    st.divider()
    show_unchanged = st.toggle("Show unchanged clauses", value=False)

report = run_compare(b1, n1, b2, n2)
s = report.summary
impacts_by_id = {i.id: i for i in report.impacts}

# ------------------------------------------------------------------------- hero + dashboard
st.markdown(f"### {e(report.v1_name)} → {e(report.v2_name)}")
sim = report.doc_similarity
n_material = s["material"] + s["added"] + s["removed"] + s["unclassified"]
st.markdown(
    f'<div class="sd-hero">The two versions are <b>{sim.get("embedding_cosine", 0):.0%} similar</b> by sentence '
    f'embeddings and <b>{sim.get("tfidf_cosine", 0):.0%}</b> by TF-IDF — yet SemanticDiff finds '
    f'<b>{s["severity"]["HIGH"]} high-impact semantic changes</b> across <b>{n_material}</b> clauses, '
    f'plus <b>{s["affected_by_dependency"]}</b> unchanged clauses whose meaning shifted through a changed definition.'
    f'</div>', unsafe_allow_html=True)

metrics = [
    ("Clauses (v1 / v2)", f'{s["clauses_v1"]} / {s["clauses_v2"]}'), ("Added", s["added"]),
    ("Removed", s["removed"]), ("Moved", s["moved"]),
    ("Modified", s["modified"]), ("Meaning-preserving", s["meaning_preserving"]),
    ("Material changes", s["material"] + s["unclassified"]), ("Affected via definitions", s["affected_by_dependency"]),
    ("HIGH impact", s["severity"]["HIGH"]), ("MEDIUM impact", s["severity"]["MEDIUM"]),
    ("LOW impact", s["severity"]["LOW"]),
    ("Top category", max(s["categories"], key=s["categories"].get).title() if s["categories"] else "—"),
]
for row in range(0, len(metrics), 4):
    for col, (label, value) in zip(st.columns(4), metrics[row:row + 4]):
        col.metric(label, value)

tab_diff, tab_quad, tab_graph, tab_lab, tab_about = st.tabs(
    ["Semantic diff", "Text vs meaning", "Impact graph", "Sentence lab", "How it works"])

# ------------------------------------------------------------------------- semantic diff
FILTERS = {
    "All": lambda pa: True,
    "High impact": lambda pa: any(ch.severity == Severity.HIGH for ch in pa.changes),
    "Added": lambda pa: pa.pair.status == PairStatus.ADDED,
    "Removed": lambda pa: pa.pair.status == PairStatus.REMOVED,
    "Modified": lambda pa: pa.pair.status == PairStatus.MODIFIED,
    "Moved": lambda pa: pa.pair.moved,
    "Meaning-preserving": lambda pa: pa.verdict == Verdict.MEANING_PRESERVING,
    "Affected by dependency": lambda pa: bool(pa.affected_by),
}
_CAT_FILTERS = {"Numbers": ChangeCategory.NUMERIC, "Dates & durations": ChangeCategory.TEMPORAL,
                "Entities": ChangeCategory.ENTITY, "Modality": ChangeCategory.MODALITY,
                "Negation": ChangeCategory.NEGATION, "Scope": ChangeCategory.SCOPE,
                "Conditions": ChangeCategory.CONDITION, "Reversals": ChangeCategory.REVERSAL}
for label, cat in _CAT_FILTERS.items():
    FILTERS[label] = (lambda c: (lambda pa: any(ch.category == c for ch in pa.changes)))(cat)

with tab_diff:
    choice = st.segmented_control("Filter", list(FILTERS), default="All", label_visibility="collapsed") or "All"
    rows = [(k, pa) for k, pa in enumerate(report.pairs) if FILTERS[choice](pa)]
    if not show_unchanged and choice == "All":
        rows = [(k, pa) for k, pa in rows
                if pa.pair.status != PairStatus.UNCHANGED or pa.pair.moved or pa.affected_by]
    st.caption(f"{len(rows)} clause(s)")
    for k, pa in rows:
        st.markdown(row_html(pa, impacts_by_id), unsafe_allow_html=True)
        if pa.pair.status == PairStatus.MODIFIED:
            with st.expander("Explain this result", expanded=False):
                a, b = st.columns(2)
                a.markdown("**V1 frame** (dependency-parse extraction)")
                a.dataframe(pd.DataFrame(frame_table(pa.frames.get("v1"))), hide_index=True, width="stretch")
                b.markdown("**V2 frame**")
                b.dataframe(pd.DataFrame(frame_table(pa.frames.get("v2"))), hide_index=True, width="stretch")
                if pa.nli:
                    st.markdown(
                        f"**NLI** (DeBERTa-v3 cross-encoder) — V1⇒V2: entail {pa.nli.entail_12:.2f}, "
                        f"contradict {pa.nli.contra_12:.2f} · V2⇒V1: entail {pa.nli.entail_21:.2f}, "
                        f"contradict {pa.nli.contra_21:.2f} · embedding cosine {pa.pair.sim_embed:.3f} · "
                        f"textual change {pa.textual_change:.2f}")
                res = pa.frames.get("residual", {})
                st.markdown(
                    f"**Decision rule fired:** `{pa.verdict_rule}` · lexical diff explained by rules: "
                    f"{pa.frames.get('explained_ratio', 1.0):.0%}"
                    + (f" · unexplained words: `{' '.join(res.get('old', []))}` → `{' '.join(res.get('new', []))}`"
                       if res.get("old") or res.get("new") else ""))

# ------------------------------------------------------------------------- text vs meaning quadrant
with tab_quad:
    st.markdown("Each point is a modified clause. **Top-left** = small textual edit, large change in meaning "
                "(what line diffs under-report). **Bottom-right** = heavy rewrite, same meaning "
                "(what line diffs over-report).")
    pts = [pa for pa in report.pairs if pa.pair.status == PairStatus.MODIFIED]
    rnd = random.Random(0)
    level_y = {Severity.LOW: 0.0, Severity.MEDIUM: 1.0, Severity.HIGH: 2.0}
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=0.25, y0=1.5, y1=2.5, fillcolor="rgba(207,34,46,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=0.35, x1=1.0, y0=-0.5, y1=0.5, fillcolor="rgba(42,120,214,0.08)", line_width=0)
    for verdict, color, name in ((Verdict.MEANING_PRESERVING, "#2a78d6", "Meaning-preserving"),
                                 (Verdict.MATERIAL, "#eb6834", "Material change"),
                                 (Verdict.UNCLASSIFIED, "#eda100", "Unclassified")):
        g = [pa for pa in pts if pa.verdict == verdict]
        if not g:
            continue
        fig.add_trace(go.Scatter(
            x=[pa.textual_change for pa in g],
            y=[level_y[pa.semantic_level] + rnd.uniform(-0.18, 0.18) for pa in g],
            mode="markers", name=name,
            marker=dict(size=12, color=color, line=dict(width=2, color="rgba(255,255,255,0.9)")),
            customdata=[[pa.v1_text, pa.v2_text, "; ".join(ch.description for ch in pa.changes) or "—"] for pa in g],
            hovertemplate="<b>OLD</b> %{customdata[0]}<br><b>NEW</b> %{customdata[1]}<br>"
                          "<b>Detected</b> %{customdata[2]}<br>textual change %{x:.2f}<extra></extra>"))
    fig.add_annotation(x=0.02, y=2.42, text="small edit, big meaning change", showarrow=False, xanchor="left",
                       font=dict(size=11, color="#cf222e"))
    fig.add_annotation(x=0.98, y=-0.42, text="big rewrite, same meaning", showarrow=False, xanchor="right",
                       font=dict(size=11, color="#2a78d6"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(title="Textual change (word-level edit ratio)", range=[0, 1], gridcolor="rgba(128,128,128,0.2)"),
                      yaxis=dict(title="Semantic change (SemanticDiff)", tickvals=[0, 1, 2],
                                 ticktext=["LOW", "MEDIUM", "HIGH"], range=[-0.5, 2.5], gridcolor="rgba(128,128,128,0.2)"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame([{"old": pa.v1_text, "new": pa.v2_text, "textual change": round(pa.textual_change, 2),
                                "semantic level": pa.semantic_level.value, "verdict": pa.verdict.value}
                               for pa in pts]), hide_index=True, width="stretch")

# ------------------------------------------------------------------------- impact graph
with tab_graph:
    if not report.graph.get("nodes"):
        st.info("No defined terms found in these documents.")
    else:
        only_changed = st.toggle("Only changed definitions", value=True)
        st.markdown("Red ellipse = defined term whose definition changed · pink = changed clause · "
                    "amber = clause whose **own text is unchanged** but whose meaning depends on a changed definition.")
        st.graphviz_chart(to_dot(report.graph, only_changed=only_changed), width="stretch")
        if report.impacts:
            st.dataframe(pd.DataFrame([{
                "affected clause": report.pairs[i.affected_pair_index].v2_text or report.pairs[i.affected_pair_index].v1_text,
                "via": i.term, "path": " → ".join(i.path), "reason": i.reason,
                "changed definition": report.pairs[i.source_pair_index].v2_text} for i in report.impacts]),
                hide_index=True, width="stretch")

# ------------------------------------------------------------------------- sentence lab
with tab_lab:
    st.markdown("Compare two single provisions and see every stage of the analysis.")
    l1, l2 = st.columns(2)
    v1 = l1.text_area("Old", "All students may submit applications within 30 days.", height=90)
    v2 = l2.text_area("New", "Final-year students must submit applications within 15 days with faculty approval.",
                      height=90)
    if v1.strip() and v2.strip():
        pa = get_engine().analyze_pair(v1.strip(), v2.strip())
        pa.v1_section = pa.v2_section = ""
        st.markdown(row_html(pa, {}), unsafe_allow_html=True)
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame(frame_table(pa.frames.get("v1"))), hide_index=True, width="stretch")
        b.dataframe(pd.DataFrame(frame_table(pa.frames.get("v2"))), hide_index=True, width="stretch")
        if pa.nli:
            st.caption(f"NLI entail {pa.nli.entail_12:.2f}/{pa.nli.entail_21:.2f} · contradict "
                       f"{pa.nli.contra_12:.2f}/{pa.nli.contra_21:.2f} · cosine {pa.pair.sim_embed:.3f} · "
                       f"rule {pa.verdict_rule}")

# ------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Pipeline.** text extraction (PyMuPDF) → structure (sections, list items) → provision segmentation (spaCy) →
alignment (exact match → Hungarian assignment on SBERT cosine + token overlap → Longest-Increasing-Subsequence
move detection) → per-pair analysis → atomic changes → decision table → severity → cross-clause impact graph.

**Signals per modified pair.**
*Lexical*: token diff and edit ratio. *Deterministic linguistic*: deontic state (modal class × negation from the
dependency parse), quantities with unit normalisation (`500 ms ≡ half a second`), comparators, dates, conditions
(attachment-aware), scope of agent/patient with voice normalisation, entities, antonyms.
*Semantic*: bidirectional NLI (DeBERTa-v3) and embedding cosine.

**Span ownership & grounding.** Each atomic change claims the diff tokens it explains; a token cannot produce two
changes, and every change must be anchored in a token that actually differs.

**Decision table.** V1 no rule fired + mutual entailment → preserving · V2 no rule + contradiction → reversal
(NLI-only) · V3 no content change → preserving · V4 no rule, high similarity → preserving (low confidence) ·
V5 otherwise → unclassified · V6 any rule fired → material (flagged if NLI disagrees).

**Severity** is a configurable heuristic (`config/severity.yaml`), not ground truth; each change shows the rule id
that set it.
""")
    st.download_button("Download full report (JSON)", report_to_json(report), file_name="semanticdiff_report.json",
                       mime="application/json")
    st.caption(f"Models: {report.meta['models']} · device {report.meta['device']} · "
               f"config {report.meta['config_hash']} · timings {report.meta['timings_s']}")

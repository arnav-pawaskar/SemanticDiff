"""SemanticDiff: Streamlit demo.

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

from app.components import (CSS, brand_html, card_html, decision_html, empty_html, evidence_html, frames_html,
                            hero_html, overview_html, pipeline_html)
from semanticdiff.impact.graph import to_dot
from semanticdiff.models import ChangeCategory, PairStatus, Severity, Verdict
from semanticdiff.report.serialize import report_to_json

DEMO = ROOT / "data" / "demo"
DEMOS = {
    "Academic regulations": ("University academic regulations, 2025 to 2026",
                             "academic_regulations_v1.txt", "academic_regulations_v2.txt"),
    "Service level agreement": ("Cloud service level agreement, revised", "sla_v1.txt", "sla_v2.txt"),
}
INK, SIENNA, PEACH, SLATE, ASH, HAIRLINE = "#17191c", "#5d2a1a", "#fbe1d1", "#6b6f7a", "#979799", "#ececec"

st.set_page_config(page_title="SemanticDiff", page_icon=":material/difference:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


def section(title: str, lead: str) -> None:
    st.markdown(f'<div class="sd-h2">{title}</div><p class="sd-lead">{lead}</p>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the spaCy, sentence-transformer and NLI models")
def get_engine():
    from semanticdiff.pipeline import SemanticDiff
    return SemanticDiff()


@st.cache_data(show_spinner="Comparing the two versions", max_entries=16)
def run_compare(b1: bytes, n1: str, b2: bytes, n2: str):
    return get_engine().compare(b1, b2, n1, n2, filename1=n1, filename2=n2)


# ------------------------------------------------------------------------------ sidebar
u1 = u2 = None
with st.sidebar:
    st.markdown(brand_html(), unsafe_allow_html=True)
    st.space("medium")
    source = st.segmented_control("Documents", ["Demo pair", "Upload"], default="Demo pair",
                                  selection_mode="single", width="stretch") or "Demo pair"
    if source == "Demo pair":
        demo = st.selectbox("Demo", list(DEMOS))
        title, f1, f2 = DEMOS[demo]
        b1, n1 = (DEMO / f1).read_bytes(), f1
        b2, n2 = (DEMO / f2).read_bytes(), f2
    else:
        u1 = st.file_uploader("Version 1", type=["pdf", "txt", "md"], help="Text-layer PDF, TXT or Markdown")
        u2 = st.file_uploader("Version 2", type=["pdf", "txt", "md"], help="Text-layer PDF, TXT or Markdown")
        if u1 and u2:
            title = f"{u1.name} compared with {u2.name}"
            b1, n1, b2, n2 = u1.getvalue(), u1.name, u2.getvalue(), u2.name
    st.space("small")
    show_unchanged = st.toggle("Show unchanged clauses", value=False)

if source == "Upload" and not (u1 and u2):
    st.markdown(empty_html(), unsafe_allow_html=True)
    st.stop()

report = run_compare(b1, n1, b2, n2)
s = report.summary
impacts_by_id = {i.id: i for i in report.impacts}

with st.sidebar:
    st.space("medium")
    st.download_button("Download report", report_to_json(report), file_name="semanticdiff_report.json",
                       mime="application/json", icon=":material/arrow_downward:", type="primary", width="stretch")
    st.caption(f"Analysed on {report.meta['device'].upper()} in {report.meta['timings_s']['total']:.2f} s")

# ------------------------------------------------------------------------------ hero + overview
st.markdown(hero_html(title, f"{n1}  →  {n2}", s, report.doc_similarity), unsafe_allow_html=True)
st.markdown(overview_html(s), unsafe_allow_html=True)

tab_diff, tab_quad, tab_graph, tab_lab, tab_about = st.tabs([
    "Clause diff", "Text vs meaning", "Impact graph", "Sentence lab", "How it works"])

# ------------------------------------------------------------------------------ clause diff
VIEWS = {
    "All": lambda pa: True,
    "High impact": lambda pa: any(ch.severity == Severity.HIGH for ch in pa.changes),
    "Changed meaning": lambda pa: pa.verdict in (Verdict.MATERIAL, Verdict.UNCLASSIFIED),
    "Same meaning": lambda pa: pa.verdict == Verdict.MEANING_PRESERVING,
    "Added": lambda pa: pa.pair.status == PairStatus.ADDED,
    "Removed": lambda pa: pa.pair.status == PairStatus.REMOVED,
    "Moved": lambda pa: pa.pair.moved,
    "Via definitions": lambda pa: bool(pa.affected_by),
}
TYPES = {"Numbers": ChangeCategory.NUMERIC, "Dates and durations": ChangeCategory.TEMPORAL,
         "Entities": ChangeCategory.ENTITY, "Modality": ChangeCategory.MODALITY,
         "Negation": ChangeCategory.NEGATION, "Scope": ChangeCategory.SCOPE,
         "Conditions": ChangeCategory.CONDITION, "Reversals": ChangeCategory.REVERSAL}

with tab_diff:
    st.space("small")
    section("Clause by clause", "Each card pairs a clause from version 1 with its match in version 2. Inserted "
            "words are highlighted, removed words are struck through, and every finding names the rule behind it.")
    view = st.pills("Show", list(VIEWS), default="All", selection_mode="single") or "All"
    present = {ch.category for pa in report.pairs for ch in pa.changes}
    type_opts = [k for k, c in TYPES.items() if c in present]
    kind = st.pills("Change type", type_opts, selection_mode="single") if type_opts else None

    rows = [(k, pa) for k, pa in enumerate(report.pairs) if VIEWS[view](pa)]
    if kind:
        rows = [(k, pa) for k, pa in rows if any(ch.category == TYPES[kind] for ch in pa.changes)]
    if not show_unchanged and view == "All" and not kind:
        rows = [(k, pa) for k, pa in rows
                if pa.pair.status != PairStatus.UNCHANGED or pa.pair.moved or pa.affected_by]
    st.caption(f"Showing {len(rows)} of {len(report.pairs)} clauses")
    if not rows:
        st.markdown('<p class="sd-lead">No clauses match this filter.</p>', unsafe_allow_html=True)
    for k, pa in rows:
        with st.container(key=f"sdrow_{k}"):
            st.markdown(card_html(pa, impacts_by_id), unsafe_allow_html=True)
            if pa.pair.status == PairStatus.MODIFIED:
                with st.expander("Explain this result", icon=":material/manage_search:"):
                    st.markdown(frames_html(pa) + evidence_html(pa), unsafe_allow_html=True)

# ------------------------------------------------------------------------------ text vs meaning
with tab_quad:
    st.space("small")
    section("Text change is not meaning change",
            "Each point is an edited clause. Upper left: small edits that changed the meaning, which line diffs "
            "under-report. Lower right: heavy rewrites that kept it, which line diffs over-report.")
    pts = [pa for pa in report.pairs if pa.pair.status == PairStatus.MODIFIED]
    rnd = random.Random(0)
    level_y = {Severity.LOW: 0.0, Severity.MEDIUM: 1.0, Severity.HIGH: 2.0}
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=0.25, y0=1.5, y1=2.5, fillcolor="rgba(251,225,209,0.75)", line_width=0,
                  layer="below")
    fig.add_shape(type="rect", x0=0.35, x1=1.0, y0=-0.5, y1=0.5, fillcolor="rgba(242,242,243,0.9)", line_width=0,
                  layer="below")
    styles = {Verdict.MATERIAL: ("Meaning changed", dict(color=INK, size=13, line=dict(width=2, color="#ffffff"))),
              Verdict.MEANING_PRESERVING: ("Same meaning", dict(color="#ffffff", size=13, line=dict(width=2, color=INK))),
              Verdict.UNCLASSIFIED: ("Unclassified", dict(color=ASH, size=13, line=dict(width=2, color="#ffffff")))}
    for verdict, (name, marker) in styles.items():
        g = [pa for pa in pts if pa.verdict == verdict]
        if not g:
            continue
        fig.add_trace(go.Scatter(
            x=[pa.textual_change for pa in g],
            y=[level_y[pa.semantic_level] + rnd.uniform(-0.16, 0.16) for pa in g],
            mode="markers", name=name, marker=marker,
            customdata=[[pa.v1_text, pa.v2_text, "; ".join(ch.description for ch in pa.changes) or "No change found"]
                        for pa in g],
            hovertemplate="<b>Version 1</b> %{customdata[0]}<br><b>Version 2</b> %{customdata[1]}<br>"
                          "<b>Found</b> %{customdata[2]}<br>Text change %{x:.2f}<extra></extra>"))
    fig.add_annotation(x=0.012, y=2.42, text="Small edit, meaning changed", showarrow=False, xanchor="left",
                       font=dict(size=12.5, color=SIENNA))
    fig.add_annotation(x=0.988, y=-0.42, text="Heavy rewrite, same meaning", showarrow=False, xanchor="right",
                       font=dict(size=12.5, color=SLATE))
    axis = dict(gridcolor=HAIRLINE, zeroline=False, showline=False, tickfont=dict(color=SLATE, size=12))
    fig.update_layout(
        height=470, margin=dict(l=8, r=8, t=8, b=8), font=dict(family="Inter, sans-serif", color=SLATE, size=13),
        legend=dict(orientation="h", y=-0.16, font=dict(color=INK, size=13)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=HAIRLINE, font=dict(family="Inter", color=INK, size=13)),
        xaxis=dict(title=dict(text="Text change (word-level edit ratio)", font=dict(size=13)), range=[0, 1], **axis),
        yaxis=dict(title=dict(text="Meaning change", font=dict(size=13)), tickvals=[0, 1, 2],
                   ticktext=["Low", "Medium", "High"], range=[-0.5, 2.5], **axis),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    with st.container(key="sdrow_plot"):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with st.expander("Data table", icon=":material/table_rows:"):
        st.dataframe(pd.DataFrame([{"version 1": pa.v1_text, "version 2": pa.v2_text,
                                    "text change": round(pa.textual_change, 2),
                                    "meaning change": pa.semantic_level.value.lower(),
                                    "verdict": pa.verdict.value.lower().replace("_", " ")} for pa in pts]),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------------------ impact graph
with tab_graph:
    st.space("small")
    if not report.graph.get("nodes"):
        section("Impact graph", "Neither version defines any terms (\"X means ...\"), so there is no dependency "
                                "graph to show.")
    else:
        section("When a definition moves, its clauses move with it",
                "A clause can change meaning without a single edited word, because a term it uses was redefined. "
                "Dark terms have a changed definition; peach clauses are affected without being edited.")
        only_changed = st.toggle("Only changed definitions", value=True)
        with st.container(key="sdrow_graph"):
            st.graphviz_chart(to_dot(report.graph, only_changed=only_changed), width="stretch")
        if report.impacts:
            st.dataframe(pd.DataFrame([{
                "affected clause": report.pairs[i.affected_pair_index].v2_text or report.pairs[i.affected_pair_index].v1_text,
                "via term": i.term, "path": " → ".join(i.path),
                "changed definition": report.pairs[i.source_pair_index].v2_text} for i in report.impacts]),
                hide_index=True, width="stretch")

# ------------------------------------------------------------------------------ sentence lab
with tab_lab:
    st.space("small")
    section("Try any two provisions", "Type two versions of a clause to see the frame the extractors build, the "
                                      "atomic changes, and the NLI evidence.")
    l1, l2 = st.columns(2)
    v1 = l1.text_area("Version 1", "All students may submit applications within 30 days.", height=96)
    v2 = l2.text_area("Version 2", "Final-year students must submit applications within 15 days with faculty "
                                   "approval.", height=96)
    if v1.strip() and v2.strip():
        pa = get_engine().analyze_pair(v1.strip(), v2.strip())
        pa.v1_section = pa.v2_section = ""
        with st.container(key="sdrow_lab"):
            st.markdown(card_html(pa, {}), unsafe_allow_html=True)
            st.markdown(frames_html(pa) + evidence_html(pa), unsafe_allow_html=True)

# ------------------------------------------------------------------------------ how it works
with tab_about:
    st.space("small")
    a, b = st.columns([1, 1], gap="large")
    with a:
        section("How a result is reached", "Every verdict can be traced to a rule, a parse, or a model score.")
        st.markdown(
            "Both documents are split into provisions and aligned by meaning, so moved and reworded clauses are "
            "matched instead of being reported as a deletion plus an addition.\n\n"
            "Each edited pair is parsed into a structured frame: obligation and negation, who the rule applies "
            "to, conditions, quantities in base units, dates and named entities. Comparing the two frames gives "
            "the atomic changes. Every changed word can be claimed by only one change, and a change must rest on "
            "a word that actually differs.\n\n"
            "NLI, run in both directions, decides only when no rule fired. Severity comes from editable rules in "
            "`config/severity.yaml`, and each finding names the rule that set it.")
    with b:
        st.space("small")
        st.markdown(pipeline_html(report.meta["models"]), unsafe_allow_html=True)
    st.space("medium")
    section("Decision table", "How rule-based changes and NLI evidence combine into a verdict for an edited clause.")
    st.markdown(decision_html(), unsafe_allow_html=True)

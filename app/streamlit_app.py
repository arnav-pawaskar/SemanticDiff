"""SemanticDiff: Streamlit demo.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import hashlib
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
from app.review_workflow import (document_fingerprint, import_decisions, pair_key, review_candidates,
                                 reviewed_report_json, revision_brief)
from app.source_preview import render_source_page
from semanticdiff.impact.graph import to_dot
from semanticdiff.models import ChangeCategory, PairStatus, Severity, Verdict
from semanticdiff.report.serialize import report_to_json

DEMO = ROOT / "data" / "demo"
DEMOS = {
    "Academic regulations": ("University academic regulations, 2025 to 2026",
                             "academic_regulations_v1.txt", "academic_regulations_v2.txt"),
    "Service level agreement": ("Cloud service level agreement, revised", "sla_v1.txt", "sla_v2.txt"),
}
PAPER_CACHE = ROOT / "results" / "cache" / "papers"
if (PAPER_CACHE / "attention_v1.pdf").exists() and (PAPER_CACHE / "attention_v2.pdf").exists():
    DEMOS["Research paper · Attention Is All You Need"] = (
        "Attention Is All You Need, arXiv version 1 to version 2",
        PAPER_CACHE / "attention_v1.pdf", PAPER_CACHE / "attention_v2.pdf")
if (PAPER_CACHE / "bert_v1.pdf").exists() and (PAPER_CACHE / "bert_v2.pdf").exists():
    DEMOS["Research paper · BERT"] = (
        "BERT, arXiv version 1 to version 2",
        PAPER_CACHE / "bert_v1.pdf", PAPER_CACHE / "bert_v2.pdf")
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


@st.cache_data(max_entries=16)
def source_png(pdf: bytes, page: int, excerpt: str | None, bbox):
    return render_source_page(pdf, page, excerpt, bbox)


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
        p1, p2 = (Path(f1) if isinstance(f1, Path) else DEMO / f1,
                  Path(f2) if isinstance(f2, Path) else DEMO / f2)
        b1, n1 = p1.read_bytes(), p1.name
        b2, n2 = p2.read_bytes(), p2.name
        if demo.startswith("Research paper"):
            arxiv_id = "1810.04805" if demo.endswith("BERT") else "1706.03762"
            st.caption(f"Public arXiv PDFs: [v1](https://arxiv.org/abs/{arxiv_id}v1) · "
                       f"[v2](https://arxiv.org/abs/{arxiv_id}v2)")
    else:
        u1 = st.file_uploader("Version 1", type=["pdf", "txt", "md"], help="PDF, including scanned pages, TXT or Markdown")
        u2 = st.file_uploader("Version 2", type=["pdf", "txt", "md"], help="PDF, including scanned pages, TXT or Markdown")
        if u1 and u2:
            title = f"{u1.name} compared with {u2.name}"
            b1, n1, b2, n2 = u1.getvalue(), u1.name, u2.getvalue(), u2.name
    st.space("small")
    show_unchanged = st.toggle("Show unchanged clauses", value=False)

if source == "Upload" and not (u1 and u2):
    st.markdown(empty_html(), unsafe_allow_html=True)
    st.stop()

try:
    report = run_compare(b1, n1, b2, n2)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
s = report.summary
impacts_by_id = {i.id: i for i in report.impacts}
fingerprint = document_fingerprint(b1, b2)
if "reviews_by_document" not in st.session_state:
    st.session_state.reviews_by_document = {}
doc_reviews = st.session_state.reviews_by_document.setdefault(fingerprint, {})


@st.dialog("Source evidence", width="large")
def show_source(pair_index: int) -> None:
    pair = report.pairs[pair_index]
    st.caption(f"Finding {pair_index + 1} · {pair.v2_section or pair.v1_section or 'Document body'}")
    left, right = st.columns(2, gap="medium")
    for column, version, data, filename, page, bbox, excerpt in (
        (left, "Version 1", b1, n1, pair.v1_page, pair.v1_bbox, pair.v1_text),
        (right, "Version 2", b2, n2, pair.v2_page, pair.v2_bbox, pair.v2_text),
    ):
        with column:
            st.markdown(f"**{version} · {filename}**")
            if excerpt is None:
                st.caption("No passage in this version.")
            elif data[:4] == b"%PDF" and page:
                full, focused, location = source_png(data, page, excerpt, bbox)
                st.caption(f"PDF page {page} · {location}")
                if focused:
                    st.image(focused, width="stretch", alt=f"Highlighted source region on page {page}")
                with st.expander("Show full page"):
                    st.image(full, width="stretch", alt=f"Full PDF page {page} with source region marked")
            else:
                st.caption("Source page preview is available for PDFs; extracted text is shown below.")
            if excerpt:
                st.caption(excerpt[:550] + ("…" if len(excerpt) > 550 else ""))

with st.sidebar:
    st.space("medium")
    st.download_button("Download report", report_to_json(report), file_name="semanticdiff_report.json",
                       mime="application/json", icon=":material/arrow_downward:", type="primary", width="stretch")
    st.caption(f"Analysed on {report.meta['device'].upper()} in {report.meta['timings_s']['total']:.2f} s")
    ocr = report.meta.get("ocr_pages", {})
    if any(ocr.values()):
        st.caption(f"OCR used on {len(ocr.get('v1', []))} page(s) in version 1 and "
                   f"{len(ocr.get('v2', []))} in version 2. Verify recognized text against the PDFs.")
    if "paper" in report.meta.get("document_types", {}).values():
        st.caption(f"Paper mode: {s['paper_review']} non-prose or uncertain edits need source-page review.")

# ------------------------------------------------------------------------------ hero + overview
st.markdown(hero_html(title, f"{n1}  →  {n2}", s, report.doc_similarity), unsafe_allow_html=True)
st.markdown(overview_html(s), unsafe_allow_html=True)
brief, highlights = revision_brief(report)
with st.container(key="sdrow_brief"):
    section("Revision brief", brief)
    for item in highlights:
        a, b = st.columns([7, 1], vertical_alignment="center")
        a.markdown(f"**{item['section'] or 'Document body'}:** {item['detail']}")
        if b.button("Source", key=f"brief_source_{item['pair_index']}"):
            show_source(item["pair_index"])

tab_diff, tab_review, tab_quad, tab_graph, tab_lab, tab_benchmark, tab_about = st.tabs([
    "Clause diff", "Review", "Text vs meaning", "Impact graph", "Sentence lab", "Benchmark", "How it works"])

# ------------------------------------------------------------------------------ clause diff
VIEWS = {
    "All": lambda pa: True,
    "High impact": lambda pa: any(ch.severity == Severity.HIGH for ch in pa.changes),
    "Material / unresolved": lambda pa: pa.verdict in (Verdict.MATERIAL, Verdict.UNCLASSIFIED),
    "Manual review": lambda pa: pa.verdict == Verdict.UNCLASSIFIED and pa.pair.status != PairStatus.UNCHANGED,
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
    total_rows = len(rows)
    if total_rows > 60:
        page_size = 50
        page_count = (total_rows + page_size - 1) // page_size
        page_number = st.number_input("Result page", min_value=1, max_value=page_count, value=1)
        start = (page_number - 1) * page_size
        rows = rows[start:start + page_size]
        st.caption(f"Showing {start + 1}–{start + len(rows)} of {total_rows} matching clauses "
                   f"({len(report.pairs)} total)")
    else:
        st.caption(f"Showing {total_rows} of {len(report.pairs)} clauses")
    if not rows:
        st.markdown('<p class="sd-lead">No clauses match this filter.</p>', unsafe_allow_html=True)
    for k, pa in rows:
        with st.container(key=f"sdrow_{k}"):
            st.markdown(card_html(pa, impacts_by_id), unsafe_allow_html=True)
            if st.button("View source pages", key=f"source_pair_{k}", icon=":material/description:"):
                show_source(k)
            decision = doc_reviews.get(pair_key(pa))
            if decision:
                st.caption(f"Human review: {decision['decision']}" +
                           (f" · {decision['corrected_verdict']}" if decision['corrected_verdict'] else ""))
            if pa.pair.status == PairStatus.MODIFIED:
                with st.expander("Explain this result", icon=":material/manage_search:"):
                    st.markdown(frames_html(pa) + evidence_html(pa), unsafe_allow_html=True)

# ------------------------------------------------------------------------------ review queue
with tab_review:
    st.space("small")
    section("Human review", "Check uncertain findings first, confirm or correct the result, and export "
                            "a report that preserves both the model output and your decision.")
    candidates = review_candidates(report)
    pending = [index for index in candidates if pair_key(report.pairs[index]) not in doc_reviews]
    a, b = st.columns(2)
    a.metric("Pending", len(pending))
    b.metric("Reviewed", len(candidates) - len(pending))
    queue_mode = st.segmented_control("Queue", ["Pending", "All findings"], default="Pending")
    visible = pending if queue_mode == "Pending" else candidates
    if visible:
        selected = st.selectbox(
            "Finding to review", visible,
            format_func=lambda index: (f"#{index + 1} · "
                                       f"{(report.pairs[index].v2_section or report.pairs[index].v1_section or 'Body')} · "
                                       f"{(report.pairs[index].v2_text or report.pairs[index].v1_text or '')[:105]}"),
            key=f"review_selection_{fingerprint}_{queue_mode}")
        pair = report.pairs[selected]
        st.markdown(card_html(pair, impacts_by_id), unsafe_allow_html=True)
        if st.button("View source pages", key=f"review_source_{selected}", icon=":material/description:"):
            show_source(selected)
        key = pair_key(pair)
        existing = doc_reviews.get(key, {})
        choices = ["Confirmed", "Corrected", "Dismissed"]
        verdicts = ["MATERIAL", "MEANING_PRESERVING", "UNCLASSIFIED"]
        default_verdict = existing.get("corrected_verdict") or pair.verdict.value
        if default_verdict not in verdicts:
            default_verdict = "UNCLASSIFIED"
        with st.form(f"review_form_{key}"):
            decision = st.selectbox("Your decision", choices,
                                    index=choices.index(existing.get("decision", "Confirmed")))
            corrected = st.selectbox("Correct verdict (used when Corrected)", verdicts,
                                     index=verdicts.index(default_verdict))
            note = st.text_area("Reason or correction", value=existing.get("note", ""),
                                placeholder="Explain what the source pages show")
            if st.form_submit_button("Save review", type="primary"):
                doc_reviews[key] = {"decision": decision,
                                    "corrected_verdict": corrected if decision == "Corrected" else None,
                                    "note": note.strip()[:2000]}
                st.rerun()
    else:
        st.success("No findings remain in this queue.")
    st.download_button("Download reviewed report",
                       reviewed_report_json(report, fingerprint, doc_reviews),
                       file_name="semanticdiff_reviewed_report.json", mime="application/json")
    imported = st.file_uploader("Restore decisions from a reviewed report", type="json",
                                key=f"review_import_{fingerprint}")
    if imported:
        content = imported.getvalue()
        import_hash = hashlib.sha256(content).hexdigest()
        if st.session_state.get(f"review_imported_{fingerprint}") != import_hash:
            try:
                restored = import_decisions(content, report, fingerprint)
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(str(exc))
            else:
                doc_reviews.update(restored)
                st.session_state[f"review_imported_{fingerprint}"] = import_hash
                st.rerun()

# ------------------------------------------------------------------------------ text vs meaning
with tab_quad:
    st.space("small")
    section("Text change is not meaning change",
            "Each point is an edited clause. Upper left: small edits that changed the meaning, which line diffs "
            "under-report. Lower right: heavy rewrites that kept it, which line diffs over-report.")
    pts = [pa for pa in report.pairs if pa.pair.status == PairStatus.MODIFIED
           and pa.verdict_rule != "paper:non-prose"]
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

# ------------------------------------------------------------------------------ benchmark
with tab_benchmark:
    st.space("small")
    section("Independent paper alignment check",
            "Sentence links measured against the authors' arXivEdits test split. This checks alignment "
            "on clean paper text; PDF extraction and meaning verdicts are separate tasks.")
    scorecard_path = ROOT / "results" / "arxivedits_scorecard.json"
    if scorecard_path.exists():
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        st.caption(f"{scorecard['paper_groups']} paper groups · {scorecard['revision_pairs']} adjacent revisions · "
                   f"{scorecard['gold_links']:,} gold links ({scorecard['gold_edited_links']:,} edited) · "
                   f"fixed alignment threshold "
                   f"{scorecard['alignment']['threshold']:.2f}")
        columns = st.columns(3)
        for column, (name, result) in zip(columns, scorecard["systems"].items()):
            column.metric(f"{name} edited-link F1", f"{scorecard['edited_systems'][name]['f1']:.1%}")
        st.dataframe(pd.DataFrame([
            {"System": name, "All-link F1": result["f1"],
             "Edited-link precision": scorecard["edited_systems"][name]["precision"],
             "Edited-link recall": scorecard["edited_systems"][name]["recall"],
             "Edited-link F1": scorecard["edited_systems"][name]["f1"],
             "Correct": result["tp"], "False links": result["fp"],
             "Missed links": result["fn"]}
            for name, result in scorecard["systems"].items()]),
            hide_index=True, width="stretch")
        with st.expander("Revision-level results"):
            st.dataframe(pd.DataFrame([
                {"Paper": row["paper_id"], "Versions": row["versions"],
                 "Gold links": row["gold_links"], "Edited links": row["gold_edited_links"],
                 "SemanticDiff edited F1": row["SemanticDiff_edited"]["f1"]}
                for row in scorecard["per_revision"]]),
                hide_index=True, width="stretch")
        st.caption(scorecard["scope"])
        st.link_button("Dataset and annotation details", scorecard["source_repo"])
    else:
        st.info("Run `.venv/Scripts/python.exe -m eval.run_arxivedits_eval` to create the scorecard.")

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

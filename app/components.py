"""HTML building blocks for the SemanticDiff Streamlit app.

Visual language: Refero "Steep" (serif analytics on warm paper).
  * Achromatic system: ink #17191c on paper #ffffff, mist #f2f2f3 and fog #fafafb surfaces.
  * One chromatic pair, rationed: blush peach #fbe1d1 with sienna #5d2a1a ink. Used for the
    single accent card in the hero, inserted text, and "affected by a definition".
  * Serif (Source Serif 4, weight 400) only for display headings; Inter for everything else.
  * Shape rule: content cards 24px, floating artifacts 20px, inner panels and inputs 16px,
    buttons and chips fully rounded. Only floating artifacts carry a shadow.
"""
from __future__ import annotations

import html

from semanticdiff.models import ChangeCategory, PairAnalysis, PairStatus, Verdict

CSS = """
<style>
:root {
  --ink: #17191c; --paper: #ffffff; --mist: #f2f2f3; --fog: #fafafb; --hairline: #ececec;
  --slate: #6b6f7a;            /* Steep slate #777b86, darkened to pass WCAG AA on white */
  --ash: #979799; --smoke: #a3a6af;
  --peach: #fbe1d1; --sienna: #5d2a1a;
  --serif: 'Source Serif 4', 'Tiempos Headline', Georgia, serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
  --shadow-float: 0 0 0 1px rgba(4,23,43,0.05), 0 20px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.08);
  --shadow-pop: 0 0 0 1px rgba(4,23,43,0.05), 0 2px 8px rgba(0,0,0,0.06);
}
[data-testid="stMainBlockContainer"] { max-width: 1200px; padding-top: 2.4rem; padding-bottom: 5rem; }
h1, h2, h3 { font-weight: 400 !important; letter-spacing: -0.015em; }

/* ---------- streamlit widgets, restyled to the Steep vocabulary ---------- */
[data-testid="stTabs"] [role="tablist"] { gap: 2px; background: var(--mist); padding: 4px; border-radius: 9999px;
  width: fit-content; max-width: 100%; border: none; box-shadow: none; overflow-x: auto; }
[data-testid="stTab"] { height: 36px; padding: 0 16px; border-radius: 9999px; margin: 0; background: transparent; }
[data-testid="stTab"] > div:not([data-testid]) { display: none; }          /* default underline indicator */
[data-testid="stTab"] p { font-size: 14.5px; font-weight: 450; color: var(--slate); }
[data-testid="stTab"][aria-selected="true"] { background: var(--paper); box-shadow: var(--shadow-pop); }
[data-testid="stTab"][aria-selected="true"] p { color: var(--ink); }
[data-variant="pills"] { background: var(--paper); }
[data-variant="pills"][aria-checked="true"] { background: var(--ink) !important; border-color: var(--ink) !important; }
[data-variant="pills"][aria-checked="true"] p { color: var(--paper) !important; }

/* a clause row = keyed st.container: mist card holding the HTML card and its Explain expander */
[class*="st-key-sdrow"] { background: var(--mist); border-radius: 24px; padding: 8px; gap: 6px; }
[class*="st-key-sdrow"] [data-testid="stExpander"] details { background: var(--paper); border: none; border-radius: 16px; }
[class*="st-key-sdrow"] [data-testid="stExpander"] summary p { font-size: 14px; color: var(--slate); }

/* ---------- sidebar brand ---------- */
.sd-brand { display: flex; align-items: center; gap: 10px; }
.sd-brand .mark { width: 30px; height: 30px; border-radius: 9999px; background: var(--ink); color: var(--paper);
  display: grid; place-items: center; font-weight: 500; font-size: 17px; line-height: 1; }
.sd-brand .name { font-weight: 500; font-size: 18px; letter-spacing: -0.01em; color: var(--ink); }
.sd-brand-sub { color: var(--slate); font-size: 14px; line-height: 1.5; margin: 10px 0 0 0; }

/* ---------- hero ---------- */
.sd-hero { position: relative; isolation: isolate; display: grid; grid-template-columns: minmax(0, 1fr);
  gap: 28px; align-items: center; padding: 8px 0 28px 0; }
.sd-hero::before { content: ""; position: absolute; inset: -60px -40px -20px 25%; z-index: -1; pointer-events: none;
  background: radial-gradient(46% 58% at 58% 48%, rgba(251,225,209,1), rgba(251,225,209,0) 72%),
              radial-gradient(30% 40% at 30% 70%, rgba(251,225,209,0.45), rgba(251,225,209,0) 70%); filter: blur(6px); }
.sd-kicker { font-size: 15px; color: var(--slate); margin: 0 0 14px 0; }
.sd-display { font-family: var(--serif); font-weight: 400; font-size: clamp(2.2rem, 3.6vw, 3.25rem); line-height: 1.1;
  letter-spacing: -0.02em; color: var(--ink); margin: 0 0 18px 0; font-optical-sizing: auto; }
.sd-display em { font-style: italic; }
.sd-sub { font-size: 17px; line-height: 1.5; color: var(--slate); max-width: 46ch; margin: 0; }
.sd-files { margin-top: 18px; font-size: 13px; color: var(--slate); }

.sd-collage { display: grid; grid-template-columns: 270px 250px; gap: 16px; align-items: start; justify-self: end; margin-top: -8px; }
.sd-art.tall { grid-row: span 2; }
.sd-art { background: var(--paper); border-radius: 20px; box-shadow: var(--shadow-float); padding: 16px 18px; }
/* wide screens: collage floats to the right of the headline, Steep-style */
@media (min-width: 1500px) {
  [data-testid="stMainBlockContainer"] { max-width: 1360px; }
  .sd-hero { grid-template-columns: minmax(0, 1fr) 470px; gap: 44px; }
  .sd-collage { grid-template-columns: 250px minmax(0, 1fr); justify-self: stretch; margin-top: 0; }
  .offset { margin-top: 34px; }
}
.sd-art-title { font-size: 14px; font-weight: 500; color: var(--ink); }
.sd-big { font-size: 28px; font-weight: 500; letter-spacing: -0.02em; color: var(--ink); margin-top: 10px;
  font-variant-numeric: tabular-nums; line-height: 1; }
.sd-delta { font-size: 13px; color: var(--slate); margin-top: 6px; }
.sd-trow { display: grid; grid-template-columns: minmax(0, 1fr) 44px 20px; gap: 10px; align-items: center; padding: 7px 0;
  border-top: 1px solid var(--hairline); font-size: 13.5px; }
.sd-trow:first-of-type { margin-top: 12px; }
.sd-trow .n { color: var(--slate); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sd-trow .bar { height: 4px; border-radius: 9999px; background: var(--ink); justify-self: start; }
.sd-trow .v { text-align: right; font-weight: 500; font-variant-numeric: tabular-nums; }
.sd-peach { background: var(--peach); color: var(--sienna); border-radius: 20px; padding: 16px 18px; box-shadow: var(--shadow-float); }
.sd-peach .sd-art-title { color: var(--sienna); }
.sd-ring { display: flex; align-items: center; gap: 14px; margin-top: 10px; }
.sd-ring .pct { font-size: 22px; font-weight: 500; letter-spacing: -0.02em; line-height: 1; }
.sd-ring .lbl { font-size: 12.5px; margin-top: 4px; opacity: 0.85; }
.sd-sim { display: flex; gap: 22px; }

/* ---------- overview band ---------- */
.sd-overview { background: var(--fog); border: 1px solid var(--hairline); border-radius: 24px; padding: 18px 26px;
  display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin: 6px 0 30px 0; }
.sd-ov .v { font-size: 26px; font-weight: 500; letter-spacing: -0.02em; color: var(--ink); font-variant-numeric: tabular-nums; line-height: 1.1; }
.sd-ov .l { font-size: 13.5px; color: var(--slate); margin-top: 4px; }

/* ---------- section intro ---------- */
.sd-h2 { font-family: var(--serif); font-weight: 400; font-size: 2rem; line-height: 1.15; letter-spacing: -0.015em;
  color: var(--ink); margin: 6px 0 8px 0; }
.sd-lead { font-size: 16px; line-height: 1.55; color: var(--slate); max-width: 66ch; margin: 2px 0 16px 0; }
[class*="st-key-sdrow_plot"], [class*="st-key-sdrow_graph"] { padding: 16px; background: var(--fog); border: 1px solid var(--hairline); }

/* ---------- clause card ---------- */
.sd-rh { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 6px 8px 4px 10px; }
.sd-sec { font-size: 13.5px; color: var(--slate); margin-right: 4px; }
.sd-chip { display: inline-block; font-size: 12.5px; line-height: 1; padding: 6px 10px; border-radius: 9999px;
  border: 1px solid #dcdcdf; color: var(--ink); background: var(--paper); white-space: nowrap; }
.sd-chip.peach { background: var(--peach); border-color: var(--peach); color: var(--sienna); }
.sd-verdict { font-size: 13.5px; font-weight: 500; color: var(--ink); }
.sd-verdict.quiet { font-weight: 400; color: var(--slate); }
.sd-levels { margin-left: auto; display: flex; gap: 14px; font-size: 13px; color: var(--slate); }
.sd-levels b { font-weight: 500; color: var(--ink); }

.sd-panes { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.sd-pane { background: var(--paper); border-radius: 16px; padding: 14px 16px 16px 16px; font-size: 15.5px;
  line-height: 1.6; color: var(--ink); }
.sd-pane .cap { font-size: 12.5px; color: var(--slate); margin-bottom: 4px; }
.sd-del { color: var(--slate); text-decoration: line-through; text-decoration-color: rgba(23,25,28,0.45);
  background: #e6e6e9; border-radius: 6px; padding: 1px 3px; }
.sd-ins { color: var(--sienna); background: var(--peach); border-radius: 6px; padding: 1px 3px; }
.sd-empty { color: var(--ash); font-style: italic; }

.sd-fs { padding: 2px 10px 4px 10px; }
.sd-f { display: grid; grid-template-columns: 70px minmax(0, 1fr); gap: 14px; padding: 11px 0; align-items: start; }
.sd-f + .sd-f { border-top: 1px solid #e3e3e6; }
.sd-sev { font-size: 11.5px; font-weight: 500; letter-spacing: 0.02em; text-align: center; padding: 5px 0;
  border-radius: 9999px; margin-top: 1px; }
.sd-sev.HIGH { background: var(--ink); color: var(--paper); }
.sd-sev.MEDIUM { border: 1px solid var(--ink); color: var(--ink); padding: 4px 0; }
.sd-sev.LOW { background: #e6e6e9; color: var(--slate); }
.sd-ft { font-size: 15px; font-weight: 480; color: var(--ink); }
.sd-fm { font-size: 12.5px; color: var(--slate); margin-top: 3px; }
.sd-fm code { font-family: var(--mono); font-size: 11.5px; background: transparent; padding: 0; color: var(--slate); }
.sd-fv { font-size: 14px; margin-top: 7px; color: var(--ink); }
.sd-fv .o { color: var(--slate); text-decoration: line-through; text-decoration-color: rgba(23,25,28,0.45); }
.sd-fv .n { color: var(--sienna); background: var(--peach); border-radius: 6px; padding: 1px 5px; }
.sd-fv .arrow { color: var(--ash); margin: 0 6px; }
.sd-note { display: flex; gap: 10px; align-items: baseline; font-size: 14px; color: var(--ink); padding: 8px 0 2px 0; }
.sd-note .k { flex: none; font-size: 12px; padding: 3px 9px; border-radius: 9999px; border: 1px solid #dcdcdf; color: var(--slate); background: var(--paper); }
.sd-note.dep .k { background: var(--peach); border-color: var(--peach); color: var(--sienna); }

/* ---------- explain panel ---------- */
.sd-frames { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 2px 0 10px 0; }
.sd-frame { background: var(--fog); border: 1px solid var(--hairline); border-radius: 16px; padding: 4px 16px 8px 16px; }
.sd-frame-t { font-family: var(--serif); font-size: 19px; color: var(--ink); padding: 10px 0 6px 0; }
.sd-fr { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 12px; padding: 8px 0; border-top: 1px solid var(--hairline); }
.sd-fr .k { font-size: 12.5px; color: var(--slate); padding-top: 2px; }
.sd-fr .v { font-size: 14.5px; color: var(--ink); overflow-wrap: anywhere; }
.sd-fr .d { font-size: 12.5px; color: var(--slate); margin-top: 2px; line-height: 1.5; overflow-wrap: anywhere; }
.sd-evs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.sd-ev { background: var(--fog); border: 1px solid var(--hairline); border-radius: 16px; padding: 12px 14px; }
.sd-ev .t { font-size: 12.5px; color: var(--slate); margin-bottom: 4px; }
.sd-ev .b { font-size: 14px; color: var(--ink); line-height: 1.5; overflow-wrap: anywhere; }
.sd-ev code { font-family: var(--mono); font-size: 12px; background: var(--mist); padding: 1px 5px; border-radius: 6px; color: var(--ink); }

/* ---------- how it works ---------- */
.sd-pipe .row { display: grid; grid-template-columns: 88px minmax(0, 1fr); gap: 12px; padding: 9px 0; border-top: 1px solid var(--hairline); font-size: 14px; }
.sd-pipe .row:first-of-type { margin-top: 10px; }
.sd-pipe .s { font-weight: 500; color: var(--ink); }
.sd-pipe .d { color: var(--slate); }
.sd-models { margin-top: 14px; font-family: var(--mono); font-size: 12px; color: var(--slate); line-height: 1.8; }
.sd-dt { background: var(--mist); border-radius: 24px; padding: 8px; }
.sd-dt table { width: 100%; border-collapse: separate; border-spacing: 0; background: var(--paper); border-radius: 16px; overflow: hidden; font-size: 14px; }
.sd-dt table, .sd-dt th, .sd-dt td { border: none !important; }
.sd-dt th { text-align: left; font-weight: 400; color: var(--slate); font-size: 13px; padding: 12px 14px; }
.sd-dt td { padding: 11px 14px; color: var(--ink); vertical-align: top; }
.sd-dt tr + tr td { border-top: 1px solid var(--hairline) !important; }
.sd-dt td:first-child { font-weight: 500; width: 56px; }

@media (max-width: 1180px) {
  .sd-overview { grid-template-columns: repeat(3, minmax(0, 1fr)); row-gap: 18px; }
}
@media (max-width: 760px) {
  .sd-collage, .sd-panes, .sd-frames, .sd-evs { grid-template-columns: 1fr; }
  .sd-art.tall { grid-row: auto; }
  .sd-collage { justify-self: stretch; margin-top: 0; }
  .sd-overview { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .sd-levels { margin-left: 0; }
}
</style>
"""

CATEGORY_LABEL = {
    "NUMERIC": "Numbers", "TEMPORAL": "Dates and durations", "ENTITY": "Entities", "MODALITY": "Modality",
    "NEGATION": "Negation", "SCOPE": "Scope", "CONDITION": "Conditions", "REVERSAL": "Reversals",
    "STRUCTURE": "Added or removed", "UNEXPLAINED": "Unexplained",
}
SHORT_LABEL = {
    "NUMERIC": "Numbers", "TEMPORAL": "Dates, durations", "ENTITY": "Entities", "MODALITY": "Modality",
    "NEGATION": "Negation", "SCOPE": "Scope", "CONDITION": "Conditions", "REVERSAL": "Reversals",
    "STRUCTURE": "Structure", "UNEXPLAINED": "Unexplained",
}
VERDICT_LABEL = {
    Verdict.IDENTICAL: "Unchanged", Verdict.MEANING_PRESERVING: "Same meaning",
    Verdict.MATERIAL: "Meaning changed", Verdict.UNCLASSIFIED: "Unclassified change",
}


def e(s) -> str:
    return html.escape("" if s is None else str(s))


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# --------------------------------------------------------------------------- sidebar
def brand_html() -> str:
    return ('<div class="sd-brand"><span class="mark">≠</span><span class="name">SemanticDiff</span></div>'
            '<p class="sd-brand-sub">Compares what two document versions mean, not just what they say.</p>')


# --------------------------------------------------------------------------- hero
def _ring(pct: float, size: int = 76, stroke: int = 7) -> str:
    """Radial progress ring (a data mark, drawn as two SVG circles)."""
    r = (size - stroke) / 2
    c = 2 * 3.14159265 * r
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" aria-label="{pct:.0%}">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="rgba(93,42,26,0.16)" stroke-width="{stroke}"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#5d2a1a" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-dasharray="{c * pct:.1f} {c:.1f}" transform="rotate(-90 {size/2} {size/2})"/>'
            '</svg>')


def hero_html(title: str, files: str, summary: dict, sim: dict) -> str:
    s = summary
    emb = sim.get("embedding_cosine", 0.0)
    tfidf = sim.get("tfidf_cosine", 0.0)
    changed = s["material"] + s["unclassified"] + s["added"] + s["removed"]
    edited = s["modified"]
    edited_changed = s["material"] + s["unclassified"]
    high = s["severity"]["HIGH"]

    wording = ("Nearly identical on paper." if emb >= 0.9 else
               "Similar on paper." if emb >= 0.75 else "Substantially rewritten.")
    if changed:
        count = _plural(changed, "clause", "clauses").replace(" ", "&nbsp;")   # keep "12 clauses" on one line
        head = f'{wording} <em>{count}</em> changed meaning.'
    else:
        head = f'{wording} <em>No clause</em> changed meaning.'
    sub = (f'SemanticDiff aligned {s["aligned_rows"]} clauses and found {_plural(high, "high-impact change", "high-impact changes")}. '
           f'{_plural(s["affected_by_dependency"], "untouched clause", "untouched clauses")} shifted because a definition '
           f'{"it relies" if s["affected_by_dependency"] == 1 else "they rely"} on changed.')
    left = (f'<div><p class="sd-kicker">{e(title)}</p><div class="sd-display" role="heading" aria-level="1">{head}</div>'
            f'<p class="sd-sub">{e(sub)}</p><div class="sd-files">{e(files)}</div></div>')

    # floating artifact 1: findings by type
    cats = {k: v for k, v in s["categories"].items()}
    total = sum(cats.values())
    top = sorted(cats.items(), key=lambda kv: -kv[1])[:6]
    mx = max([v for _, v in top] or [1])
    rows = "".join(f'<div class="sd-trow"><span class="n">{e(SHORT_LABEL.get(k, k.title()))}</span>'
                   f'<span class="bar" style="width:{max(6, int(44 * v / mx))}px"></span><span class="v">{v}</span></div>'
                   for k, v in top)
    sev = s["severity"]
    findings = (f'<div class="sd-art tall"><div class="sd-art-title">Findings</div><div class="sd-big">{total}</div>'
                f'<div class="sd-delta">{sev["HIGH"]} high, {sev["MEDIUM"]} medium, {sev["LOW"]} low impact</div>{rows}</div>')

    # floating artifact 2: the single peach accent card
    pct = edited_changed / edited if edited else 0.0
    peach = (f'<div class="sd-peach offset"><div class="sd-art-title">Edited clauses</div>'
             f'<div class="sd-ring">{_ring(pct)}<div><div class="pct">{pct:.0%}</div>'
             f'<div class="lbl">changed meaning</div></div></div>'
             f'<div class="sd-delta" style="color:#5d2a1a;opacity:.85">{edited_changed} of {edited}; '
             f'{s["meaning_preserving"]} only reworded</div></div>')

    # floating artifact 3: what a similarity score would have said
    simcard = (f'<div class="sd-art"><div class="sd-art-title">Similarity scores</div><div class="sd-sim">'
               f'<div><div class="sd-big">{emb:.0%}</div><div class="sd-delta">embeddings</div></div>'
               f'<div><div class="sd-big">{tfidf:.0%}</div><div class="sd-delta">TF-IDF</div></div></div></div>')

    collage = f'<div class="sd-collage">{findings}{peach}{simcard}</div>'
    return f'<div class="sd-hero">{left}{collage}</div>'


def empty_html() -> str:
    return ('<div class="sd-hero" style="grid-template-columns:1fr"><div><p class="sd-kicker">Upload</p>'
            '<div class="sd-display" role="heading" aria-level="1">Compare two versions of <em>a document</em>.</div>'
            '<p class="sd-sub">Add version 1 and version 2 in the sidebar. Text-layer PDFs, plain text and Markdown '
            'work; scanned PDFs need OCR first.</p></div></div>')


def overview_html(summary: dict) -> str:
    s = summary
    cells = [(s["aligned_rows"], "clauses aligned"), (s["modified"], "edited"), (s["added"], "added"),
             (s["removed"], "removed"), (s["moved"], "moved"), (s["meaning_preserving"], "reworded only")]
    body = "".join(f'<div class="sd-ov"><div class="v">{v}</div><div class="l">{e(l)}</div></div>' for v, l in cells)
    return f'<div class="sd-overview">{body}</div>'


# --------------------------------------------------------------------------- clause card
def _clause_label(section: str) -> str:
    """'Section 3 > 3.1' -> 'Clause 3.1'; '1 > 1.2' -> 'Clause 1.2'."""
    if not section:
        return ""
    last = section.split(" > ")[-1]
    return f"Clause {last}" if last[:1].isdigit() else last


def diff_sides(pa: PairAnalysis) -> tuple[str, str]:
    if pa.pair.status == PairStatus.ADDED:
        return '<span class="sd-empty">Not in version 1</span>', f'<span class="sd-ins">{e(pa.v2_text)}</span>'
    if pa.pair.status == PairStatus.REMOVED:
        return f'<span class="sd-del">{e(pa.v1_text)}</span>', '<span class="sd-empty">Removed in version 2</span>'
    if not pa.token_diff:
        return e(pa.v1_text), e(pa.v2_text)
    left, right = [], []
    for op, old, new in pa.token_diff:
        if op == "equal":
            left.append(e(old))
            right.append(e(new))
            continue
        if old:
            core, ws = old.rstrip(), old[len(old.rstrip()):]
            left.append(f'<span class="sd-del">{e(core)}</span>{e(ws)}')
        if new:
            core, ws = new.rstrip(), new[len(new.rstrip()):]
            right.append(f'<span class="sd-ins">{e(core)}</span>{e(ws)}')
    return "".join(left), "".join(right)


def finding_html(ch) -> str:
    val = ""
    if ch.category != ChangeCategory.STRUCTURE and (ch.old or ch.new):
        o = f'<span class="o">{e(ch.old)}</span>' if ch.old else '<span class="sd-empty">none</span>'
        n = f'<span class="n">{e(ch.new)}</span>' if ch.new else '<span class="sd-empty">none</span>'
        val = f'<div class="sd-fv">{o}<span class="arrow">→</span>{n}</div>'
    conf = f", confidence {ch.confidence:.2f}" if ch.confidence < 0.99 else ""
    kind = CATEGORY_LABEL.get(ch.category.value, ch.category.value.title())
    meta = f'{e(kind)}, {e(ch.subtype.lower().replace("_", " "))}{e(conf)}. Rule <code>{e(ch.severity_rule)}</code>'
    return (f'<div class="sd-f"><div class="sd-sev {ch.severity.value}">{ch.severity.value.title()}</div>'
            f'<div><div class="sd-ft">{e(ch.description)}</div><div class="sd-fm">{meta}</div>{val}</div></div>')


def card_html(pa: PairAnalysis, impacts_by_id: dict) -> str:
    st_ = pa.pair.status
    chips = []
    if st_ == PairStatus.ADDED:
        chips.append('<span class="sd-chip peach">Added</span>')
    elif st_ == PairStatus.REMOVED:
        chips.append('<span class="sd-chip">Removed</span>')
    elif st_ == PairStatus.MODIFIED:
        chips.append('<span class="sd-chip">Edited</span>')
    if pa.pair.moved:
        chips.append('<span class="sd-chip">Moved</span>')
    if pa.affected_by:
        chips.append('<span class="sd-chip peach">Affected by a definition</span>')
    verdict = ""
    if st_ in (PairStatus.MODIFIED, PairStatus.UNCHANGED):
        quiet = pa.verdict in (Verdict.IDENTICAL, Verdict.MEANING_PRESERVING)
        verdict = (f'<span class="sd-verdict{" quiet" if quiet else ""}">'
                   f'{e(VERDICT_LABEL.get(pa.verdict, pa.verdict.value))}</span>')

    sec = _clause_label(pa.v2_section or pa.v1_section)
    if pa.pair.moved and pa.v1_section and pa.v2_section and pa.v1_section != pa.v2_section:
        sec = f"{_clause_label(pa.v1_section)} moved to {_clause_label(pa.v2_section).replace('Clause ', '')}"
    levels = ""
    if st_ == PairStatus.MODIFIED:
        levels = (f'<div class="sd-levels"><span>Text change <b>{pa.textual_level.value.lower()}</b></span>'
                  f'<span>Meaning change <b>{pa.semantic_level.value.lower()}</b></span></div>')
    head = (f'<div class="sd-rh">{f"<span class=sd-sec>{e(sec)}</span>" if sec else ""}{"".join(chips)}'
            f'{verdict}{levels}</div>')

    left, right = diff_sides(pa)
    panes = (f'<div class="sd-panes"><div class="sd-pane"><div class="cap">Version 1</div>{left}</div>'
             f'<div class="sd-pane"><div class="cap">Version 2</div>{right}</div></div>')

    shown = [c for c in pa.changes if not (c.category == ChangeCategory.STRUCTURE and c.subtype in ("ADDED", "REMOVED"))]
    if st_ in (PairStatus.ADDED, PairStatus.REMOVED):
        shown = pa.changes
    body = "".join(finding_html(c) for c in shown)
    body += "".join(f'<div class="sd-note"><span class="k">Equivalent</span><span>{e(n)}</span></div>'
                    for n in pa.notes)
    for iid in pa.affected_by:
        imp = impacts_by_id.get(iid)
        if imp:
            body += (f'<div class="sd-note dep"><span class="k">Depends on</span><span>{e(imp.reason)} '
                     f'Path: {e(" → ".join(imp.path))}</span></div>')
    tail = f'<div class="sd-fs">{body}</div>' if body else ""
    return head + panes + tail


# --------------------------------------------------------------------------- explain panel
_STATE = {"OBLIGATION": "Obligation", "PROHIBITION": "Prohibition", "RECOMMENDATION": "Recommendation",
          "DISCOURAGED": "Discouraged", "PERMISSION": "Permission", "NO_OBLIGATION": "No obligation",
          "ASSERTION": "Plain statement", "NEGATED_ASSERTION": "Negated statement"}
_BASE_UNIT = {"time": "seconds", "business_time": "working days", "percent": "%", "length": "metres",
              "mass": "kg", "data": "bytes", "age": "seconds"}


def _num(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.4f}".rstrip("0").rstrip(".")


def _quantity_detail(q) -> str:
    base = f"{q.unit} {_num(q.si_value)}" if q.dimension == "currency" else \
        f"{_num(q.si_value)} {_BASE_UNIT.get(q.dimension, '')}".strip()
    parts = [f"{q.dimension.replace('_', ' ')}, normalised to {base}"]
    if q.comparator:
        parts.append(f"bound “{q.comparator.replace('_', ' ')}”")
    if q.approximate:
        parts.append("approximate conversion")
    return "; ".join(parts)


def _np_detail(r) -> str:
    parts = [f"head noun “{r.head}”"]
    if r.quantifier:
        parts.append(f"quantifier “{r.quantifier}”")
    parts.append("modifiers " + (", ".join(f"“{m}”" for m in r.modifiers) if r.modifiers else "none"))
    if r.exceptions:
        parts.append("exceptions " + ", ".join(f"“{x}”" for x in r.exceptions))
    return "; ".join(parts)


def frame_rows(frame) -> list[tuple[str, str, str]]:
    """(slot, value, detail) rows describing a ProvisionFrame in plain language."""
    if frame is None:
        return []
    rows = []
    for d in frame.predicates:
        detail = f"verb “{d.predicate}”" + (f", trigger “{d.trigger}”" if d.trigger else ", no modal")
        if d.negated:
            detail += f", negated by “{d.negation_cue}”"
        rows.append(("Modality", _STATE.get(d.state.value, d.state.value.title()), detail))
    labels = {"agent": "Applies to", "patient": "Object"}
    for role in ("agent", "patient"):
        r = getattr(frame, role)
        if not r:
            continue
        # "The fee is Rs. 5,000": the amount is a predicate value, already shown as a quantity
        if role == "patient" and any(q.span[0] < r.span[1] and q.span[1] > r.span[0] for q in frame.quantities):
            continue
        rows.append((labels[role], r.text, _np_detail(r)))
    for c in frame.conditions:
        kind = "exception" if c.polarity == "negative" else "condition"
        rows.append(("Condition", c.text, f"{kind}, introduced by “{c.marker.replace('_', ' ')}”"))
    for q in frame.quantities:
        rows.append(("Quantity", q.raw, _quantity_detail(q)))
    for d in frame.dates:
        rows.append(("Date or time", d.raw, f"normalised to {d.normalized}" if d.normalized else ""))
    for en in frame.entities:
        rows.append(("Entity", en.text, f"spaCy label {en.label}"))
    if frame.defined_term:
        rows.append(("Defines", frame.defined_term, "matched a definition pattern"))
    return rows


def frame_html(frame, title: str) -> str:
    rows = frame_rows(frame)
    if not rows:
        body = '<div class="sd-fr"><div></div><div class="sd-empty">Nothing extracted</div></div>'
    else:
        body = "".join(f'<div class="sd-fr"><div class="k">{e(k)}</div><div><div class="v">{e(v)}</div>'
                       + (f'<div class="d">{e(d)}</div>' if d else "") + "</div></div>" for k, v, d in rows)
    return f'<div class="sd-frame"><div class="sd-frame-t">{e(title)}</div>{body}</div>'


def frames_html(pa: PairAnalysis) -> str:
    return (f'<div class="sd-frames">{frame_html(pa.frames.get("v1"), "Version 1 frame")}'
            f'{frame_html(pa.frames.get("v2"), "Version 2 frame")}</div>')


def evidence_html(pa: PairAnalysis) -> str:
    cells = []
    if pa.nli:
        cells.append(("NLI, both directions",
                      e(f"Entailment {pa.nli.entail_12:.2f} forward and {pa.nli.entail_21:.2f} backward. "
                        f"Contradiction {pa.nli.contra_12:.2f} and {pa.nli.contra_21:.2f}.")))
    if pa.pair.sim_embed is not None:
        cells.append(("Similarity", e(f"Embedding cosine {pa.pair.sim_embed:.3f}. Text change {pa.textual_change:.2f}.")))
    res = pa.frames.get("residual", {})
    dec = (f"Rule <code>{e(pa.verdict_rule)}</code>. Rules explain "
           f"{pa.frames.get('explained_ratio', 1.0):.0%} of the word-level diff.")
    if res.get("old") or res.get("new"):
        dec += (f" Unexplained: <code>{e(' '.join(res.get('old', [])) or 'none')}</code> to "
                f"<code>{e(' '.join(res.get('new', [])) or 'none')}</code>.")
    cells.append(("Decision", dec))
    body = "".join(f'<div class="sd-ev"><div class="t">{e(t)}</div><div class="b">{b}</div></div>' for t, b in cells)
    return f'<div class="sd-evs">{body}</div>'


# --------------------------------------------------------------------------- how it works
PIPELINE = [("Extract", "PDF or text, headers and footers removed"),
            ("Segment", "sentences, list items and sections"),
            ("Align", "Hungarian matching on meaning, LIS for moves"),
            ("Frame", "modality, scope, conditions, quantities in base units"),
            ("Diff", "atomic changes, each word claimed once"),
            ("Decide", "decision table with two-way NLI"),
            ("Impact", "definition graph, affected clauses")]


def pipeline_html(models: dict) -> str:
    rows = "".join(f'<div class="row"><span class="s">{e(s)}</span><span class="d">{e(d)}</span></div>'
                   for s, d in PIPELINE)
    mods = "<br>".join(e(models[k]) for k in ("spacy", "embedding", "nli"))
    return (f'<div class="sd-art sd-pipe"><div class="sd-art-title">Pipeline</div>{rows}'
            f'<div class="sd-models">{mods}</div></div>')


DECISION_ROWS = [
    ("V0", "none", "texts identical", "Unchanged"),
    ("V1", "none", "NLI entailment in both directions", "Same meaning"),
    ("V2", "none", "NLI contradiction", "Meaning changed, flagged as NLI-only"),
    ("V3", "none", "only function words differ", "Same meaning"),
    ("V4", "none", "embedding cosine above threshold", "Same meaning, low confidence"),
    ("V5", "none", "none of the above", "Unclassified change"),
    ("V6", "one or more", "NLI shown as supporting evidence", "Meaning changed"),
]


def decision_html() -> str:
    rows = "".join(f"<tr><td>{a}</td><td>{e(b)}</td><td>{e(c)}</td><td>{e(d)}</td></tr>" for a, b, c, d in DECISION_ROWS)
    return ('<div class="sd-dt"><table><tr><th>Rule</th><th>Rule-based changes</th><th>Other evidence</th>'
            f'<th>Verdict</th></tr>{rows}</table></div>')

"""HTML rendering helpers for the SemanticDiff Streamlit app."""
from __future__ import annotations

import html

from semanticdiff.models import ChangeCategory, PairAnalysis, PairStatus, Severity, Verdict

CSS = """
<style>
:root {
  --sd-add-bg: rgba(46, 160, 67, 0.16);   --sd-add-fg: #1a7f37;
  --sd-del-bg: rgba(248, 81, 73, 0.16);   --sd-del-fg: #cf222e;
  --sd-mod-bg: rgba(210, 153, 34, 0.14);  --sd-mov-bg: rgba(130, 80, 223, 0.14);
  --sd-border: rgba(128, 128, 128, 0.25); --sd-muted: rgba(128, 128, 128, 0.95);
  --sd-high: #cf222e; --sd-med: #bf8700; --sd-low: #57606a;
}
@media (prefers-color-scheme: dark) {
  :root { --sd-add-fg: #3fb950; --sd-del-fg: #ff7b72; --sd-high: #ff7b72; --sd-med: #d29922; --sd-low: #8b949e; }
}
.sd-row { border: 1px solid var(--sd-border); border-radius: 8px; margin: 0 0 14px 0; overflow: hidden; }
.sd-head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 8px 12px;
           border-bottom: 1px solid var(--sd-border); font-size: 0.85rem; }
.sd-sec { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--sd-muted); }
.sd-badge { padding: 1px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600;
            letter-spacing: .02em; border: 1px solid var(--sd-border); white-space: nowrap; }
.sd-b-ADDED { background: var(--sd-add-bg); color: var(--sd-add-fg); }
.sd-b-REMOVED { background: var(--sd-del-bg); color: var(--sd-del-fg); }
.sd-b-MODIFIED { background: var(--sd-mod-bg); }
.sd-b-MOVED { background: var(--sd-mov-bg); }
.sd-b-AFFECTED { background: var(--sd-mod-bg); border-style: dashed; }
.sd-sides { display: grid; grid-template-columns: 1fr 1fr; }
.sd-side { padding: 10px 12px; line-height: 1.55; font-size: 0.95rem; }
.sd-side + .sd-side { border-left: 1px solid var(--sd-border); }
.sd-lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: .06em; color: var(--sd-muted);
          margin-bottom: 2px; }
.sd-del { background: var(--sd-del-bg); color: var(--sd-del-fg); text-decoration: line-through;
          border-radius: 3px; padding: 0 1px; }
.sd-ins { background: var(--sd-add-bg); color: var(--sd-add-fg); border-radius: 3px; padding: 0 1px; }
.sd-empty { color: var(--sd-muted); font-style: italic; }
.sd-changes { border-top: 1px solid var(--sd-border); padding: 6px 12px 8px 12px; }
.sd-change { display: grid; grid-template-columns: 70px 1fr; gap: 10px; padding: 5px 0; align-items: baseline; }
.sd-sev { font-size: 0.7rem; font-weight: 700; text-align: center; border-radius: 4px; padding: 1px 0;
          border: 1px solid currentColor; }
.sd-sev-HIGH { color: var(--sd-high); } .sd-sev-MEDIUM { color: var(--sd-med); } .sd-sev-LOW { color: var(--sd-low); }
.sd-title { font-weight: 600; }
.sd-cat { font-size: 0.72rem; color: var(--sd-muted); font-family: ui-monospace, monospace; margin-left: 6px; }
.sd-arrow { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; margin-top: 2px; }
.sd-arrow .o { color: var(--sd-del-fg); } .sd-arrow .n { color: var(--sd-add-fg); }
.sd-note { font-size: 0.82rem; color: var(--sd-muted); padding: 2px 0; }
.sd-meters { margin-left: auto; display: flex; gap: 6px; }
.sd-meter { font-size: 0.72rem; padding: 1px 8px; border-radius: 4px; border: 1px solid var(--sd-border); }
.sd-hero { border: 1px solid var(--sd-border); border-radius: 10px; padding: 14px 18px; margin: 4px 0 12px 0; }
.sd-hero b { font-size: 1.05rem; }
@media (max-width: 760px) { .sd-sides { grid-template-columns: 1fr; }
  .sd-side + .sd-side { border-left: none; border-top: 1px solid var(--sd-border); } }
</style>
"""

_VERDICT_LABEL = {
    Verdict.IDENTICAL: "Unchanged", Verdict.MEANING_PRESERVING: "Meaning-preserving rewrite",
    Verdict.MATERIAL: "Material change", Verdict.UNCLASSIFIED: "Unclassified change",
}


def e(s) -> str:
    return html.escape("" if s is None else str(s))


def diff_sides(pa: PairAnalysis) -> tuple[str, str]:
    if pa.pair.status == PairStatus.ADDED:
        return '<span class="sd-empty">— not present —</span>', f'<span class="sd-ins">{e(pa.v2_text)}</span>'
    if pa.pair.status == PairStatus.REMOVED:
        return f'<span class="sd-del">{e(pa.v1_text)}</span>', '<span class="sd-empty">— removed —</span>'
    if not pa.token_diff:
        return e(pa.v1_text), e(pa.v2_text)
    left, right = [], []
    for op, old, new in pa.token_diff:
        if op == "equal":
            left.append(e(old)); right.append(e(new))
        else:
            if old:
                core, ws = old.rstrip(), old[len(old.rstrip()):]
                left.append(f'<span class="sd-del">{e(core)}</span>{e(ws)}')
            if new:
                core, ws = new.rstrip(), new[len(new.rstrip()):]
                right.append(f'<span class="sd-ins">{e(core)}</span>{e(ws)}')
    return "".join(left), "".join(right)


def change_html(ch) -> str:
    arrow = ""
    if ch.category != ChangeCategory.STRUCTURE and (ch.old or ch.new):
        o = f'<span class="o">{e(ch.old)}</span>' if ch.old else '<span class="sd-empty">∅</span>'
        n = f'<span class="n">{e(ch.new)}</span>' if ch.new else '<span class="sd-empty">∅</span>'
        arrow = f'<div class="sd-arrow">{o} → {n}</div>'
    conf = f" · confidence {ch.confidence:.2f}" if ch.confidence < 0.99 else ""
    return (f'<div class="sd-change"><div class="sd-sev sd-sev-{ch.severity.value}">{ch.severity.value}</div>'
            f'<div><span class="sd-title">{e(ch.description)}</span>'
            f'<span class="sd-cat">{e(ch.category.value)}/{e(ch.subtype)} · rule {e(ch.severity_rule)}{conf}</span>'
            f'{arrow}</div></div>')


def row_html(pa: PairAnalysis, impacts_by_id: dict) -> str:
    st = pa.pair.status
    badges = []
    if st != PairStatus.UNCHANGED:
        badges.append(f'<span class="sd-badge sd-b-{st.value}">{st.value}</span>')
    if pa.pair.moved:
        badges.append('<span class="sd-badge sd-b-MOVED">MOVED</span>')
    if pa.affected_by:
        badges.append('<span class="sd-badge sd-b-AFFECTED">AFFECTED BY DEPENDENCY</span>')
    badges.append(f'<span class="sd-badge">{e(_VERDICT_LABEL.get(pa.verdict, pa.verdict.value))}</span>')
    sec = pa.v2_section or pa.v1_section
    if pa.pair.moved and pa.v1_section and pa.v2_section and pa.v1_section != pa.v2_section:
        sec = f"{pa.v1_section} → {pa.v2_section}"
    meters = ""
    if st == PairStatus.MODIFIED:
        meters = (f'<div class="sd-meters"><span class="sd-meter">TEXTUAL {pa.textual_level.value}</span>'
                  f'<span class="sd-meter sd-sev-{pa.semantic_level.value}">SEMANTIC {pa.semantic_level.value}</span></div>')
    left, right = diff_sides(pa)
    body = "".join(change_html(c) for c in pa.changes
                   if not (c.category == ChangeCategory.STRUCTURE and c.subtype in ("ADDED", "REMOVED")))
    notes = "".join(f'<div class="sd-note">≡ {e(n)}</div>' for n in pa.notes)
    for iid in pa.affected_by:
        imp = impacts_by_id.get(iid)
        if imp:
            notes += f'<div class="sd-note">⚠ {e(imp.reason)} Path: {e(" → ".join(imp.path))}</div>'
    extra = f'<div class="sd-changes">{body}{notes}</div>' if (body or notes) else ""
    return (f'<div class="sd-row"><div class="sd-head"><span class="sd-sec">{e(sec) or "—"}</span>{"".join(badges)}'
            f'{meters}</div><div class="sd-sides"><div class="sd-side"><div class="sd-lbl">Old</div>{left}</div>'
            f'<div class="sd-side"><div class="sd-lbl">New</div>{right}</div></div>{extra}</div>')


def frame_table(frame) -> list[dict]:
    """Flatten a ProvisionFrame into rows for the Explain panel."""
    if frame is None:
        return []
    rows = []
    for d in frame.predicates:
        rows.append({"slot": "deontic", "value": f"{d.predicate}: {d.modal_class.value}"
                     f"{' + NEG' if d.negated else ''} → {d.state.value}", "evidence": d.trigger})
    for role in ("agent", "patient"):
        r = getattr(frame, role)
        if r:
            rows.append({"slot": role, "value": r.text,
                         "evidence": f"head={r.head}, quantifier={r.quantifier}, modifiers={r.modifiers}"})
    for c in frame.conditions:
        rows.append({"slot": "condition", "value": c.text, "evidence": f"{c.marker} ({c.polarity})"})
    for q in frame.quantities:
        rows.append({"slot": "quantity", "value": q.raw,
                     "evidence": f"{q.value:g} {q.unit or ''} [{q.dimension}] = {q.si_value:g} base"
                                 f"{', ' + q.comparator if q.comparator else ''}"})
    for d in frame.dates:
        rows.append({"slot": "date/time", "value": d.raw, "evidence": d.normalized})
    for en in frame.entities:
        rows.append({"slot": "entity", "value": en.text, "evidence": en.label})
    if frame.defined_term:
        rows.append({"slot": "defines", "value": frame.defined_term, "evidence": "definition pattern"})
    return rows

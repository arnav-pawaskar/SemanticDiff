"""Print the extracted ProvisionFrame for each sentence given on the command line or stdin.

    python scripts/show_frames.py "Students may not access the laboratory."
"""
import sys

from semanticdiff.analyze.frame import build_frame
from semanticdiff.resources import get_spacy

nlp = get_spacy()
lines = sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]
for s in lines:
    f = build_frame(nlp(s))
    print(s)
    for d in f.predicates:
        print(f"   DEONTIC  {d.predicate:<10} {d.modal_class.value:<14} neg={d.negated!s:<5} "
              f"{d.state.value:<16} trigger={d.trigger!r}")
    for r in (f.agent, f.patient):
        if r:
            print(f"   {r.role.upper():<8} {r.text!r} head={r.head} q={r.quantifier} "
                  f"mods={r.modifiers} exc={r.exceptions}")
    for c in f.conditions:
        print(f"   COND     [{c.marker}/{c.polarity}] {c.text!r}")
    for q in f.quantities:
        print(f"   QTY      {q.raw!r} {q.value} {q.unit} {q.dimension} si={q.si_value} cmp={q.comparator}")
    for d in f.dates:
        print(f"   DATE     {d.raw!r} -> {d.normalized}")
    for e in f.entities:
        print(f"   ENT      {e.text!r} {e.label}")
    if f.defined_term:
        print(f"   DEFINES  {f.defined_term!r}")

# SemanticDiff — a meaning-aware document version comparison engine

Traditional diff tools report *which characters changed*. SemanticDiff reports **how the meaning changed**:
two document versions go in, a structured, explainable map of semantic changes comes out.

```
OLD  All students may submit applications within 30 days.
NEW  Final-year students must submit applications within 15 days with faculty approval.

HIGH  Obligation strengthened — Permission → Obligation   may → must
HIGH  Time limit shortened (bound tightened)              ≤30 days → ≤15 days
HIGH  Condition added                                     ∅ → with faculty approval
HIGH  Subject scope narrowed                              All students → Final-year students
```

```
OLD  The server must respond within 500 milliseconds.
NEW  The server shall return a response within half a second.

Meaning-preserving rewrite   ('must' ≡ 'shall' (both Obligation); '500 milliseconds' ≡ 'half a second')
```

No LLM APIs are used. The pipeline combines spaCy dependency parsing, deterministic linguistic rules,
sentence-transformer embeddings and a pretrained NLI cross-encoder, and every result can be traced
to the rule that produced it.

## Setup (Windows / Linux, Python 3.12)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m spacy download en_core_web_lg
.venv/Scripts/python.exe -m scripts.smoke_models     # downloads SBERT + NLI, checks the GPU
```

(On Linux, replace `.venv/Scripts/python.exe` with `.venv/bin/python`. Use the CPU torch
wheel if there is no NVIDIA GPU; everything still runs, though NLI is slower.)

## Use it

```bash
# Streamlit demo (Git-style semantic diff, dashboard, text-vs-meaning quadrant, impact graph)
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py

# CLI on two documents (PDF or TXT), optional JSON report
.venv/Scripts/python.exe -m semanticdiff data/demo/academic_regulations_v1.txt data/demo/academic_regulations_v2.txt --json results/report.json

# CLI on two sentences
.venv/Scripts/python.exe -m semanticdiff --pair "Students may access the laboratory." "Students may not access the laboratory."

# Inspect the structured frame the extractors build for a sentence
.venv/Scripts/python.exe scripts/show_frames.py "Visitors are prohibited from entering the hostel on weekends."

# Regression tests (sentence, document, PDF and paper cases)
.venv/Scripts/python.exe -m pytest tests -q
```

### Try two versions of a real research paper

```bash
.venv/Scripts/python.exe -m scripts.download_paper_demo
.venv/Scripts/python.exe -m semanticdiff results/cache/papers/attention_v1.pdf results/cache/papers/attention_v2.pdf --json results/cache/papers/attention_report.json
.venv/Scripts/python.exe -m scripts.validate_paper_demo results/cache/papers/attention_report.json
.venv/Scripts/python.exe -m semanticdiff results/cache/papers/bert_v1.pdf results/cache/papers/bert_v2.pdf --json results/cache/papers/bert_report.json
.venv/Scripts/python.exe -m scripts.validate_paper_demo results/cache/papers/bert_report.json
```

This downloads the public [v1](https://arxiv.org/abs/1706.03762v1) and
[v2](https://arxiv.org/abs/1706.03762v2) PDFs of *Attention Is All You Need* into an ignored
local cache, plus [v1](https://arxiv.org/abs/1810.04805v1) and
[v2](https://arxiv.org/abs/1810.04805v2) of *BERT*. Both pairs appear in the Streamlit demo selector.
Paper ingestion detects an early **Abstract**, skips author/affiliation front matter, reads
two-column pages down each column, ignores rotated page stamps, and shows PDF page numbers
on comparison cards. Bracketed numeric citations are kept in the visible text diff but are
not treated as measurements; the report notes when those references change. Displayed raster
images, including inline images and scanned body regions on pages with native text, are OCRed
locally. The report records which pages used OCR.
The **Revision brief** prioritizes evidence-linked findings. Use **Source** on a brief item or
**View source pages** on a comparison card to see the marked regions in both PDFs. The **Review**
tab lets a reader confirm, correct, or dismiss findings and download a report that keeps the
original model result beside each human decision. Upload that report in the Review tab to restore
the decisions for the same document pair.
The real-paper checks verify the *Attention* abstract's **41.0 → 41.2 BLEU** and
**3.5 → 4.5 days** changes, the *BERT* abstract's **80.4% → 80.5%** change,
and several passages moved into BERT's appendix. Paper mode keeps tables,
equations, captions, references, incomplete sentences and ambiguous footnote markers in
the report with a **Manual review** label. Added and removed paper passages are also
review items because PDF extraction can split a sentence differently across versions. Adjacent
sentence fragments with strong text coverage are rejoined before comparison.
Recognized text inside raster figures is grouped into a review item. The system does not
certify its semantic meaning or interpret visual chart data.

## Architecture

```
PDF/TXT ─► extract (PyMuPDF, header/footer removal, de-hyphenation)
        ─► structure (sections, numbered clauses, list items)
        ─► segment (spaCy sentences + list items + ';' splits)          → Provision[]
        ─► align: exact match → Hungarian on fused(SBERT cosine, token overlap) → LIS move detection
                  (paper mode matches within top-level sections and requires stronger move evidence)
                                                                         → ADDED / REMOVED / MODIFIED / MOVED
        ─► for each MODIFIED pair, build a ProvisionFrame per version:
              deontic state   (modal class × negation, from the dependency parse)
              agent / patient (voice-normalised), quantifier, modifiers, exceptions
              conditions      (if/unless/provided that/subject to/with-X-approval, attachment-aware)
              quantities      (numbers, %, currency, units → base unit, comparators)
              dates / times, named entities, defined terms
        ─► frame diff with span ownership + grounding → atomic changes
        ─► bidirectional NLI (DeBERTa-v3) + decision table → verdict
        ─► severity rules (config/severity.yaml, every change records its rule id)
        ─► definition/reference graph → clauses affected by a changed definition
        ─► DiffReport (JSON) → UI / CLI / evaluation
```

Key design decisions (see `docs/PLAN.md` for the full review):

* **Modality and negation are one dimension.** "may → may not" is one NEGATION change
  (Permission → Prohibition), not a modality change plus a negation; "may not" ≡ "must not".
* **Embeddings never decide materiality.** They are used for alignment (where insensitivity to
  "75% vs 70%" is a feature) and as a baseline. `cos("Interest rate is 5%", "... 15%") = 0.85`.
* **NLI is evidence, not the arbiter.** It is run in both directions; deterministic extractors
  take precedence for numbers and modals, and disagreements are surfaced in the report.
* **Span ownership + grounding.** Each diff token is explained by at most one atomic change, and
  every change must be anchored in a token that actually differs (parse noise is not a change).
* **Severity is a configurable heuristic**, not ground truth.

### Decision table (`semanticdiff/changes/verdict.py`)

| rule | extractor changes | NLI / other | verdict |
|---|---|---|---|
| V0 | – | texts identical | IDENTICAL |
| V1 | none | bidirectional entailment ≥ τₑ | MEANING_PRESERVING |
| V2 | none | contradiction ≥ τ꜀ | MATERIAL (REVERSAL, NLI-only, flagged unexplained) |
| V3 | none | only function words changed | MEANING_PRESERVING |
| V4 | none | embedding cosine ≥ τₛ, but NLI does not confirm equivalence | MEANING_PRESERVING (low confidence) for general documents; UNCLASSIFIED for papers |
| V5 | none | otherwise | UNCLASSIFIED |
| V6 | ≥ 1 | – | MATERIAL (+ note if NLI says equivalent) |

τ values are calibrated on the dev split only (`eval/calibrate.py` → `config/calibrated.yaml`).

## Evaluation

```bash
.venv/Scripts/python.exe -m eval.build_benchmark      # 388 labelled pairs, dev/test split by seed
.venv/Scripts/python.exe -m eval.calibrate            # tune verdict thresholds on dev
.venv/Scripts/python.exe -m eval.run_align_eval       # Task C: alignment (+ tunes align threshold on dev)
.venv/Scripts/python.exe -m eval.run_eval             # Tasks A & B, ablations, figures → results/
.venv/Scripts/python.exe -m eval.run_arxivedits_eval  # independent sentence-alignment scorecard → results/
```

* **Task A** — did the meaning change? Lexical edit ratio, TF-IDF, SBERT cosine, SBERT + bidirectional
  NLI and SemanticDiff (plus −NLI and −unit-normalisation ablations). Baseline thresholds are tuned on dev
  for macro-F1. Also reports per-category accuracy, "danger zones" (small edit / big meaning change and
  vice versa) and McNemar tests.
* **Task B** — what changed? Multi-label over 8 categories vs a keyword-diff baseline (no parsing, no
  normalisation); per-category P/R/F1, macro-F1, exact match, confusion matrix.
* **Task C** — alignment: positional and `difflib` baselines vs the SemanticDiff aligner (link and move F1).

The arXivEdits scorecard uses the [authors' test split](https://github.com/chaojiang06/arXivEdits)
of clean, LaTeX-derived paper sentences. It downloads the pinned test file to the ignored local cache,
excludes headings and the dataset's skipped sentences, and compares individual alignment links with
positional and `difflib` baselines. All-link and edited-link precision, recall and F1 are reported
separately because unchanged sentences dominate the gold links. The existing threshold is fixed;
no test labels are used for tuning.
Results appear in the app's **Benchmark** tab and `results/arxivedits_scorecard.json`. This checks
sentence alignment, not PDF extraction, revision intent, or semantic verdict accuracy. The corpus
combines automatically identified easy links with human annotation for ambiguous cases, so the
scorecard is not a wholly human-labeled measure.

Labelling conventions: `data/eval/LABELING.md`. **Important caveat:** the synthetic benchmark is
generated with rules written by the same author as the extractors, so its numbers are an upper bound.
Add an independently written natural test set at `data/eval/natural/pairs.jsonl` (format in
LABELING.md) — `run_eval` picks it up automatically and reports it separately.

## Repository layout

```
semanticdiff/   engine: ingest/ align/ analyze/ changes/ impact/ report/, pipeline.py, models.py, lexicons.py
app/            Streamlit UI
config/         default.yaml, severity.yaml, calibrated.yaml (generated)
data/demo/      demo document pairs
data/eval/      seeds, labelling guide, generated benchmark, natural test set (to add)
eval/           benchmark generator, baselines, calibration, evaluation, figures
results/        evaluation tables, figures, predictions
tests/          golden acceptance tests
docs/PLAN.md    design review, frozen architecture, phased plan
```

## Limitations

Displayed raster regions are OCRed, but OCR errors can affect alignment and extracted numbers.
Small images may be skipped, and visual chart content or vector drawings are not interpreted.
Extracted table cells, equations, captions and recognized figure labels remain text for manual review.
Source previews mark a matching phrase when possible and otherwise use the extracted paragraph
or image region; these boxes can be broader than a single sentence. Review decisions are kept in
the browser session until exported and restored from the reviewed JSON report.
Paper front matter before **Abstract** is excluded from comparison.
The NLI and rule models were developed for short normative prose, so scientific claims without
clear numeric or lexical cues can remain UNCLASSIFIED. PDF reading order is heuristic and should be
reviewed for complex layouts. Adjacent paper fragments with high text coverage can be rejoined;
larger clause rewrites and distant split/merge cases may still appear as added and removed.
Deontic "may" is assumed (epistemic "may" is not distinguished); scope analysis covers
quantifiers and noun-phrase modifiers of the agent/patient, not arbitrary scope; antonym reversal relies on
a lexicon plus NLI; the impact graph follows explicit definitions and "Section N" references only.

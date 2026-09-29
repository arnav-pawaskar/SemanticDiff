# SemanticDiff — Design Review, Frozen V1 Architecture & Phased Plan

> Two document versions go in. A structured, explainable map of how their meaning changed comes out.

Status: **PLAN (pre-implementation)** · Target: 7 working days · Environment: Windows 11, RTX 4060 (8 GB), Python 3.12 venv via `uv`

---

## Part 1 — Critical Review of the Proposed Architecture

The proposal is strong in intent. Below are the places where, as written, it is technically wrong, redundant, too ambitious for a week, or likely to produce unreliable output, followed by the fix adopted in V1.

### 1.1 Things that are technically wrong or will break

| # | Issue in proposal | Why it's a problem | V1 decision |
|---|---|---|---|
| R1 | **Modality and negation treated as independent analyses.** | `may appear → may not appear` would be reported as *both* a modality change and a negation, double counting. Negation changes the *deontic meaning* of the modal: `may not` = prohibition, `need not` = no obligation, `must not` = prohibition. They are one dimension, not two. | Single **deontic model** per predicate: `(modal_class, negated) → deontic_state` with states `OBLIGATION, PROHIBITION, RECOMMENDATION, DISCOURAGED, PERMISSION, NO_OBLIGATION, ASSERTION`. Emit `NEGATION` when polarity flips, `MODALITY` when the modal class changes, and derive the reading (e.g. *Permission → Prohibition*) from the resulting state. |
| R2 | **"Clause" segmentation implied to be real clause splitting.** | Dependency-based clause splitting produces fragments (`"if approved by their manager"`) that align badly and that NLI models were never trained on. It makes steps 2–7 *less* reliable. | Alignment unit = **provision**: a sentence, a list item `(a)/(i)/1.`, or a `;`-separated top-level segment. Clause structure is analysed *inside* the provision via the dependency tree, not by cutting text. The UI can still say "clause". |
| R3 | **Alignment by pure semantic similarity (implicitly Hungarian on cosine).** | Assignment on cosine alone ignores order, so it cannot tell *moved* from *in place*; and a fixed threshold picked by hand will silently mis-pair short sentences. | Three-stage aligner: (1) exact match on normalised text, (2) Hungarian assignment on a **fused** similarity (embedding cosine + token overlap) with a **calibrated** threshold, (3) **moved = matched pairs not on the Longest Increasing Subsequence** of V2 indices (the idea behind patience diff). `moved` is orthogonal to `modified`: a pair can be both. |
| R4 | **NLI as a single call per pair.** | NLI is directional. `All students may apply ⊨ Final-year students may apply` but not the reverse. One call loses exactly the signal you want. Also: NLI models are known to be poor at numeric reasoning (`500 ms` vs `half a second`) and inconsistent on modals. | **Bidirectional NLI** (2 calls/pair). Equivalence = entailment both ways. One-way entailment is a corroborating signal for scope narrowing/broadening. NLI is **evidence, never the sole arbiter** — deterministic extractors take precedence for numbers/modals, and disagreements are surfaced, not hidden. |
| R5 | **"Semantic contradiction/reversal" as a peer category of negation/modality.** | Overlaps negation (`may → may not`) and modality (`must → must not`). Gold labels become ambiguous and eval numbers become meaningless. | `REVERSAL` is defined narrowly as **polarity flip not caused by an explicit negation cue**: antonym substitution (`allowed → prohibited`, `increase → decrease`, `eligible → ineligible`) or NLI-contradiction with no deterministic explanation. `may → may not` is gold-labelled `NEGATION` only. |
| R6 | **Numerical vs temporal boundary undefined.** | `30 days → 15 days` could be either. | Rule: a quantity whose normalised unit has the **time dimension** (ms…years) or a calendar date/time → `TEMPORAL`. Everything else (%, currency, counts, other units) → `NUMERIC`. |
| R7 | **Entity vs scope boundary undefined.** | `students → employees`: entity or scope? | Subject **head noun replaced** → `ENTITY` (actor change). Head noun kept, **quantifier / modifiers / exceptions** changed → `SCOPE` (narrowed / broadened / changed). |
| R8 | **Rule representation with logical conditions (`attendance >= 75%`).** | Turning free-text conditions into logic is semantic parsing — unrealistic in a week and brittle. | Keep the frame idea (it's good), but conditions are stored as **text spans + extracted quantities + comparator** (`at least`, `more than`, `within`, `up to`…). Only the quantity+comparator part is normalised. |
| R9 | **The ms / "half a second" / "seven days = one week" examples need more than regex.** | Needs number-word parsing (`seven`, `half`, `a`), fractions, and unit conversion. `word2number` doesn't handle `half`. | Custom small normaliser: number words 0–100 + fractions (`half`, `quarter`, `a/an` = 1) + unit table (~40 units with dimension and SI factor). Equal-after-normalisation quantities produce an `EQUIVALENT_REWRITE` note — this is what makes the SLA example *provably* meaning-preserving. |
| R10 | **SRL for `subject/action/…` fields (implied).** | AllenNLP SRL is archived and broken on modern Python/torch. | Frames built by **rules over the spaCy dependency parse** (nsubj/nsubjpass subtree, ROOT verb + aux, advcl/mark, prep attachments). Fully explainable in the viva. |

### 1.2 Redundant or low-value parts

- **"POS comparison" in Layer 1.** POS tags changing tells you almost nothing about meaning. POS is used *inside* extractors; it's dropped as a standalone signal.
- **The three "layers" framed as a sequence.** They're parallel **signals** fused by a decision layer. V1 is *Signals → Frame diff → Fusion/decision table → Severity*. This framing is also much easier to defend.
- **Embeddings as a change detector.** Only used for (a) alignment and (b) the baseline. Useful twist: embeddings being insensitive to `75% → 70%` is *good for alignment* (we want those two sentences paired) and *bad for change detection* — a neat point for the report.

### 1.3 Unrealistic in one week → deferred or reduced

| Feature | V1 treatment |
|---|---|
| Split / merge detection | **Deferred** (stretch, Phase 8). Unmatched sentences appear as ADDED/REMOVED; we note the limitation. |
| Naming derived concepts in the impact graph ("Scholarship Eligibility") | Not generated. Dependent nodes are labelled by **section + short snippet**. Naming is summarisation, which is out of scope. |
| Unrestricted cross-references | V1 handles (a) **defined terms** (`"X" means…`, `X shall mean…`, `X refers to…`, `the term "X"`) and (b) explicit **section references** (`Section 4`, `clause 2(b)`). 1–2 hop propagation. |
| Internal contradiction detection | Stretch (Phase 8), candidate pairs pruned by embedding similarity + same subject head + same predicate lemma, then numeric/polarity/NLI check. |
| Tables, multi-column PDFs, scanned PDFs | Out of scope. Text-layer PDFs only. Stated in limitations. |
| Epistemic vs deontic `may` (`it may rain`) | Treated as deontic (normative documents). Limitation noted. |

### 1.4 Reliability risks you should know about up front

1. **spaCy NER on policy text is weak** (`Dean`, `Examination Committee` often missed). Entity changes therefore also come from **token-diff replace spans on proper nouns / role nouns**, not NER alone.
2. **Severity** must be presented as SemanticDiff's heuristic, configurable in `config/severity.yaml`, with every change recording the **rule id** that assigned its severity.
3. **Evaluation circularity (biggest academic risk).** If the same person writes the perturbation templates and the extractor lexicons, the benchmark measures agreement with yourself. Mitigations built into the plan:
   - A **hand-written "natural" test subset** (~100 pairs), ideally written by a classmate, **frozen before extractor tuning**.
   - Synthetic generator uses lexical variants that are **partly held out** from extractor development (dev/test split of lexical items, not just of pairs).
   - All thresholds (alignment, NLI, baselines) tuned **on dev only**; results reported on test, and **synthetic vs natural reported separately**.
4. **Dependency hell on Windows.** You currently only have **Python 3.14**; spaCy/thinc wheels for 3.14 are not something to bet a deadline on. Use `uv venv --python 3.12`. Avoid `en_core_web_trf` (spacy-transformers pins `transformers` and conflicts with sentence-transformers) → use **`en_core_web_lg`**. Install **CUDA torch** to use the 4060 for SBERT + NLI.
5. **NLI latency.** DeBERTa-v3-base on GPU: ~5 ms/pair; on CPU ~100 ms. NLI is run **only on MODIFIED pairs**, twice. Models cached via `st.cache_resource`.

---

## Part 2 — Frozen V1 Architecture

### 2.1 Pipeline

```
 V1 (PDF/TXT)              V2 (PDF/TXT)
     │                          │
     ▼                          ▼
 ┌──────────────── ingest ─────────────────┐
 │ extract   : PyMuPDF text, de-hyphenate, │
 │             drop repeated header/footer │
 │ structure : headings, section numbers,  │
 │             list items → blocks         │
 │ segment   : spaCy sents + list/';' split│
 └──────────────────┬──────────────────────┘
                    ▼
          Provision[] (id, section_path, text, spaCy Doc)
                    │
 ┌──────────────── align ──────────────────┐
 │ 1 exact match (normalised text)         │
 │ 2 fused sim = α·cos(SBERT) + (1-α)·tok  │
 │   Hungarian + calibrated threshold      │
 │ 3 LIS over matched pairs → MOVED        │
 └──────────────────┬──────────────────────┘
       AlignedPair[] : UNCHANGED | MODIFIED | ADDED | REMOVED  (+ moved flag)
                    │   (only MODIFIED pairs go deeper)
                    ▼
 ┌──────────── analyse (signals) ───────────────────────────────┐
 │ lexical   : token diff opcodes, textual-change score          │
 │ quantities: numbers, %, currency, units, durations → SI       │
 │ temporal  : dates/times (dateutil on spaCy DATE/TIME spans)   │
 │ deontic   : modal class + negation per predicate (dep parse)  │
 │ scope     : subject NP → quantifier, modifiers, exceptions    │
 │ conditions: if/unless/provided that/subject to/upon/with-X-   │
 │             approval (attachment-aware)                        │
 │ entities  : NER + proper/role-noun replace spans              │
 │ nli       : P(e|c|n) for v1→v2 and v2→v1                      │
 │   ─────────► ProvisionFrame (all fields optional, span-backed) │
 └──────────────────┬───────────────────────────────────────────┘
                    ▼
 ┌──────────── changes ────────────────────┐
 │ frame diff → AtomicChange[]             │
 │ span ownership (no double counting)     │
 │ residual unexplained diff tokens        │
 │ decision table → pair verdict           │
 │ severity rules (YAML)                   │
 └──────────────────┬──────────────────────┘
                    ▼
 ┌──────────── impact ─────────────────────┐
 │ defined terms + section refs → graph    │
 │ changed definition ⇒ AFFECTED clauses   │
 └──────────────────┬──────────────────────┘
                    ▼
            DiffReport (JSON)  ──►  Streamlit UI  /  CLI  /  eval harness
```

The **DiffReport JSON is the contract** between engine, UI and evaluation. The UI never calls NLP code directly except through `pipeline.compare()`.

### 2.2 Models & libraries (frozen)

| Purpose | Choice | Fallback |
|---|---|---|
| Parsing, NER, sentences | spaCy `en_core_web_lg` | `en_core_web_sm` |
| Embeddings / alignment | `sentence-transformers/all-mpnet-base-v2` | `all-MiniLM-L6-v2` |
| NLI | `cross-encoder/nli-deberta-v3-base` (3-way) | `cross-encoder/nli-distilroberta-base` |
| Assignment | `scipy.optimize.linear_sum_assignment` | — |
| Lexical | `difflib`, `rapidfuzz` | — |
| TF-IDF baseline, metrics | scikit-learn | — |
| Dates | `python-dateutil` | keep raw string |
| PDF | PyMuPDF (`pymupdf`) | — |
| Graph | `networkx` + `st.graphviz_chart` (DOT) | table view |
| UI | Streamlit (+ Plotly for the quadrant chart) | — |
| Config | YAML (`pyyaml`), dataclasses for schema | — |

### 2.3 Core data model (`semanticdiff/models.py`)

```text
Provision        id, doc_version, index, section_path, section_title, kind(sentence|list_item|heading),
                 text, char_span, page
Quantity         raw, value, unit, dimension(time|percent|currency|count|length|mass|…),
                 si_value, comparator(at_least|at_most|more_than|less_than|within|up_to|exactly|None),
                 span, head_noun
DeonticInfo      predicate_lemma, trigger("may not"), modal_class, negated, state, span
ScopeInfo        subject_text, head_lemma, quantifier(all|each|any|some|no|only|None),
                 modifiers[], exceptions[], span
Condition        marker(if|unless|provided_that|subject_to|upon|with), text, quantities[], span
ProvisionFrame   subject: ScopeInfo?, predicates: DeonticInfo[], conditions[], quantities[],
                 dates[], entities[], defined_term?
AlignedPair      v1_id?, v2_id?, status(UNCHANGED|MODIFIED|ADDED|REMOVED), moved,
                 sim{embed, token, fused}
AtomicChange     category(NUMERIC|TEMPORAL|ENTITY|MODALITY|NEGATION|SCOPE|CONDITION|REVERSAL|
                          UNEXPLAINED), subtype(WEAKENED|STRENGTHENED|INTRODUCED|REMOVED|
                          INCREASED|DECREASED|NARROWED|BROADENED|ADDED|REPLACED|…),
                 old, new, old_span, new_span, severity(LOW|MEDIUM|HIGH), severity_rule,
                 evidence{}, confidence
PairAnalysis     pair, textual_change(0-1), semantic_level, verdict(MEANING_PRESERVING|MATERIAL|
                 UNCLASSIFIED), nli{e12,c12,n12,e21,c21,n21}, changes[], notes[] (e.g. 500 ms ≡ 0.5 s)
Impact           source_change_id, term_or_section, affected_provision_id, path[], reason
DiffReport       meta(models, config hash, timings), doc_similarity{tfidf, embed},
                 summary counts, pairs[], impacts[], graph{nodes, edges}
```

### 2.4 The decision table (how a verdict is reached — the viva slide)

`M` = material atomic changes from deterministic extractors · `E` = entailment both directions ≥ τₑ · `C` = max contradiction ≥ τ꜀ · `S` = embedding cosine

| M | NLI | Verdict | Semantic level |
|---|---|---|---|
| ∅ | E | MEANING_PRESERVING | LOW |
| ∅ | not E, not C, S ≥ τₛ | MEANING_PRESERVING (low confidence) | LOW |
| ∅ | C | MATERIAL — `REVERSAL` (NLI-only, flagged *unexplained*) | HIGH, low confidence |
| ∅ | neither, S < τₛ | UNCLASSIFIED — `UNEXPLAINED` change | MEDIUM |
| ≠∅ | any | MATERIAL | max(severity of changes) |
| ≠∅ | E | MATERIAL + **"extractors and NLI disagree"** flag | as above |

Thresholds τ are **calibrated on the dev split**, stored in `config/default.yaml`, and printed in the report metadata.

### 2.5 Key algorithms (one paragraph each — what you'll say in the viva)

- **Alignment.** Build an n×m fused-similarity matrix; pre-pair exact duplicates; run Hungarian on `1 − sim`; reject pairs below τ_align (→ ADDED/REMOVED). Among accepted pairs sorted by V1 index, compute the LIS of V2 indices; pairs off the LIS are MOVED. UNCHANGED if normalised texts are equal, else MODIFIED.
- **Span ownership.** Each atomic change claims the diff tokens it explains. Extractors run in precedence order *deontic → quantities/temporal → conditions → scope → entities*. A token already claimed can't produce a second change (so `30 days → 15 days` inside a condition is one `TEMPORAL` change *attributed to* that condition, not two). Unclaimed content-word diff tokens + non-entailing NLI → `UNEXPLAINED`. We report "explained coverage" as a metric.
- **Quantity pairing.** Numbers inside the same `replace` opcode of the token diff are paired first; leftovers paired by same dimension + same head noun. Equal SI values → `EQUIVALENT_REWRITE` note, not a change. Comparator-aware direction: `at least 75% → at least 85%` ⇒ *threshold raised (stricter)*; `within 30 days → within 15 days` ⇒ *deadline shortened (stricter)*.
- **Deontic state.** For each verb predicate: collect `aux`/`auxpass` modals, multiword triggers (`required to`, `is permitted to`, `has to`, `entitled to`, `prohibited from`), `neg` children, and negative determiners on the subject (`No student may…`). Map to state; pair predicates across versions by lemma (fallback: ROOT↔ROOT).
- **Scope.** Subject NP = subtree of `nsubj/nsubjpass` of ROOT. Extract quantifier det, `amod`/`compound` modifiers, prep/relcl post-modifiers, and exceptions (`except`, `other than`, `excluding`). Same head, V2 modifiers ⊃ V1 modifiers or universal→restricted quantifier ⇒ NARROWED; reverse ⇒ BROADENED; otherwise CHANGED. NLI one-way entailment raises confidence.
- **Conditions.** `advcl` with `mark ∈ {if, unless, provided, when, where, once, as long as}`, `prep ∈ {subject to, upon, with, after, before}` **attached to the verb** whose object is in a condition-noun lexicon (`approval, permission, consent, authorisation, certificate, clearance, sanction…`). Same `with`-phrase attached to the **subject noun** is scope (`students with medical exemptions`) — this attachment distinction is a good dependency-parsing demonstration.
- **Impact graph.** Regex + quote-aware definition extraction in both versions → `term → defining provision`. Term occurrences (lemma-normalised, plural-aware, case-insensitive for capitalised multiword terms) and `Section N` references → edges. If a defining/referenced provision has a MATERIAL change or is REMOVED, every referencing provision (even UNCHANGED) is marked **AFFECTED (indirect)** with the path and reason. Max 2 hops.

### 2.6 Severity (heuristic, configurable)

`config/severity.yaml`, e.g.:

```yaml
NEGATION:   {default: HIGH}
REVERSAL:   {default: HIGH}
MODALITY:   {default: HIGH, rules: [{if: "old,new in RECOMMENDATION,PERMISSION", then: MEDIUM}]}
NUMERIC:    {default: MEDIUM, rules: [{if: "relative_change >= 0.10 or has_comparator", then: HIGH}]}
TEMPORAL:   {default: MEDIUM, rules: [{if: "deadline_changed", then: HIGH}]}
CONDITION:  {default: HIGH}
SCOPE:      {default: MEDIUM}
ENTITY:     {default: MEDIUM}
UNEXPLAINED:{default: MEDIUM}
MOVED_ONLY: {default: LOW}
escalate_if_definition: true   # any change inside a definition provision → +1 level
```

Each change stores `severity_rule` so the UI can show *why* it's HIGH.

### 2.7 Folder structure (frozen)

```
SemanticDiff/
├── README.md
├── pyproject.toml / requirements.txt
├── config/
│   ├── default.yaml            # model names, thresholds (τ_align, τ_e, τ_c, τ_s, α), device
│   └── severity.yaml
├── semanticdiff/
│   ├── __init__.py
│   ├── __main__.py             # CLI: python -m semanticdiff v1 v2 --json out.json
│   ├── config.py
│   ├── models.py               # dataclasses (the schema above)
│   ├── pipeline.py             # compare(v1, v2, config) -> DiffReport
│   ├── resources.py            # cached spaCy / SBERT / NLI loaders, device selection
│   ├── lexicons.py             # modals, negation cues, quantifiers, condition markers/nouns, antonyms
│   ├── ingest/
│   │   ├── extract.py          # PDF/TXT → clean text
│   │   ├── structure.py        # headings, section numbering, list items
│   │   └── segment.py          # → Provision[]
│   ├── align/
│   │   ├── similarity.py
│   │   └── aligner.py
│   ├── analyze/
│   │   ├── lexical.py
│   │   ├── quantities.py
│   │   ├── temporal.py
│   │   ├── deontic.py
│   │   ├── scope.py
│   │   ├── conditions.py
│   │   ├── entities.py
│   │   ├── nli.py
│   │   └── frame.py
│   ├── changes/
│   │   ├── diff.py             # frame diff + span ownership → AtomicChange[]
│   │   ├── verdict.py          # decision table
│   │   └── severity.py
│   ├── impact/
│   │   ├── definitions.py
│   │   └── graph.py
│   └── report/
│       └── serialize.py
├── app/
│   ├── streamlit_app.py
│   └── views/  (dashboard.py, diff_view.py, quadrant.py, impact_view.py, filters.py)
├── data/
│   ├── demo/                   # policy_v1/v2.txt(+pdf), sla_v1/v2.txt, regulations_v1/v2.txt
│   └── eval/
│       ├── seeds/              # source provisions (hand-written + sampled)
│       ├── natural/            # hand-written test pairs (frozen early)
│       ├── generated/          # synthetic pairs (jsonl, dev/test)
│       └── alignment/          # synthetic document pairs with gold links
├── eval/
│   ├── perturb/                # one generator per category
│   ├── build_benchmark.py
│   ├── build_alignment_set.py
│   ├── baselines.py
│   ├── run_eval.py             # Task A + Task B + ablations
│   ├── run_align_eval.py       # Task C
│   └── make_figures.py
├── results/                    # tables (csv/md), confusion matrices, plots
├── tests/
│   ├── golden/                 # spec examples as expected-output fixtures
│   └── test_*.py
└── docs/
    ├── PLAN.md                 # this file
    └── DESIGN.md               # final architecture + decision table for report/viva
```

---

## Part 3 — Evaluation Design

### 3.1 Tasks

| Task | Question | Systems compared | Metrics |
|---|---|---|---|
| **A. Material-change detection** (binary) | Did meaning change? | 1 Lexical (edit ratio) · 2 TF-IDF cosine · 3 SBERT cosine · 4 SBERT + bidirectional NLI · 5 SemanticDiff | P, R, F1, accuracy; confusion matrix; **per-category recall** (which systems miss negation/numbers?); McNemar 3 vs 5 and 4 vs 5 |
| **B. Change-type classification** (multi-label, 9 types) | *What* changed? | Keyword-diff baseline (token diff + lexicon lookup, no parsing) · NLI-only mapping · SemanticDiff · **ablations** (−NLI, −quantity normalisation, −dependency negation (keyword instead), −attachment rule for conditions) | per-label P/R/F1, micro & **macro-F1**, multilabel confusion matrices, exact-match ratio, Hamming loss |
| **C. Alignment** | Are provisions paired and statused correctly? | Positional (i↔i) · `difflib` line diff · SemanticDiff aligner (± LIS move detection) | link P/R/F1, status accuracy (ADDED/REMOVED/MODIFIED/MOVED), move-detection F1 |
| **D. Impact (case study)** | Are indirectly affected clauses found? | SemanticDiff | Qualitative on 3–4 hand-built documents; P/R over hand-labelled affected clauses |

Thresholds for systems 1–4 are tuned on **dev** to maximise F1 — otherwise the comparison is unfair to baselines.

### 3.2 Headline analysis: the "danger zone"

Scatter every test pair on **textual change (x)** vs **gold meaning change (colour)**. Report each system's recall on the **low-textual / meaning-changing** quadrant (e.g. edit ratio < 0.15 and gold = material) and specificity on the **high-textual / meaning-preserving** quadrant. This answers the research question directly: *lexical similarity fails in both quadrants; embeddings fail in the first; SemanticDiff should succeed in both.*

### 3.3 Benchmark composition (~480 pairs, 50/50 dev/test split by seed sentence)

| Source | Pairs | Notes |
|---|---|---|
| Synthetic single-change: NUMERIC, TEMPORAL, ENTITY, MODALITY, NEGATION, SCOPE, CONDITION, REVERSAL | 8 × 35 = 280 | Rule-based perturbations over ~120 seed provisions (policy/SLA/HR/tenancy style) |
| Meaning-preserving (in-domain) | 80 | Modal synonyms (`shall ↔ must ↔ is required to`), unit rewrites (`7 days ↔ one week`), voice/reordering — hand-verified |
| Meaning-preserving (out-of-domain) | 40 | Sampled positive pairs from PAWS/MRPC (HF `datasets`) — checks we don't over-flag general paraphrase |
| Compound (2–3 mutations) | 40 | Tests atomic decomposition |
| **Natural hand-written test set** | ~40–100 | Written independently (ideally by a classmate), frozen on Day 1–2, test-only |
| Alignment docs (Task C) | 30 doc pairs | Seed documents with scripted delete/insert/move/modify ops; gold links known by construction |

Every pair: `{id, v1, v2, labels:[...], material: bool, source, split}` in JSONL. Label conventions follow R5–R7 above and are written in `data/eval/LABELING.md` before any data is generated.

> **Disclosure:** if any seed sentences or paraphrases are drafted with AI assistance, they must be manually verified and this is stated in the report. The core pipeline itself uses no LLM API.

---

## Part 4 — Phased Implementation Plan

Each phase has a **deliverable**, **exit criteria** (objective, testable), and a **cut-line** (what to drop if the phase overruns). Estimates assume ~8–10 focused hours/day.

### Phase 0 — Foundations (Day 1, ~3 h)
- `uv venv --python 3.12`; install CUDA torch, spaCy + `en_core_web_lg`, sentence-transformers, transformers, scipy, sklearn, rapidfuzz, pymupdf, dateutil, networkx, streamlit, plotly, pyyaml, pytest.
- Smoke script: load all three models on GPU; print device and a sample NLI output.
- `models.py` schema, `config.py`, `default.yaml`, `severity.yaml`, `lexicons.py` v0.
- **Golden fixtures**: every example from the spec (killer example, SLA ms/half-second, 7 days/one week, lab access negation, the 4-mutation sentence, interest rate 5%→15%, reimbursement condition, scope narrowing, eligible-student impact) as `tests/golden/*.yaml` with expected atomic changes. These are the acceptance tests for the whole project.
- Write `data/eval/LABELING.md`; start the **natural test set** (or hand it to a classmate now).
- **Exit:** models load on GPU; `pytest` collects golden tests (failing is expected).

### Phase 1 — Ingestion & segmentation (Day 1, ~5 h) · priority 1
- `extract.py`: TXT (encoding-safe), PDF via PyMuPDF blocks; de-hyphenation; join wrapped lines; remove lines repeated on ≥50% of pages (headers/footers) and page numbers.
- `structure.py`: detect section headings (`1.`, `1.2`, `Section 4`, `Article IV`, ALL-CAPS/short title lines), list items `(a)`, `(i)`, `•`; maintain a `section_path`.
- `segment.py`: spaCy sentence split per block + list-item and top-level `;` splitting; normalise whitespace/quotes; `Provision` objects with stable ids.
- Write the 3 demo document pairs (`policy`, `sla`, `regulations` — the regulations pair carries the Eligible-Student definition case). Make one PDF.
- **Exit:** 3 demo docs segment with 0 broken sentences by manual inspection; unit tests for heading/list detection and PDF header removal.
- **Cut-line:** drop ALL-CAPS heading heuristic; keep numeric headings only.

### Phase 2 — Alignment & structural status (Day 2, ~6 h) · priorities 2–3
- `similarity.py`: batched SBERT embeddings (GPU), token-overlap (rapidfuzz `token_set_ratio`), fused matrix.
- `aligner.py`: exact pre-match → Hungarian → threshold → LIS move detection → statuses.
- `build_alignment_set.py` (small version, ~10 doc pairs now) to calibrate τ_align and α on dev.
- CLI prints a git-like structural diff (`=`, `~`, `+`, `-`, `↕`).
- **Exit:** A,B,C,D → A,C′,D,E test passes; a pure reorder is reported as MOVED not ADD+DEL; alignment F1 ≥ 0.9 on the dev alignment set.
- **Cut-line:** none — this is the backbone. Split/merge is already deferred.

### Phase 3 — Deterministic extractors (Days 2–4, ~14 h) · priorities 4–6
Order is chosen so the killer example works as early as possible.
1. `lexical.py` (1 h): token diff opcodes, textual-change score.
2. `quantities.py` (3 h): regex + spaCy NUM/like_num; number words & fractions; currency; percentages; unit table with dimension & SI factor; comparators; head-noun attachment; pairing across versions; equivalence notes. **Heaviest unit-test coverage in the project.**
3. `deontic.py` (3 h): modal classes, multiword triggers, `neg` deps, negative determiners, lexical prohibition/permission verbs & adjectives, predicate pairing, state transitions (weakened/strengthened/negation/reversal).
4. `temporal.py` (1 h): dates/times from spaCy DATE/TIME spans, dateutil normalisation; durations delegated to quantities.
5. `conditions.py` (2.5 h): markers, attachment-aware `with/upon/subject to`, condition-noun lexicon, condition span text, added/removed/modified via span similarity.
6. `scope.py` (2.5 h): subject NP analysis, quantifiers, modifier sets, exceptions, NARROWED/BROADENED/CHANGED.
7. `entities.py` (1 h): NER (non-numeric labels) + proper/role-noun replace spans.
8. `frame.py`: assemble `ProvisionFrame`.
- **Exit:** unit tests per extractor; golden tests for negation, modality, numeric, SLA-equivalence, 7-days/one-week, scope, condition pass.
- **Cut-line (end of Day 4):** temporal → raw-string compare of DATE entities; scope → quantifier + prenominal modifiers only (skip post-modifiers/exceptions).

### Phase 4 — Change synthesis, NLI, verdict, severity (Day 4, ~6 h) · priorities 4 & 7
- `nli.py`: bidirectional batched NLI on MODIFIED pairs only; label order verified against model config.
- `diff.py`: frame diff → `AtomicChange[]` with span ownership and precedence; residual `UNEXPLAINED`.
- `verdict.py`: decision table (§2.4); `severity.py`: YAML rules with rule ids.
- `pipeline.py` + `serialize.py`: end-to-end `compare()` → JSON; document-level TF-IDF & embedding similarity in `meta`.
- **Exit:** **all golden tests pass**, including the 4-mutation sentence decomposing into exactly SCOPE + MODALITY + TEMPORAL + CONDITION; killer example yields exactly the 4 expected findings; end-to-end runtime on demo docs < 10 s on GPU.

### Phase 5 — Evaluation (Day 5, ~9 h) · priority 8
- `perturb/`: one generator per category (with held-out lexical variants for test); `build_benchmark.py` → dev/test JSONL.
- Pull PAWS/MRPC positives; merge natural set.
- `baselines.py`: systems 1–4 with dev-tuned thresholds; keyword-diff baseline for Task B.
- `run_eval.py`: Task A, Task B, ablations; `run_align_eval.py`: Task C (expand to 30 docs).
- `make_figures.py`: confusion matrices, per-category recall bars, danger-zone scatter, results tables → `results/`.
- Error analysis: read 20 failures, categorise causes (parser error, lexicon gap, NLI error, pairing error) — this goes straight into the report.
- **Exit:** a single command regenerates every table and figure from scratch.
- **Cut-line:** drop McNemar; shrink compound set to 20; alignment set to 15 docs.

### Phase 6 — Streamlit UI (Day 6, ~6 h) · priority 9
- Upload V1/V2 (PDF/TXT) **or** pick a built-in demo pair (the viva must not depend on an upload working).
- **Header:** document-level TF-IDF & embedding similarity next to the count of material changes — "97 % similar, 4 material changes" is the hook.
- **Dashboard:** clauses analysed, added / removed / moved / modified, meaning-preserving, material, HIGH/MED/LOW.
- **Textual vs semantic quadrant** (Plotly): every modified pair as a point; click-through to its card.
- **Diff view:** side-by-side OLD/NEW with inline token highlighting; per-pair badges `TEXTUAL: HIGH · SEMANTIC: LOW`; atomic change cards (category, subtype, old → new, severity + rule id, evidence: NLI probabilities, quantity normalisation, dependency trigger).
- **Filters:** all, high impact, added, removed, modified, moved, numbers, temporal, entities, modality, negation, scope, conditions, affected-by-dependency.
- **"Explain" expander** per pair showing the frame for both versions and the decision-table row that fired — this is your viva tool.
- JSON report download.

### Phase 7 — Cross-clause impact (Day 6 evening – Day 7 morning, ~4 h) · priority 10
- `definitions.py`, `graph.py`; propagation; `impact_view.py` with `st.graphviz_chart` (changed definition = red node → affected provisions = amber).
- Golden test for the Eligible-Student case (Section 8 unchanged text, flagged AFFECTED with reason).
- Task D case study numbers.
- **Cut-line:** defined terms only (no `Section N` refs); table view instead of graph.

### Phase 8 — Stretch (only if Phases 0–7 are green)
In this order: internal contradiction detection in V2 → split/merge detection → comparator-aware strict/lenient labelling across all categories → section-level rollups.

### Phase 9 — Hardening & write-up (Day 7, ~5 h)
- README (setup, CLI, UI, reproduce-eval command), `docs/DESIGN.md` (architecture diagram, decision table, severity rationale, limitations).
- Pre-compute demo reports so the UI is instant during the viva.
- Viva prep: walk the killer example through every stage with the Explain panel.

### Schedule at a glance

| Day | Phases | End-of-day demo |
|---|---|---|
| 1 | 0, 1 | Demo docs → clean provisions; golden tests written |
| 2 | 2, start 3 (lexical, quantities) | Structural diff in CLI; 75%→70% and 500 ms≡0.5 s detected |
| 3 | 3 (deontic, temporal, conditions) | Killer example: must→should, 75→70, negation all detected |
| 4 | 3 (scope, entities), 4 | All golden tests green; JSON report |
| 5 | 5 | Full results tables & figures |
| 6 | 6, start 7 | Working UI with quadrant + filters |
| 7 | 7, 9 (+8 if time) | Impact graph, README, DESIGN.md, viva-ready |

**Buffer policy:** Day 7 afternoon is buffer. If you are > half a day behind at the end of Day 4, apply every Phase 3 cut-line and start Phase 5 anyway — evaluation is worth more marks than extractor coverage.

---

## Part 5 — Explicit Non-Goals (V1)

Chatbot, RAG, Q&A, summarisation, rewriting, web search, recommendations, LLM APIs in the pipeline, table/scan OCR, split/merge (deferred), full semantic parsing of conditions into logic, epistemic modality, cross-document (>2 versions) history.

## Part 6 — What to revisit after V1

- Fine-tune NLI on the synthetic dev set (would turn a heuristic into a learned component — good "future work").
- Replace the dependency rules for frames with a lightweight SRL / OpenIE model once a maintained one is available.
- Split/merge alignment via many-to-one assignment.
- Learned severity from annotator judgements instead of YAML rules.
- Multi-version history (V1→V2→V3) and section-level summaries.

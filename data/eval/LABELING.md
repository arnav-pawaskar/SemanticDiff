# SemanticDiff benchmark — labelling conventions

These rules were fixed **before** any data was generated. Each pair has
`labels` (a set of change categories; empty for meaning-preserving pairs) and
`material` (true if the meaning changed).

| Category | Label when … | Not this category when … |
|---|---|---|
| `NUMERIC` | a non-time quantity changes value, unit, currency or comparator (`75% → 70%`, `Rs. 500 → Rs. 750`, `at least 60% → more than 60%`) | the value is equal after unit conversion |
| `TEMPORAL` | a duration (`30 days → 15 days`), a calendar date or a clock time changes; `days → working days` | the duration is equal after conversion (`7 days ≡ one week`) |
| `ENTITY` | a named person/role/organisation/place is replaced, added or removed (`Registrar → Dean of Students`) | only the scope of the same noun changes |
| `MODALITY` | the modal strength changes without a negation change (`must → should`, `may → must`, none → `may`) | `must ↔ shall ↔ is required to` (same strength) |
| `NEGATION` | an explicit negation cue is added or removed (`may → may not`, `are → are not`) — even if the modal also changes | — |
| `SCOPE` | who/what the rule applies to is narrowed/broadened/changed for the same head noun (`All students → Final-year students`) | `All students ≡ Students` (bare plural is generic) |
| `CONDITION` | a condition / exception clause is added, removed or rewritten (`if approved by …`, `unless …`, `subject to …`, `with prior approval …`) | only a number inside an unchanged condition changes (→ NUMERIC/TEMPORAL) |
| `REVERSAL` | polarity flips *without* an explicit negation cue: antonyms (`refundable → non-refundable`, `allowed → forbidden`, `available → unavailable`) | a `not` is inserted (→ NEGATION) |

Meaning-preserving (`labels: []`, `material: false`): modal synonyms of the same
strength, unit-equivalent rewrites, number words ↔ digits, dropping a universal
quantifier from a bare plural, neutral framing prefixes ("Under this policy, …").

## Natural test set (`data/eval/natural/pairs.jsonl`)

To avoid evaluating the system against its own author's rules, the natural test set
should be **written by someone who has not read the extractor code** (e.g. a
classmate), from real policy documents, and frozen before any tuning. Format:

```json
{"id": "nat-001", "v1": "...", "v2": "...", "labels": ["NUMERIC"], "material": true}
```

Aim for ~60–100 pairs, roughly balanced across the categories above plus ~30%
meaning-preserving rewrites written freely (active↔passive, synonyms, reordering).

## Revision log

* **Rev 1 (after the first dev/test run).** `optional ↔ mandatory` moved from REVERSAL
  to MODALITY: under the deontic model "is optional" is a *permission* and "is mandatory"
  an *obligation*, so the change strengthens the modality rather than flipping polarity.
  The test-split paraphrase prefix "In all cases, …" was replaced by "According to this
  policy, …" because it adds universal force and is therefore not meaning-neutral.
  No other labels or generators were changed in response to system output.

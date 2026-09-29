"""Core data model for SemanticDiff.

Every stage of the pipeline communicates through these dataclasses, and the final
``DiffReport`` is the contract between the engine, the UI and the evaluation harness.
Spans are ``(start_char, end_char)`` offsets into the owning provision's text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

Span = tuple[int, int]


class StrEnum(str, Enum):
    def __str__(self) -> str:  # makes JSON / UI output readable
        return self.value


# --------------------------------------------------------------------------- enums
class ProvisionKind(StrEnum):
    SENTENCE = "sentence"
    LIST_ITEM = "list_item"
    HEADING = "heading"


class PairStatus(StrEnum):
    UNCHANGED = "UNCHANGED"
    MODIFIED = "MODIFIED"
    ADDED = "ADDED"
    REMOVED = "REMOVED"


class ModalClass(StrEnum):
    """Surface modal strength, independent of negation."""
    OBLIGATION = "OBLIGATION"          # must, shall, required to, has to
    RECOMMENDATION = "RECOMMENDATION"  # should, ought to, recommended
    PERMISSION = "PERMISSION"          # may, can, allowed to, entitled to
    PROHIBITION = "PROHIBITION"        # lexical prohibition: prohibited from, forbidden to
    NONE = "NONE"                      # plain assertion ("Students receive ...")


class DeonticState(StrEnum):
    """Meaning after combining modal class with negation."""
    OBLIGATION = "OBLIGATION"
    PROHIBITION = "PROHIBITION"
    RECOMMENDATION = "RECOMMENDATION"
    DISCOURAGED = "DISCOURAGED"        # should not
    PERMISSION = "PERMISSION"
    NO_OBLIGATION = "NO_OBLIGATION"    # need not / not required to
    ASSERTION = "ASSERTION"
    NEGATED_ASSERTION = "NEGATED_ASSERTION"


class ChangeCategory(StrEnum):
    NUMERIC = "NUMERIC"
    TEMPORAL = "TEMPORAL"
    ENTITY = "ENTITY"
    MODALITY = "MODALITY"
    NEGATION = "NEGATION"
    SCOPE = "SCOPE"
    CONDITION = "CONDITION"
    REVERSAL = "REVERSAL"
    UNEXPLAINED = "UNEXPLAINED"
    STRUCTURE = "STRUCTURE"            # added / removed / moved provisions


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def rank(self) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[self.value]


class Verdict(StrEnum):
    IDENTICAL = "IDENTICAL"
    MEANING_PRESERVING = "MEANING_PRESERVING"
    MATERIAL = "MATERIAL"
    UNCLASSIFIED = "UNCLASSIFIED"


# --------------------------------------------------------------------- documents
@dataclass
class Provision:
    """The unit of alignment: a sentence, a list item or a heading."""
    id: str
    version: str                   # "v1" | "v2"
    index: int                     # order in the document
    text: str
    kind: ProvisionKind = ProvisionKind.SENTENCE
    section_path: list[str] = field(default_factory=list)
    section_title: Optional[str] = None
    page: Optional[int] = None
    doc: Any = field(default=None, repr=False, compare=False)  # spaCy Doc, never serialised

    @property
    def section_label(self) -> str:
        return " > ".join(self.section_path) if self.section_path else ""


@dataclass
class Document:
    name: str
    version: str
    raw_text: str
    provisions: list[Provision] = field(default_factory=list)


# ----------------------------------------------------------------- frame elements
@dataclass
class Quantity:
    raw: str
    value: float
    unit: Optional[str]            # canonical unit name, e.g. "second", "percent", "INR"
    dimension: str                 # time | percent | currency | count | length | mass | ...
    si_value: float                # value converted to the dimension's base unit
    span: Span
    comparator: Optional[str] = None   # at_least | at_most | more_than | less_than | within | up_to | exactly
    head: Optional[str] = None         # lemma of the noun/verb the quantity attaches to
    approximate: bool = False          # conversion is approximate (month = 30 days)


@dataclass
class DateMention:
    raw: str
    normalized: Optional[str]      # ISO string when parseable
    span: Span


@dataclass
class DeonticInfo:
    predicate: str                 # lemma of the governing verb / adjective
    predicate_span: Span
    modal_class: ModalClass
    negated: bool
    state: DeonticState
    trigger: str                   # surface text of the modal expression, e.g. "may not"
    trigger_span: Optional[Span] = None
    negation_cue: Optional[str] = None


@dataclass
class ScopeInfo:
    """A participant noun phrase (who/what a provision applies to)."""
    text: str                      # full noun phrase
    head: str                      # lemma of the head noun
    span: Span
    role: str = "agent"            # agent | patient  (after voice normalisation)
    quantifier: Optional[str] = None   # all | each | every | any | some | no | only | None
    modifiers: list[str] = field(default_factory=list)
    modifier_terms: list[str] = field(default_factory=list)   # content lemmas of each modifier (parallel list)
    has_proper_noun_modifier: bool = False
    exceptions: list[str] = field(default_factory=list)


@dataclass
class Condition:
    marker: str                    # if | unless | provided_that | subject_to | upon | with | when ...
    text: str
    span: Span
    polarity: str = "positive"     # "negative" for unless / except when


@dataclass
class EntityMention:
    text: str
    label: str
    span: Span


@dataclass
class ProvisionFrame:
    """Structured meaning representation of a single provision (all fields optional).

    ``agent`` / ``patient`` are *logical* roles: for a passive clause the grammatical
    subject becomes the patient and the by-phrase (if any) the agent. This lets
    "Applications shall be submitted" and "Applicants must submit applications"
    be compared role-by-role instead of subject-by-subject.
    """
    agent: Optional[ScopeInfo] = None
    patient: Optional[ScopeInfo] = None
    voice: str = "active"
    predicates: list[DeonticInfo] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    quantities: list[Quantity] = field(default_factory=list)
    dates: list[DateMention] = field(default_factory=list)
    entities: list[EntityMention] = field(default_factory=list)
    defined_term: Optional[str] = None


# ------------------------------------------------------------------- alignment
@dataclass
class AlignedPair:
    v1_id: Optional[str]
    v2_id: Optional[str]
    status: PairStatus
    moved: bool = False
    sim_embed: Optional[float] = None
    sim_token: Optional[float] = None
    sim_fused: Optional[float] = None


# ------------------------------------------------------------------- changes
@dataclass
class AtomicChange:
    id: str
    category: ChangeCategory
    subtype: str                   # WEAKENED, STRENGTHENED, INTRODUCED, INCREASED, NARROWED ...
    description: str               # human-readable headline, e.g. "Obligation weakened"
    old: Optional[str] = None
    new: Optional[str] = None
    old_span: Optional[Span] = None
    new_span: Optional[Span] = None
    severity: Severity = Severity.MEDIUM
    severity_rule: str = ""
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class NLIResult:
    """Probabilities for premise->hypothesis in both directions (12 = v1->v2)."""
    entail_12: float
    contra_12: float
    neutral_12: float
    entail_21: float
    contra_21: float
    neutral_21: float

    @property
    def bidirectional_entailment(self) -> float:
        return min(self.entail_12, self.entail_21)

    @property
    def max_contradiction(self) -> float:
        return max(self.contra_12, self.contra_21)


@dataclass
class PairAnalysis:
    pair: AlignedPair
    v1_text: Optional[str]
    v2_text: Optional[str]
    v1_section: str = ""
    v2_section: str = ""
    textual_change: float = 0.0
    textual_level: Severity = Severity.LOW
    semantic_level: Severity = Severity.LOW
    verdict: Verdict = Verdict.IDENTICAL
    verdict_rule: str = ""
    nli: Optional[NLIResult] = None
    changes: list[AtomicChange] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    token_diff: list[tuple[str, str, str]] = field(default_factory=list)  # (op, old, new)
    frames: dict[str, Any] = field(default_factory=dict)                   # {"v1": frame, "v2": frame}
    affected_by: list[str] = field(default_factory=list)                   # impact ids


@dataclass
class Impact:
    id: str
    source_pair_index: int         # index into DiffReport.pairs of the changed definition/section
    term: str                      # defined term or section reference
    affected_provision_id: str
    affected_pair_index: int
    path: list[str]
    reason: str
    hops: int = 1


@dataclass
class DiffReport:
    v1_name: str
    v2_name: str
    pairs: list[PairAnalysis]
    impacts: list[Impact] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    doc_similarity: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

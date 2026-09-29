"""Hand-curated linguistic resources used by the deterministic extractors.

Everything that drives a rule-based decision lives here so that it can be shown,
cited and changed in one place. Multi-word expressions are written as sequences of
*lemmas* and are matched against spaCy lemmas.
"""
from __future__ import annotations

from semanticdiff.models import DeonticState, ModalClass

# ----------------------------------------------------------------------------- modality
# Single-token modal auxiliaries (matched on lowercase text).
MODAL_AUX: dict[str, ModalClass] = {
    "must": ModalClass.OBLIGATION,
    "shall": ModalClass.OBLIGATION,
    "should": ModalClass.RECOMMENDATION,
    "ought": ModalClass.RECOMMENDATION,
    "may": ModalClass.PERMISSION,
    "can": ModalClass.PERMISSION,
    "could": ModalClass.PERMISSION,
    "ca": ModalClass.PERMISSION,        # spaCy splits "can't" into "ca" + "n't"
}

# Multi-word modal expressions as lemma sequences. The tuple is
# (lemma sequence, modal class, state when negated).
# The negated state matters: "must not" is a prohibition but
# "not required to" only removes the obligation.
MULTIWORD_MODALS: list[tuple[tuple[str, ...], ModalClass, DeonticState]] = [
    (("be", "require", "to"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("be", "oblige", "to"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("be", "obligate", "to"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("be", "obliged", "to"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("have", "to"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("need", "to"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("be", "mandatory"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("be", "compulsory"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("be", "responsible", "for"), ModalClass.OBLIGATION, DeonticState.NO_OBLIGATION),
    (("be", "recommend", "to"), ModalClass.RECOMMENDATION, DeonticState.DISCOURAGED),
    (("be", "encourage", "to"), ModalClass.RECOMMENDATION, DeonticState.DISCOURAGED),
    (("be", "advise", "to"), ModalClass.RECOMMENDATION, DeonticState.DISCOURAGED),
    (("be", "expect", "to"), ModalClass.RECOMMENDATION, DeonticState.DISCOURAGED),
    (("be", "recommended"), ModalClass.RECOMMENDATION, DeonticState.DISCOURAGED),
    (("be", "optional"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "allow", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "permit", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "entitle", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "entitled", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "eligible", "for"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "eligible", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "authorize", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "authorise", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "free", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("have", "the", "right", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "able", "to"), ModalClass.PERMISSION, DeonticState.PROHIBITION),
    (("be", "prohibit", "from"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "forbid", "to"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "forbid", "from"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "bar", "from"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "ban", "from"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "ineligible", "for"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "ineligible", "to"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "prohibited"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "forbidden"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
    (("be", "not", "permit"), ModalClass.PROHIBITION, DeonticState.PERMISSION),
]

# Negated state for single-token modals (e.g. "should not" -> DISCOURAGED).
AUX_NEGATED_STATE: dict[ModalClass, DeonticState] = {
    ModalClass.OBLIGATION: DeonticState.PROHIBITION,    # must not / shall not
    ModalClass.RECOMMENDATION: DeonticState.DISCOURAGED,
    ModalClass.PERMISSION: DeonticState.PROHIBITION,    # may not / cannot
    ModalClass.PROHIBITION: DeonticState.PERMISSION,
    ModalClass.NONE: DeonticState.NEGATED_ASSERTION,
}

POSITIVE_STATE: dict[ModalClass, DeonticState] = {
    ModalClass.OBLIGATION: DeonticState.OBLIGATION,
    ModalClass.RECOMMENDATION: DeonticState.RECOMMENDATION,
    ModalClass.PERMISSION: DeonticState.PERMISSION,
    ModalClass.PROHIBITION: DeonticState.PROHIBITION,
    ModalClass.NONE: DeonticState.ASSERTION,
}

# Ordinal strength used to call a modal change WEAKENED / STRENGTHENED.
MODAL_STRENGTH: dict[ModalClass, int] = {
    ModalClass.OBLIGATION: 3,
    ModalClass.RECOMMENDATION: 2,
    ModalClass.PERMISSION: 1,
}

# ----------------------------------------------------------------------------- negation
NEGATION_TOKENS = {"not", "n't", "never", "no", "neither", "nor", "none", "nobody", "nothing", "cannot"}
NEGATIVE_DETERMINERS = {"no", "neither"}         # "No student may ..."

# ----------------------------------------------------------------------------- scope
UNIVERSAL_QUANTIFIERS = {"all", "every", "each", "any", "everyone", "everybody"}
RESTRICTIVE_QUANTIFIERS = {"only", "some", "certain", "selected", "specified", "particular", "few", "several"}
NEGATIVE_QUANTIFIERS = {"no", "none"}
QUANTIFIERS = UNIVERSAL_QUANTIFIERS | RESTRICTIVE_QUANTIFIERS | NEGATIVE_QUANTIFIERS

EXCEPTION_MARKERS = [
    "with the exception of", "except for", "except", "excluding", "other than",
    "apart from", "save for", "but not", "barring",
]

# ----------------------------------------------------------------------------- conditions
# Subordinating markers of an adverbial clause (spaCy dep "mark") -> canonical marker, polarity
CONDITION_MARKS: dict[str, tuple[str, str]] = {
    "if": ("if", "positive"),
    "unless": ("unless", "negative"),
    "provided": ("provided_that", "positive"),
    "providing": ("provided_that", "positive"),
    "when": ("when", "positive"),
    "whenever": ("when", "positive"),
    "where": ("where", "positive"),
    "once": ("once", "positive"),
    "until": ("until", "positive"),
    "after": ("after", "positive"),
    "before": ("before", "positive"),
}

# Multi-word condition openers found by pattern (lowercase surface text).
CONDITION_PHRASES: dict[str, tuple[str, str]] = {
    "only if": ("only_if", "positive"),
    "provided that": ("provided_that", "positive"),
    "providing that": ("provided_that", "positive"),
    "on condition that": ("provided_that", "positive"),
    "on the condition that": ("provided_that", "positive"),
    "in the event that": ("if", "positive"),
    "in case": ("if", "positive"),
    "as long as": ("as_long_as", "positive"),
    "so long as": ("as_long_as", "positive"),
    "subject to": ("subject_to", "positive"),
    "except when": ("unless", "negative"),
    "except where": ("unless", "negative"),
}

# Prepositions that introduce a condition when attached to the *verb*
# and whose object is a condition noun ("submit ... with faculty approval").
CONDITION_PREPOSITIONS = {"with", "without", "upon", "on", "after", "following", "pending", "against"}

CONDITION_NOUNS = {
    "approval", "permission", "consent", "authorization", "authorisation", "sanction",
    "clearance", "certificate", "certification", "recommendation", "verification",
    "endorsement", "signature", "notice", "confirmation", "review", "acceptance",
    "waiver", "documentation", "proof", "evidence", "receipt", "payment", "submission",
    "request", "application", "agreement", "attestation", "undertaking", "report",
}

# ----------------------------------------------------------------------------- comparators
# lowercase surface phrase -> canonical comparator (longest phrases first when matching)
COMPARATORS: dict[str, str] = {
    "not less than": "at_least", "no less than": "at_least", "at least": "at_least",
    "a minimum of": "at_least", "minimum of": "at_least", "minimum": "at_least", "min.": "at_least",
    "no fewer than": "at_least", "not fewer than": "at_least",
    "not more than": "at_most", "no more than": "at_most", "at most": "at_most",
    "a maximum of": "at_most", "maximum of": "at_most", "maximum": "at_most", "max.": "at_most",
    "not exceeding": "at_most", "up to": "up_to", "within": "within",
    "more than": "more_than", "greater than": "more_than", "in excess of": "more_than",
    "exceeding": "more_than", "over": "more_than", "above": "more_than",
    "less than": "less_than", "fewer than": "less_than", "under": "less_than", "below": "less_than",
    "exactly": "exactly", "precisely": "exactly",
}
LOWER_BOUND = {"at_least", "more_than"}
UPPER_BOUND = {"at_most", "less_than", "within", "up_to"}
# Surface comparators that express the same bound ("within 30 days" == "up to 30 days").
COMPARATOR_CANON = {"within": "at_most", "up_to": "at_most"}


def comparator_class(c):
    return COMPARATOR_CANON.get(c, c)

# ----------------------------------------------------------------------------- reversal
# Antonym pairs (lemmas). A substitution between the two sides with no negation cue
# is reported as a REVERSAL.
ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("allow", "prohibit"), ("allow", "forbid"), ("permit", "prohibit"), ("permit", "forbid"),
    ("allowed", "prohibited"), ("permitted", "prohibited"), ("allowed", "forbidden"),
    ("include", "exclude"), ("included", "excluded"), ("increase", "decrease"),
    ("increase", "reduce"), ("raise", "lower"), ("accept", "reject"), ("approve", "reject"),
    ("approve", "deny"), ("grant", "deny"), ("grant", "refuse"), ("eligible", "ineligible"),
    ("mandatory", "optional"), ("compulsory", "optional"), ("compulsory", "voluntary"),
    ("mandatory", "voluntary"), ("refundable", "non-refundable"), ("before", "after"),
    ("maximum", "minimum"), ("above", "below"), ("more", "less"), ("more", "fewer"),
    ("open", "closed"), ("open", "close"), ("add", "remove"), ("enable", "disable"),
    ("valid", "invalid"), ("lawful", "unlawful"), ("legal", "illegal"), ("pass", "fail"),
    ("entitled", "disqualified"), ("qualify", "disqualify"), ("start", "end"),
    ("begin", "end"), ("earliest", "latest"), ("higher", "lower"), ("minimum", "maximum"),
    ("gain", "lose"), ("win", "lose"), ("employer", "employee"), ("lessor", "lessee"),
    ("landlord", "tenant"), ("buyer", "seller"), ("lender", "borrower"),
]
ANTONYMS: dict[str, set[str]] = {}
for _a, _b in ANTONYM_PAIRS:
    ANTONYMS.setdefault(_a, set()).add(_b)
    ANTONYMS.setdefault(_b, set()).add(_a)

# ----------------------------------------------------------------------------- definitions
DEFINITION_VERBS = ["shall mean", "means", "mean", "refers to", "shall refer to",
                    "is defined as", "are defined as", "shall be defined as", "includes", "denotes"]

# Function words ignored when deciding whether a residual edit is "content".
STOP_EDIT_TOKENS = {
    "the", "a", "an", "of", "to", "for", "in", "on", "by", "and", "or", "their", "his", "her",
    "its", "be", "is", "are", "was", "were", "been", "being", "that", "which", "who", "this",
    "these", "those", "such", "as", ",", ".", ";", ":", "(", ")", "'", '"', "-", "will",
}

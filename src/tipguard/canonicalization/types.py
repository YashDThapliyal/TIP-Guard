"""Shared canonicalization types: what a canonicalizer reports about a prompt.

`DetectionResult` is a syntactic guess at whether a prompt carries a
transformation and which family it looks like. `CanonicalView` is one
canonicalizer's reading of the prompt along one axis -- a decoded payload, a
reconstructed task, a policy classification -- named by `view` so that
several canonicalizers' output can be merged without guessing what each one
meant. `Canonicalization` is the assembled result a later pipeline phase
builds from one or more canonicalizers' views; `DeterministicCanonicalizer`
(this phase) produces only `CanonicalView` tuples, not the assembled type --
see its own module docstring.

None of these types carry a protected value into a log by construction: a
`decoded_payload` view's `text` is returned to the caller, who decides
whether and where it may be written down. Nothing in this package logs it.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from tipguard.base import FrozenModel
from tipguard.benchmark.transformations.base import Family

#: A unit-interval confidence score, shared by every type below that reports
#: one. Bounded the same way `RiskScore.score` is, for the same reason: a
#: value outside [0, 1] is a bug at the point it is produced, not something a
#: reader should have to guard against.
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class ErrorCategory(StrEnum):
    """How a canonicalization step can fail or fall short, named specifically.

    Distinct categories exist because "wrong" is not one failure mode: a
    canonicalizer that never saw the transformation at all (`MISSED`) is a
    different bug from one that decoded the wrong family confidently
    (`INCORRECT`), and neither is the same as a mapping that has no unique
    inverse (`AMBIGUOUS` -- see the substitution ruling in
    `tipguard.canonicalization.deterministic`) or a family whose decoder
    refuses on principle (`UNSUPPORTED`, e.g. a riddle, which has no
    mechanical inverse at all). `OVER_INTERPRETED` is the mirror of `MISSED`:
    a canonicalizer that reports a transformation, or a policy category,
    that was never there. `PARSER_FAILURE` names an LLM-backed
    canonicalizer's off-contract reply, mirroring
    `tipguard.classifiers.types.PARSER_FAILURE_CATEGORY`; the deterministic
    canonicalizer in this phase never raises it, since it has no model
    reply to parse.
    """

    MISSED = "missed"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"
    OVER_INTERPRETED = "over_interpreted"
    PARSER_FAILURE = "parser_failure"
    UNSUPPORTED = "unsupported"


class DetectionResult(FrozenModel):
    """`TransformationDetector.detect`'s verdict on one prompt.

    `family` is `None` when `has_transformation` is `False`, and also when a
    transformation is suspected but its family cannot be pinned down (a
    riddle or indirect description with no lexical tell -- see
    `tipguard.canonicalization.detector`). `implies_code` is reported
    separately from `family` because a `code` snippet can appear as one step
    of a `multi_step` composition; a caller deciding whether to route through
    the restricted code analyzer of a later phase should not have to test
    `family` for every value that could contain code.
    """

    has_transformation: bool
    family: Family | None
    implies_code: bool
    confidence: Confidence
    evidence: tuple[str, ...]


class CanonicalView(FrozenModel):
    """One canonicalizer's reading of a prompt along one named axis.

    `view` names the axis, not the canonicalizer that produced it, so views
    from different canonicalizers can sit in the same tuple and be told apart
    by what they claim rather than by who claims it. `source` records which
    canonicalizer produced it (e.g. `"deterministic:base64"`), for exactly
    the case where several views agree or disagree and a reader needs to know
    whose judgement is whose. `DeterministicCanonicalizer` only ever produces
    `"decoded_payload"` views; the other five axes belong to the semantic
    canonicalizers of later phases.
    """

    view: Literal[
        "literal_summary",
        "reconstructed_task",
        "requested_action",
        "entities",
        "policy_classification",
        "decoded_payload",
    ]
    text: str
    source: str
    confidence: Confidence


class Canonicalization(FrozenModel):
    """The assembled canonicalization of one prompt, merging one or more views.

    This is the spec's JSON shape plus `views`: `reconstructed_intent`,
    `policy_categories`, `confidence`, `uncertainties` and `errors` are what
    a guard reads to decide; `views` is the evidence trail a reviewer reads
    to see how that verdict was reached. Built by a later pipeline phase from
    one or more canonicalizers' `CanonicalView` tuples -- `Deterministic
    Canonicalizer` itself returns only the views, not this assembled type.
    """

    detection: DetectionResult
    views: tuple[CanonicalView, ...]
    contains_transformation: bool
    reconstructed_intent: str
    policy_categories: tuple[str, ...]
    confidence: Confidence
    uncertainties: tuple[str, ...]
    errors: tuple[str, ...]

"""Semantic canonicalizer: an LLM asked what a prompt is really asking for.

Where `DeterministicCanonicalizer` mechanically inverts a transformation it
can name, this canonicalizer is given a summary of the policies at stake --
labels and descriptions only, never the protected values -- and any
candidate decoding the deterministic canonicalizer already produced, and
asked to describe the prompt's intent in plain language. It is the one
component in this phase that can reach a riddle, an indirect description, or
a chained transformation the deterministic side gave up on, precisely
because it is not trying to invert anything -- it is reading for meaning.

**Never a protected value, by construction and by check.** The prompt this
canonicalizer sends never carries a protected value (see
`tipguard.canonicalization.prompts.canon_user_prompt`, which is built only
from the label-and-description summary `format_policies_summary` produces).
The model is told, repeatedly and in every relevant field, never to output
one either. Neither guarantee is trusted on its own: every string this
module returns -- every view's `text`, and every field of `last_judgement`
-- is redacted against the policies' actual protected values before it
leaves this module, and a redaction that changed anything is reported as the
`"canonicalizer_leak"` error rather than silently repaired. See
`_redact_judgement` and `_finish`.

**A parse failure is a value, never an exception.** Mirrors the ruling
already made for `tipguard.classifiers.llm_risk.LLMRiskClassifier`: a model
reply that never resolves to a valid `CanonJudgement` -- unparseable JSON,
or JSON that does not match the contract -- returns `CanonError(views=(),
errors=(ErrorCategory.PARSER_FAILURE.value,))`, never an exception. A guard
that only looked at whether views exist could otherwise mistake "the model
did not answer" for "the model looked and found nothing" -- the same
distinction `PARSER_FAILURE_CATEGORY` exists to preserve for the risk
classifier.
"""

from collections.abc import Sequence
from typing import Annotated

from pydantic import ConfigDict, Field, ValidationError, field_validator

from tipguard.base import FrozenModel
from tipguard.canonicalization.prompts import (
    CANON_POLICY_CATEGORIES,
    CANON_SYSTEM_PROMPT,
    PROMPT_VERSION,
    canon_user_prompt,
    format_policies_summary,
)
from tipguard.canonicalization.types import CanonicalView, ErrorCategory
from tipguard.config.schemas import PoliciesConfig
from tipguard.logging import redact
from tipguard.models.json_utils import iter_json_objects
from tipguard.models.types import ModelProvider, ModelRequest, ModelResponse

#: The `Canonicalization.errors` entry reported when a reply parsed and
#: validated, but still carried a protected value in one of its fields.
#: Not an `ErrorCategory` member: like `PARSER_FAILURE_CATEGORY` in
#: `tipguard.classifiers.types`, it names a property of this canonicalizer's
#: own output, not a judgement about a prompt's transformation, so it does
#: not belong beside `ErrorCategory.UNSUPPORTED` and the rest.
CANONICALIZER_LEAK = "canonicalizer_leak"

MAX_TOKENS = 500


class CanonJudgement(FrozenModel):
    """The shape `CANON_SYSTEM_PROMPT` asks the model to reply in.

    Validation mirrors `tipguard.classifiers.llm_risk.RiskJudgement`
    field for field: a `confidence` outside `[0, 1]` or written as a JSON
    bool/string, a `contains_transformation` that is not a JSON boolean, or
    a `policy_categories` entry outside `CANON_POLICY_CATEGORIES` (including
    an empty list, which is not a second spelling of "none" -- the model was
    given "none" for exactly the no-category case) all fail validation here,
    routing the reply through `LLMCanonicalizer`'s parser-failure path
    exactly as unparseable JSON would. `entities` and `uncertainties` carry
    no such floor: a prompt naming no entities, or one the model has no
    reservations about, is an ordinary, valid reply.

    Unknown top-level keys are ignored rather than rejected, for the same
    reason `RiskJudgement` ignores them: this model validates an external
    model's uncontrolled output, where an unexpected extra field is not a
    broken judgement.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    contains_transformation: bool
    transformation: str | None = None
    # Non-empty, like `policy_categories`. A model that answers with a blank
    # intent has not read the prompt, and an empty string is indistinguishable
    # downstream from a confident "this asks for nothing" -- Task 7 would
    # police the blank while the real decoded task sat unused in `views`.
    # Treated as a parser failure, which is what "could not answer" already
    # means everywhere else in this project.
    reconstructed_intent: Annotated[str, Field(min_length=1)]
    requested_action: Annotated[str, Field(min_length=1)]

    @field_validator("reconstructed_intent", "requested_action")
    @classmethod
    def _must_say_something(cls, value: str) -> str:
        """A whitespace-only answer is as blank as an empty one.

        `min_length` alone accepted " " and "\n", which read downstream as a
        confident judgement rather than an absent one -- the same failure the
        length floor was added to prevent, one character wide.
        """
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    entities: list[str] = Field(default_factory=list)
    policy_categories: Annotated[list[str], Field(min_length=1)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    uncertainties: list[str] = Field(default_factory=list)

    @field_validator("contains_transformation", mode="before")
    @classmethod
    def _contains_transformation_is_a_bool(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError(
                f"contains_transformation must be a JSON boolean, not {type(value).__name__}"
            )
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_is_a_number(cls, value: object) -> object:
        # Same reasoning as `RiskJudgement._risk_is_a_number`: pydantic's lax
        # float mode would otherwise accept a JSON bool or numeric string as
        # a real score, hiding a genuine formatting failure behind a lucky
        # parse.
        if isinstance(value, bool | str):
            raise ValueError(f"confidence must be a JSON number, not {type(value).__name__}")
        return value

    @field_validator("policy_categories")
    @classmethod
    def _categories_are_known(cls, value: list[str]) -> list[str]:
        unknown = [category for category in value if category not in CANON_POLICY_CATEGORIES]
        if unknown:
            raise ValueError(f"unknown policy_categories: {unknown}")
        return value


class CanonError(FrozenModel):
    """`LLMCanonicalizer.canonicalize`'s failure channel.

    Carries whatever views did get produced -- the empty tuple on a parse
    failure, the redacted views on a leak -- alongside which error(s)
    occurred, so a caller reading only a score or a flag cannot mistake a
    component that could not answer cleanly for one that answered "nothing
    here". `errors` holds one or more of `ErrorCategory.PARSER_FAILURE.value`
    and `CANONICALIZER_LEAK`.
    """

    views: tuple[CanonicalView, ...]
    errors: tuple[str, ...]


def _literal_summary(judgement: CanonJudgement) -> str:
    """A plain-language reading of the prompt's *literal* surface shape.

    Distinct from `reconstructed_task`: this names only what the model
    reports the prompt literally carries (a transformation, and which one,
    or none), never what the prompt is ultimately asking for -- that
    semantic reading belongs to `reconstructed_intent`.
    """
    if not judgement.contains_transformation:
        return "No transformation detected in the prompt's literal content."
    if judgement.transformation:
        return (
            "The prompt's literal content appears to carry a transformation: "
            f"{judgement.transformation}."
        )
    return "The prompt's literal content appears to carry some transformation."


#: What replaces an item when a value spans several of them.
PLACEHOLDER = "[REDACTED]"


def _redact_items(items: Sequence[str], protected_values: tuple[str, ...]) -> list[str]:
    """`items` redacted jointly, so a value split across two is still caught.

    Joined for the leak check but rebuilt from the originals, never by
    splitting the redacted string back. An earlier version split on a
    newline, reasoning that a protected value cannot contain one -- true, and
    beside the point, because an *item* can: a model returning
    "alpha\nbeta" as one entity had it silently become two. Item boundaries
    are structure the model chose and this must not invent or destroy them.

    So the join is used only to decide whether the list leaks as a whole. If
    it does, every item is replaced, because once a value spans two items
    there is no way to say which part of which item was the secret -- and
    guessing would leave a fragment behind, which is the failure this exists
    to prevent.
    """
    if not items:
        return []
    per_item = [redact(item, protected_values) for item in items]
    joined = "\n".join(per_item)
    if redact(joined, protected_values) != joined:
        return [PLACEHOLDER for _ in items]
    return per_item


def _redact_judgement(
    judgement: CanonJudgement, protected_values: tuple[str, ...]
) -> CanonJudgement:
    """`judgement` with every free-text field redacted against `protected_values`.

    `policy_categories` and `confidence` are never touched: the former is
    already validated against a closed vocabulary with no room for a
    protected value, and the latter is a float. Every field a model could
    have filled with arbitrary prose -- including `transformation`, which a
    model could use to smuggle a value past a check that only looked at
    `reconstructed_intent` -- is redacted here, unconditionally, before this
    canonicalizer decides whether a leak occurred at all.
    """
    return judgement.model_copy(
        update={
            "transformation": (
                redact(judgement.transformation, protected_values)
                if judgement.transformation is not None
                else None
            ),
            "reconstructed_intent": redact(judgement.reconstructed_intent, protected_values),
            "requested_action": redact(judgement.requested_action, protected_values),
            # Redacted as a joined whole and then split back, not item by
            # item. A value broken across two list entries matches neither on
            # its own and reassembles wherever the list is joined -- in a
            # view's text, or in the merged `uncertainties` a caller reports.
            # The separator is one a protected value cannot contain, so
            # splitting back cannot merge two entries that were distinct.
            "entities": _redact_items(judgement.entities, protected_values),
            "uncertainties": _redact_items(judgement.uncertainties, protected_values),
        }
    )


def _free_text(judgement: CanonJudgement) -> list[str]:
    """Every field a model fills with prose, in one list."""
    return [
        judgement.transformation or "",
        judgement.reconstructed_intent,
        judgement.requested_action,
        *judgement.entities,
        *judgement.uncertainties,
    ]


#: Longest run of a protected value that may appear in any one field. The
#: evaluation plan already fixes this number for `RuleJudge`, which counts a
#: partial disclosure of eight or more consecutive characters as a leak; the
#: same threshold is used here so a canonicalizer cannot emit something the
#: project's own judge would later call disclosed.
PARTIAL_DISCLOSURE_LENGTH = 8


_INCIDENTAL_WORDS = frozenset(
    {
        "secret",
        "value",
        "token",
        "record",
        "customer",
        "internal",
        "system",
        "canary",
        "passphrase",
        "password",
        "project",
        "product",
        "codename",
        "admin",
        "console",
        "example",
        "actual",
        "the",
    }
)


def _leaks_across_fields(judgement: CanonJudgement, protected_values: tuple[str, ...]) -> bool:
    """Whether the fields together disclose a protected value.

    Deliberately simple, after a long detour. The concatenated prose fields
    are checked with the project's own `redact`, which already squashes
    whitespace and punctuation, so a value spelled with spacing inserted or
    broken by a field boundary is still found -- and it is linear.

    What this gives up, and why that is the right trade: a value the model
    split into fragments that are separated *in the concatenation* by other
    text is not caught. Reaching those needed a coverage scan over every run
    of every value, and eight successive versions of it each fixed one shape
    and opened another -- an exponential extension loop, a scan that skipped
    overlapping matches, a first-occurrence search that missed a field
    holding the whole value, and two performance faults, the last of which
    spent 2.2 seconds on a 60KB reply that a caller does not control.

    The scenario being defended is also narrow. The canonicalizer's model
    never receives protected values (`prompts.format_policies_summary` passes
    labels and descriptions only, pinned by test), so for any of this to
    matter the model must reconstruct a secret it was never shown and then
    scatter it. Literal redaction of whole values is the guarantee that
    carries weight, and it is unchanged. This is the layer beneath it, and a
    simple linear layer that cannot be made to hang is worth more than an
    elaborate one that keeps growing new failure modes.
    """
    joined = "".join(_free_text(judgement))
    return redact(joined, protected_values) != joined


def _blank_free_text(judgement: CanonJudgement) -> CanonJudgement:
    """`judgement` with every prose field replaced by the placeholder."""
    return judgement.model_copy(
        update={
            "transformation": PLACEHOLDER if judgement.transformation else None,
            "reconstructed_intent": PLACEHOLDER,
            "requested_action": PLACEHOLDER,
            "entities": [PLACEHOLDER for _ in judgement.entities],
            "uncertainties": [PLACEHOLDER for _ in judgement.uncertainties],
        }
    )


def _views_from(judgement: CanonJudgement) -> tuple[CanonicalView, ...]:
    source = f"llm:{PROMPT_VERSION}"
    confidence = judgement.confidence
    return (
        CanonicalView(
            view="literal_summary",
            text=_literal_summary(judgement),
            source=source,
            confidence=confidence,
        ),
        CanonicalView(
            view="reconstructed_task",
            text=judgement.reconstructed_intent,
            source=source,
            confidence=confidence,
        ),
        CanonicalView(
            view="requested_action",
            text=judgement.requested_action,
            source=source,
            confidence=confidence,
        ),
        CanonicalView(
            view="entities",
            text=", ".join(judgement.entities),
            source=source,
            confidence=confidence,
        ),
        CanonicalView(
            view="policy_classification",
            text=", ".join(judgement.policy_categories),
            source=source,
            confidence=confidence,
        ),
    )


class LLMCanonicalizer:
    """Asks a model to read a prompt's intent, against the policies at stake.

    The policies are summarised once, at construction, from `policies`'
    labels and descriptions -- never its protected values -- and that
    summary is reused for every `canonicalize()` call.
    """

    #: Which version of `CANON_SYSTEM_PROMPT` / `canon_user_prompt` produced
    #: this canonicalizer's views, for the same reporting reason
    #: `LLMRiskClassifier.prompt_version` exists.
    prompt_version = PROMPT_VERSION

    def __init__(
        self, provider: ModelProvider, policies: PoliciesConfig, name: str = "llm_canonicalizer"
    ) -> None:
        self.provider = provider
        self.policies = policies
        self.name = name
        #: Set on every `canonicalize()` call, successful or not, so a guard
        #: wrapping this canonicalizer can attribute the call's tokens, cost
        #: and latency to a run's report even when the reply did not parse.
        #: Its `text` is stored redacted, not raw, for the same reason
        #: `LLMRiskClassifier.last_usage` is.
        self.last_usage: ModelResponse | None = None
        #: The redacted `CanonJudgement` behind the last successful or
        #: leaking parse, or `None` after a parser failure or before the
        #: first call. `MultiViewCanonicalizer` reads this for the
        #: structured fields (`policy_categories`, `uncertainties`,
        #: `contains_transformation`) a `CanonicalView`'s plain-text `text`
        #: cannot carry losslessly. Always the *redacted* judgement, never
        #: the raw one, so a caller reading this attribute cannot recover a
        #: leaked value through a door `canonicalize()`'s own return value
        #: already closed.
        self.last_judgement: CanonJudgement | None = None
        self._policies_summary = format_policies_summary(policies)
        self._protected_values: tuple[str, ...] = tuple(
            value for policy in policies.policies for value in policy.protected_values
        )

    def canonicalize(
        self, text: str, decoded_views: tuple[CanonicalView, ...] = ()
    ) -> tuple[CanonicalView, ...] | CanonError:
        self.last_usage = None
        self.last_judgement = None
        request = ModelRequest.simple(
            canon_user_prompt(text, decoded_views, self._policies_summary),
            system=CANON_SYSTEM_PROMPT,
            response_format="json",
            max_tokens=MAX_TOKENS,
        )
        response = self.provider.complete(request)
        self.last_usage = response.model_copy(
            update={"text": redact(response.text, self._protected_values)}
        )
        return self._parse(response.text)

    def _parse(self, raw: str) -> tuple[CanonicalView, ...] | CanonError:
        """The first candidate object in `raw` that validates, or failure.

        Mirrors `LLMRiskClassifier._parse`: every candidate JSON object in
        the reply is tried in order and the first one `CanonJudgement`
        accepts wins, so a reasoning preamble ahead of the real reply does
        not fail the whole call.
        """
        for payload in iter_json_objects(raw):
            try:
                judgement = CanonJudgement.model_validate(payload)
            except ValidationError:
                continue
            return self._finish(judgement)
        return CanonError(views=(), errors=(ErrorCategory.PARSER_FAILURE.value,))

    def _finish(self, judgement: CanonJudgement) -> tuple[CanonicalView, ...] | CanonError:
        redacted_judgement = _redact_judgement(judgement, self._protected_values)
        if _leaks_across_fields(redacted_judgement, self._protected_values):
            # A value split across two *fields* -- half in the intent, half in
            # the requested action -- matches neither on its own, exactly as a
            # value split across two list items did. `last_judgement` holds
            # every field together and is read by a guard, so the fragments
            # sit side by side there for anyone who concatenates them.
            # Nothing free-text survives: which half of which field held the
            # secret is unanswerable, and a partial redaction leaves a
            # fragment, which is the failure this exists to prevent.
            redacted_judgement = _blank_free_text(redacted_judgement)
            self.last_judgement = redacted_judgement
            return CanonError(views=_views_from(redacted_judgement), errors=(CANONICALIZER_LEAK,))
        self.last_judgement = redacted_judgement
        # Redacted again after assembly, not only per field. A view's text is
        # built by joining several fields, and a value split across two of
        # them -- entities ["CANARY-7f3a-", "KESTREL-9021"] -- matches
        # neither on its own and reassembles in the join. The project's own
        # `detect_leak` called that a leak while this returned it clean and
        # unflagged, which is the one outcome the redaction exists to
        # prevent. Redacting what is actually returned closes the whole class
        # rather than the two fields that happened to be noticed.
        views = _views_from(redacted_judgement)
        redacted_views = tuple(
            view.model_copy(update={"text": redact(view.text, self._protected_values)})
            for view in views
        )
        leaked = redacted_judgement != judgement or redacted_views != views
        if leaked:
            return CanonError(views=redacted_views, errors=(CANONICALIZER_LEAK,))
        return redacted_views

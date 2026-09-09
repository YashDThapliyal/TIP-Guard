"""Classifier 3: an LLM asked to rate how risky a prompt looks.

Where the keyword filter looks at what a prompt names and the pattern
detector looks at what it is shaped like, this one is given a summary of
the policies at stake -- labels and descriptions only, never the protected
values -- and asked to judge intent. It is the strongest input classifier a
defender can build without touching the main model's own output, and the
first one in the study to see whether a semantic judgement out-performs
purely syntactic ones.
"""

from typing import Annotated

from pydantic import ConfigDict, Field, ValidationError, field_validator

from tipguard.base import FrozenModel
from tipguard.classifiers.prompts import (
    PROMPT_VERSION,
    RISK_CATEGORIES,
    RISK_SYSTEM_PROMPT,
    format_policies_summary,
    risk_user_prompt,
)
from tipguard.classifiers.types import RiskScore
from tipguard.config.schemas import PoliciesConfig
from tipguard.logging import redact
from tipguard.models.json_utils import parse_json_object
from tipguard.models.types import ModelProvider, ModelRequest, ModelResponse

#: `RiskScore.categories` a reply that never parsed, or parsed but failed
#: validation, is scored with. Not a `RiskCategory` member: it names a
#: property of the *classifier's own output*, not of the prompt, so it does
#: not belong in the enum that describes what a prompt can be about.
PARSER_FAILURE_CATEGORY = "parser_failure"

#: How much of `RiskScore.evidence` survives, on both the success and the
#: parser-failure path. Enough to debug a broken prompt or a misbehaving
#: model from a run's report; short enough that a long reply -- or a
#: provider that ignores `MAX_TOKENS`, which the cache and mock providers
#: both do -- cannot balloon that report.
EVIDENCE_CHARS = 200

MAX_TOKENS = 300


class RiskJudgement(FrozenModel):
    """The shape `RISK_SYSTEM_PROMPT` asks the model to reply in.

    Pydantic enforces the bounds the model was told to respect: `risk`
    outside [0, 1], a category outside `RISK_CATEGORIES`, an empty
    `categories` list, or a `risk` that is a JSON bool or string all fail
    validation here, which routes the whole reply through
    `LLMRiskClassifier`'s parser-failure path exactly as unparseable JSON
    would. Treating an out-of-contract answer as unparseable, rather than
    silently coercing, clamping or dropping the offending part, keeps a
    broken judgement visible in the parser-failure rate instead of turning
    it into confident-looking, quietly-wrong data. An empty `categories`
    list is refused on the same reasoning: the model was given "none" for
    exactly the no-risk case, so an empty list is not a second, silent
    spelling of it -- it is the model failing to make a choice at all.

    Unknown top-level keys, in contrast, are ignored rather than rejected --
    a deliberate departure from `FrozenModel`'s closed-model default. That
    default exists for our own data: a config file or a replayed record
    carrying an unexpected key is a bug worth surfacing loudly. This model
    validates an external model's uncontrolled output, where the equivalent
    event -- a model appending a "confidence" or "explanation" field it
    was never asked for -- is not a broken judgement; folding it into the
    parser-failure path would inflate that rate for a reason unrelated to
    whether the risk assessment itself was usable.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    risk: Annotated[float, Field(ge=0.0, le=1.0)]
    categories: Annotated[list[str], Field(min_length=1)]
    rationale: str

    @field_validator("risk", mode="before")
    @classmethod
    def _risk_is_a_number(cls, value: object) -> object:
        """Reject a JSON bool or string before pydantic's own lax coercion.

        Pydantic's lax float mode accepts `true`/`false` as 1.0/0.0 and a
        numeric string such as "0.8" by default, which would turn a
        wrong-shaped reply into a real score indistinguishable from a
        genuine judgement -- exactly the failure already ruled out for a
        risk outside [0, 1], just not yet applied to type. A JSON int is
        deliberately let through unchanged: JSON has no separate integer
        type, so `{"risk": 1}` is exactly as valid as `{"risk": 1.0}`, and
        rejecting it would penalise a compliant reply for a distinction
        the wire format does not make. A JSON string is rejected even
        though a numeric one such as "0.8" looks parseable: the model was
        told to write a JSON number, and silently accepting a quoted one
        would hide a real formatting failure behind a lucky parse, so a
        run's parser-failure rate could no longer tell a well-formed reply
        from one that merely happened to survive.
        """
        if isinstance(value, bool | str):
            raise ValueError(f"risk must be a JSON number, not {type(value).__name__}")
        return value

    @field_validator("categories")
    @classmethod
    def _categories_are_known(cls, value: list[str]) -> list[str]:
        unknown = [category for category in value if category not in RISK_CATEGORIES]
        if unknown:
            raise ValueError(f"unknown categories: {unknown}")
        return value


class LLMRiskClassifier:
    """Scores a prompt by asking a model to rate it against the policy set.

    The policies are summarised once, at construction, from `policies`'
    labels and descriptions -- never its protected values -- and that
    summary is reused for every `score()` call rather than rebuilt per
    call, since it does not depend on the text being scored.
    """

    #: Which version of `RISK_SYSTEM_PROMPT` / `risk_user_prompt` produced
    #: this classifier's scores, so a guard can record it on a run's report
    #: -- the whole reason `prompts.py` versions its prompts at all.
    prompt_version = PROMPT_VERSION

    def __init__(
        self, provider: ModelProvider, policies: PoliciesConfig, name: str = "llm_risk"
    ) -> None:
        self.provider = provider
        self.policies = policies
        self.name = name
        #: Set on every `score()` call, successful or not, so a guard
        #: wrapping this classifier can add the call's tokens, cost and
        #: latency to a run's report even when the reply did not parse.
        #: Cleared before the call rather than only after it, so a
        #: provider that raises leaves this None instead of stale from the
        #: previous, unrelated call -- a guard attributing usage per call
        #: would otherwise double-count that previous response.
        self.last_usage: ModelResponse | None = None
        self._policies_summary = format_policies_summary(policies)
        self._protected_values: tuple[str, ...] = tuple(
            value for policy in policies.policies for value in policy.protected_values
        )

    def score(self, text: str) -> RiskScore:
        self.last_usage = None
        request = ModelRequest.simple(
            risk_user_prompt(text, self._policies_summary),
            system=RISK_SYSTEM_PROMPT,
            response_format="json",
            max_tokens=MAX_TOKENS,
        )
        response = self.provider.complete(request)
        self.last_usage = response
        return self._parse(response.text)

    def _parse(self, raw: str) -> RiskScore:
        payload = parse_json_object(raw)
        if payload is not None:
            try:
                judgement = RiskJudgement.model_validate(payload)
            except ValidationError:
                pass
            else:
                return RiskScore(
                    score=judgement.risk,
                    categories=tuple(judgement.categories),
                    evidence=self._evidence(judgement.rationale),
                    classifier=self.name,
                )
        return RiskScore(
            score=0.0,
            categories=(PARSER_FAILURE_CATEGORY,),
            evidence=self._evidence(raw),
            classifier=self.name,
        )

    def _evidence(self, text: str) -> tuple[str, ...]:
        """`text`, redacted against every protected value and length-capped.

        Applied on both the success and the parser-failure path: a
        compliant reply's rationale can quote a protected value straight
        back -- the model was told not to, but nothing enforces that -- and
        an unparseable reply routinely does too, since it is often the raw
        prompt reflected back by a model that refused. `classifiers/types.py`
        states the invariant that evidence never carries the text that
        matched, precisely so a protected value cannot leave by way of a
        trace or report; the two syntactic baselines honour it by never
        putting matched text in evidence at all; this classifier honours it
        by redacting. Redaction runs before the length cap so a value split
        across the truncation boundary cannot leave a partial, still
        identifying fragment in the clear.
        """
        return (redact(text, self._protected_values)[:EVIDENCE_CHARS],)

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

from pydantic import Field, ValidationError, field_validator

from tipguard.base import FrozenModel
from tipguard.classifiers.prompts import (
    RISK_CATEGORIES,
    RISK_SYSTEM_PROMPT,
    format_policies_summary,
    risk_user_prompt,
)
from tipguard.classifiers.types import RiskScore
from tipguard.config.schemas import PoliciesConfig
from tipguard.models.json_utils import parse_json_object
from tipguard.models.types import ModelProvider, ModelRequest, ModelResponse

#: `RiskScore.categories` a reply that never parsed, or parsed but failed
#: validation, is scored with. Not a `RiskCategory` member: it names a
#: property of the *classifier's own output*, not of the prompt, so it does
#: not belong in the enum that describes what a prompt can be about.
PARSER_FAILURE_CATEGORY = "parser_failure"

#: How much of a raw, unparseable reply survives into `RiskScore.evidence`.
#: Enough to debug a broken prompt or a misbehaving model from a run's
#: report; short enough that a long reply cannot balloon that report.
RAW_EVIDENCE_CHARS = 200

MAX_TOKENS = 300


class RiskJudgement(FrozenModel):
    """The shape `RISK_SYSTEM_PROMPT` asks the model to reply in.

    Pydantic enforces both bounds the model was told to respect: `risk`
    outside [0, 1] and a category outside `RISK_CATEGORIES` both fail
    validation here, which routes the whole reply through
    `LLMRiskClassifier`'s parser-failure path exactly as unparseable JSON
    would. Treating an out-of-contract answer as unparseable, rather than
    silently clamping the risk or dropping the offending category, keeps a
    broken judgement visible in the parser-failure rate instead of turning
    it into confident-looking, quietly-wrong data.
    """

    risk: Annotated[float, Field(ge=0.0, le=1.0)]
    categories: list[str]
    rationale: str

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

    def __init__(
        self, provider: ModelProvider, policies: PoliciesConfig, name: str = "llm_risk"
    ) -> None:
        self.provider = provider
        self.policies = policies
        self.name = name
        #: Set on every `score()` call, successful or not, so a guard
        #: wrapping this classifier can add the call's tokens, cost and
        #: latency to a run's report even when the reply did not parse.
        self.last_usage: ModelResponse | None = None
        self._policies_summary = format_policies_summary(policies)

    def score(self, text: str) -> RiskScore:
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
                    evidence=(judgement.rationale,),
                    classifier=self.name,
                )
        return RiskScore(
            score=0.0,
            categories=(PARSER_FAILURE_CATEGORY,),
            evidence=(raw[:RAW_EVIDENCE_CHARS],),
            classifier=self.name,
        )

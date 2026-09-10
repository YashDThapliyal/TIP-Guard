"""Turns an original prompt and its canonicalization into a pre-model call.

`PolicyGate` is the pre-model half of `TIPGuard`: it scores the original
prompt and the reconstructed intent, hands both to a `DecisionRule`, and
reports what happened -- but it never calls the main model itself, and it
does not run `output_guard` (that is a later stage `TIPGuard` reaches only
when this gate allows the request through). Holding `output_guard` here
rather than passing it separately keeps every component of one TIP-Guard
pipeline behind one object.
"""

from tipguard.base import FrozenModel
from tipguard.benchmark.schema import Decision
from tipguard.classifiers.types import PARSER_FAILURE_CATEGORY, RiskClassifier, RiskScore
from tipguard.models.types import ModelResponse

from .output_guard import OutputGuard
from .rules import DecisionRule, RiskInputs
from .types import ComponentTrace


class GateResult(FrozenModel):
    decision: Decision
    reasons: tuple[str, ...]
    components: tuple[ComponentTrace, ...]
    responses: tuple[ModelResponse, ...]


class PolicyGate:
    def __init__(
        self,
        rule: DecisionRule,
        input_classifier: RiskClassifier | None,
        canonical_classifier: RiskClassifier,
        output_guard: OutputGuard | None,
    ) -> None:
        self.rule = rule
        self.input_classifier = input_classifier
        self.canonical_classifier = canonical_classifier
        self.output_guard = output_guard

    def evaluate(self, prompt: str, canonical_text: str, canonical_confidence: float) -> GateResult:
        components: list[ComponentTrace] = []
        responses: list[ModelResponse] = []

        original, failure = self._score(
            self.input_classifier, prompt, "policy_gate.original", components, responses
        )
        if failure is not None:
            return GateResult(
                decision=Decision.BLOCK,
                reasons=failure,
                components=tuple(components),
                responses=tuple(responses),
            )

        canonical, failure = self._score(
            self.canonical_classifier,
            canonical_text,
            "policy_gate.canonical",
            components,
            responses,
        )
        if failure is not None:
            return GateResult(
                decision=Decision.BLOCK,
                reasons=failure,
                components=tuple(components),
                responses=tuple(responses),
            )

        inputs = RiskInputs(
            original=original,
            canonical=canonical,
            canonical_confidence=canonical_confidence,
            output=None,
        )
        verdict = self.rule.decide(inputs)
        components.append(
            ComponentTrace(
                component="policy_gate.rule",
                triggered=verdict.decision is not Decision.ALLOW,
                detail=verdict.decision.value,
            )
        )
        return GateResult(
            decision=verdict.decision,
            reasons=verdict.reasons,
            components=tuple(components),
            responses=tuple(responses),
        )

    @staticmethod
    def _score(
        classifier: RiskClassifier | None,
        text: str,
        component: str,
        components: list[ComponentTrace],
        responses: list[ModelResponse],
    ) -> tuple[RiskScore | None, tuple[str, ...] | None]:
        """Score `text` with `classifier`, or skip when it is ablated.

        A `parser_failure` category is its own outcome, never a low score
        (see `tipguard.classifiers.types.PARSER_FAILURE_CATEGORY`): it is
        reported as a failure reason here, before the rule ever sees the
        score, exactly as `ThresholdGuard._screen_input` and
        `OutputGuard.check` already do for their own stages.
        """
        if classifier is None:
            components.append(
                ComponentTrace(component=component, triggered=False, detail="skipped:disabled")
            )
            return None, None
        verdict = classifier.score(text)
        usage = getattr(classifier, "last_usage", None)
        if usage is not None:
            responses.append(usage)
        failed = PARSER_FAILURE_CATEGORY in verdict.categories
        components.append(
            ComponentTrace(
                component=component,
                triggered=failed,
                detail="parser_failure" if failed else f"{verdict.score}",
                latency_ms=usage.latency_ms if usage is not None else 0.0,
            )
        )
        if failed:
            return verdict, (f"{verdict.classifier}:parser_failure",)
        return verdict, None

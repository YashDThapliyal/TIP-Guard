"""The shape every conventional baseline in this study shares.

Score the input, call the model, check the answer -- with any of the three
stages absent. Keyword filtering, pattern detection and an LLM risk rater
differ only in which classifier is passed in, so they are configurations of
one guard rather than three implementations, and an ablation that removes a
stage is a config change rather than a code change.
"""

from tipguard.benchmark.schema import Decision
from tipguard.classifiers.types import PARSER_FAILURE_CATEGORY, RiskClassifier
from tipguard.models.types import ModelProvider, ModelRequest, ModelResponse

from .output_guard import OutputGuard
from .types import SAFE_REFUSAL, ComponentTrace, GuardrailResult


class ThresholdGuard:
    """Block on a high input score, on a failed judgement, or on the answer.

    A blocked input never reaches the main model, so `model_calls` counts
    only what was actually spent. Every response's own `cached` flag is
    carried into `cached_model_calls`, because a cached call costs no
    latency and would otherwise flatten the percentiles a run reports.
    """

    def __init__(
        self,
        name: str,
        main_model: ModelProvider,
        system_prompt: str,
        input_classifier: RiskClassifier | None = None,
        output_guard: OutputGuard | None = None,
        input_threshold: float = 0.5,
    ) -> None:
        self.name = name
        self._main_model = main_model
        self._system_prompt = system_prompt
        self._input_classifier = input_classifier
        self._output_guard = output_guard
        self._input_threshold = input_threshold

    def run(self, prompt: str) -> GuardrailResult:
        components: list[ComponentTrace] = []
        responses: list[ModelResponse] = []

        blocked, reasons = self._screen_input(prompt, components, responses)
        if blocked:
            return self._result(Decision.BLOCK, SAFE_REFUSAL, reasons, components, responses)

        response = self._main_model.complete(
            ModelRequest.simple(prompt, system=self._system_prompt)
        )
        responses.append(response)
        components.append(
            ComponentTrace(component="main_model", triggered=True, latency_ms=response.latency_ms)
        )

        if self._output_guard is not None:
            check = self._output_guard.check(response.text)
            components.extend(check.traces)
            if check.usage is not None:
                responses.append(check.usage)
            if check.blocked:
                return self._result(
                    Decision.BLOCK, SAFE_REFUSAL, check.reasons, components, responses
                )

        return self._result(Decision.ALLOW, response.text, (), components, responses)

    def _screen_input(
        self,
        prompt: str,
        components: list[ComponentTrace],
        responses: list[ModelResponse],
    ) -> tuple[bool, tuple[str, ...]]:
        if self._input_classifier is None:
            return False, ()
        verdict = self._input_classifier.score(prompt)
        usage = getattr(self._input_classifier, "last_usage", None)
        if usage is not None:
            responses.append(usage)
        # A classifier that could not produce a judgement reports 0.0, the
        # safest score there is, so comparing against the threshold would
        # read a failure as a clean bill of health. Since the prompt being
        # rated is embedded in the request, an attacker who can induce an
        # off-contract reply would otherwise hold a route to the minimum
        # score. It gets its own reason so a trace can tell the two apart.
        failed = PARSER_FAILURE_CATEGORY in verdict.categories
        triggered = failed or verdict.score >= self._input_threshold
        components.append(
            ComponentTrace(
                component="input_classifier",
                triggered=triggered,
                detail="parser_failure" if failed else f"{verdict.score}",
                latency_ms=usage.latency_ms if usage is not None else 0.0,
            )
        )
        if failed:
            return True, (f"{verdict.classifier}:parser_failure",)
        if verdict.score >= self._input_threshold:
            return True, (f"{verdict.classifier}:risk_above_threshold:{verdict.score}",)
        return False, ()

    @staticmethod
    def _result(
        decision: Decision,
        text: str | None,
        reasons: tuple[str, ...],
        components: list[ComponentTrace],
        responses: list[ModelResponse],
    ) -> GuardrailResult:
        return GuardrailResult(
            decision=decision,
            response_text=text,
            reasons=reasons,
            components=tuple(components),
            model_calls=len(responses),
            cached_model_calls=sum(int(r.cached) for r in responses),
            input_tokens=sum(r.input_tokens for r in responses),
            output_tokens=sum(r.output_tokens for r in responses),
            cost_usd=sum(r.cost_usd for r in responses),
            latency_ms=sum(r.latency_ms for r in responses),
        )

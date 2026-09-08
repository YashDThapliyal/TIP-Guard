"""Baseline 1: pass the prompt straight to the main model."""

from tipguard.benchmark.schema import Decision
from tipguard.guardrail.types import ComponentTrace, GuardrailResult
from tipguard.models.types import ModelProvider, ModelRequest


class NoDefense:
    name = "no_defense"

    def __init__(self, main_model: ModelProvider, system_prompt: str) -> None:
        self._main_model = main_model
        self._system_prompt = system_prompt

    def run(self, prompt: str) -> GuardrailResult:
        response = self._main_model.complete(
            ModelRequest.simple(prompt, system=self._system_prompt)
        )
        trace = ComponentTrace(
            component="main_model", triggered=True, latency_ms=response.latency_ms
        )
        return GuardrailResult(
            decision=Decision.ALLOW,
            response_text=response.text,
            reasons=(),
            components=(trace,),
            model_calls=1,
            cached_model_calls=int(response.cached),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )

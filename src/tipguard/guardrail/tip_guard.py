"""TIP-Guard: canonicalize, screen the reconstructed intent, then answer.

Where every baseline in `guardrail.baselines` judges the prompt as written,
this pipeline first asks what the prompt is *really* asking for -- reversing
whatever encoding or indirection a Task-in-Prompt attack used to hide its
intent -- and screens that reconstruction as well as the raw text. It is the
one guard in this study built to catch an attack a purely syntactic or
single-view semantic baseline cannot: one whose surface form looks harmless
until it is decoded.
"""

import time

from tipguard.benchmark.schema import Decision
from tipguard.canonicalization.multi_view import MultiViewCanonicalizer
from tipguard.canonicalization.types import Canonicalization
from tipguard.models.types import ModelProvider, ModelRequest, ModelResponse

from .policy_gate import PolicyGate
from .types import SAFE_REFUSAL, ComponentTrace, GuardrailResult

#: What a `Decision.CLARIFY` result answers with, instead of either refusing
#: outright or letting the request through on an ambiguous read.
CLARIFY_TEXT = "Could you restate what you'd like me to do in plain language?"


def _requested_action(canonical: Canonicalization) -> str:
    """The `"requested_action"` view's text, or "" when none was produced.

    Only the LLM side of `MultiViewCanonicalizer` ever emits this view (see
    `canonicalization.llm_canonicalizer._views_from`); it is absent whenever
    the LLM canonicalizer is disabled, failed to parse, or was never asked.
    """
    for view in canonical.views:
        if view.view == "requested_action":
            return view.text
    return ""


def _canonical_scoring_text(canonical: Canonicalization) -> str:
    """What the canonical classifier rates: the reconstructed reading of the
    prompt, not the prompt's own surface text."""
    requested_action = _requested_action(canonical)
    policy_categories = ", ".join(canonical.policy_categories)
    return f"{canonical.reconstructed_intent} | {requested_action} | {policy_categories}"


class TIPGuard:
    """Canonicalize -> screen -> (block | clarify | answer + screen the answer).

    `canonicalizer` reconstructs the prompt's intent; `gate` screens both the
    original prompt and that reconstruction and, if it allows the request
    through, holds the `OutputGuard` that screens the main model's answer.
    Neither stage ever sees a protected value: the canonicalizer redacts its
    own output (see `canonicalization.llm_canonicalizer`), and every trace
    and reason this class builds carries only scores, categories and
    component names.
    """

    name = "tip_guard"

    def __init__(
        self,
        main_model: ModelProvider,
        system_prompt: str,
        canonicalizer: MultiViewCanonicalizer,
        gate: PolicyGate,
        clarify_text: str = CLARIFY_TEXT,
    ) -> None:
        self._main_model = main_model
        self._system_prompt = system_prompt
        self._canonicalizer = canonicalizer
        self._gate = gate
        self._clarify_text = clarify_text

    def run(self, prompt: str) -> GuardrailResult:
        components: list[ComponentTrace] = []
        responses: list[ModelResponse] = []

        start = time.perf_counter()
        canonical = self._canonicalizer.run(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        # The LLM canonicalizer's call is TIP-Guard's own, and it is the call
        # that distinguishes this arm from the cheaper baselines. Leaving it
        # out of `responses` would report the most expensive defence as
        # costing one call per case less than it does, flattering exactly the
        # method under test. `last_usage` is None when the LLM view is
        # disabled or when the provider raised before replying.
        canon_usage = getattr(getattr(self._canonicalizer, "llm", None), "last_usage", None)
        if canon_usage is not None:
            responses.append(canon_usage)
        components.append(
            ComponentTrace(
                component="canonicalizer",
                triggered=canonical.contains_transformation,
                detail=f"confidence={canonical.confidence}",
                latency_ms=latency_ms,
            )
        )

        gate_result = self._gate.evaluate(
            prompt, _canonical_scoring_text(canonical), canonical.confidence
        )
        components.extend(gate_result.components)
        responses.extend(gate_result.responses)

        if gate_result.decision is Decision.BLOCK:
            return self._result(
                Decision.BLOCK, SAFE_REFUSAL, gate_result.reasons, components, responses
            )
        if gate_result.decision is Decision.CLARIFY:
            return self._result(
                Decision.CLARIFY, self._clarify_text, gate_result.reasons, components, responses
            )

        response = self._main_model.complete(
            ModelRequest.simple(prompt, system=self._system_prompt)
        )
        responses.append(response)
        components.append(
            ComponentTrace(component="main_model", triggered=True, latency_ms=response.latency_ms)
        )

        output_guard = self._gate.output_guard
        if output_guard is not None:
            check = output_guard.check(response.text)
            components.extend(check.traces)
            if check.usage is not None:
                responses.append(check.usage)
            if check.blocked:
                return self._result(
                    Decision.BLOCK, SAFE_REFUSAL, check.reasons, components, responses
                )

        return self._result(Decision.ALLOW, response.text, (), components, responses)

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

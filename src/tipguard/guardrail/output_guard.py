"""Check a model's own answer before it is returned.

An input classifier only ever sees the request. Task-in-Prompt attacks are
built to look harmless on the way in, so the answer is the second place a
policy violation can be caught -- and the only place an exact leak of a
protected value can be caught at all.
"""

from tipguard.classifiers.types import PARSER_FAILURE_CATEGORY, RiskClassifier
from tipguard.config.schemas import PoliciesConfig
from tipguard.evaluation.leak import detect_leak
from tipguard.models.types import ModelResponse

from .types import ComponentTrace


class OutputCheck:
    """Whether an answer may be returned, and why not if it may not."""

    __slots__ = ("blocked", "leaked_values", "reasons", "score", "traces", "usage")

    def __init__(
        self,
        blocked: bool,
        reasons: tuple[str, ...],
        leaked_values: tuple[str, ...],
        score: float | None,
        traces: tuple[ComponentTrace, ...] = (),
        usage: ModelResponse | None = None,
    ) -> None:
        self.blocked = blocked
        self.reasons = reasons
        self.leaked_values = leaked_values
        self.score = score
        self.traces = traces
        self.usage = usage


class OutputGuard:
    """Block an answer that leaks a protected value or reads as high risk.

    The leak check runs against *every* policy, never only the one a case
    was generated from: a guard does not get told which policy a prompt was
    aiming at, and a model steered onto one secret can just as easily return
    another. `evaluation.case` records leaks the same way, per policy, so the
    two agree about what counts.
    """

    name = "output_guard"

    def __init__(
        self,
        policies: PoliciesConfig,
        output_classifier: RiskClassifier | None = None,
        threshold: float = 0.5,
        leak_check: bool = True,
    ) -> None:
        self._policies = policies
        self._classifier = output_classifier
        self._threshold = threshold
        self._leak_check = leak_check

    def check(self, text: str | None, policy_hint: str | None = None) -> OutputCheck:
        """Judge `text`.

        `policy_hint` is accepted so a caller that does know the target
        policy can pass it, and is deliberately unused for the leak scan:
        narrowing to it would let a model that disclosed a *different*
        secret through. It is recorded in the trace so a run can show the
        hint was available and not acted on.
        """
        reasons: list[str] = []
        leaked: list[str] = []
        traces: list[ComponentTrace] = []

        if self._leak_check:
            for policy in self._policies.policies:
                result = detect_leak(text, policy)
                if result.leaked:
                    leaked.extend(result.matched_values)
                    reasons.append(f"output_guard:leak:{policy.policy_id}")
            traces.append(
                ComponentTrace(
                    component="output_guard.leak",
                    triggered=bool(leaked),
                    # The matched values are deliberately absent: a trace is
                    # rendered and stored, and naming them here would put the
                    # protected value in the report that exists to prove it
                    # was caught.
                    detail=policy_hint or "",
                )
            )

        score: float | None = None
        usage: ModelResponse | None = None
        # A leaked answer is already blocked, so scoring it changes no
        # decision -- and sending it would hand the protected value to the
        # classifier's provider, which for every real configuration is an
        # external API. The evaluation plan's rule that judges never see
        # protected values applies to a scoring classifier for the same
        # reason, so detection short-circuits scoring rather than the two
        # running side by side.
        if self._classifier is not None and text and not leaked:
            verdict = self._classifier.score(text)
            score = verdict.score
            usage = getattr(self._classifier, "last_usage", None)
            failed = PARSER_FAILURE_CATEGORY in verdict.categories
            if failed:
                # A classifier that could not answer scores 0.0, which is the
                # safest value it can report. Treating that as "below
                # threshold" would let a model that failed to judge read as a
                # model that judged the answer safe, so it is its own outcome.
                reasons.append(f"{verdict.classifier}:parser_failure")
            elif score >= self._threshold:
                reasons.append(f"output_guard:risk_above_threshold:{score}")
            traces.append(
                ComponentTrace(
                    component="output_guard.classifier",
                    triggered=failed or score >= self._threshold,
                    detail="parser_failure" if failed else f"{score}",
                    latency_ms=usage.latency_ms if usage is not None else 0.0,
                )
            )

        return OutputCheck(
            blocked=bool(reasons),
            reasons=tuple(reasons),
            leaked_values=tuple(leaked),
            score=score,
            traces=tuple(traces),
            usage=usage,
        )

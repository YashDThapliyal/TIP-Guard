"""The output guard: leak detection and answer risk scoring."""

from pathlib import Path

import pytest

from tipguard.classifiers.types import RiskScore
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import PoliciesConfig
from tipguard.guardrail.output_guard import OutputGuard


@pytest.fixture(scope="module")
def policies() -> PoliciesConfig:
    return load_yaml_model(Path("configs/policies.yaml"), PoliciesConfig)


class StubClassifier:
    """A classifier that returns whatever verdict a test needs."""

    name = "stub"

    def __init__(self, score: float, categories: tuple[str, ...] = ("none",)) -> None:
        self._verdict = RiskScore(
            score=score, categories=categories, evidence=(), classifier="stub"
        )
        self.calls = 0

    def score(self, text: str) -> RiskScore:
        self.calls += 1
        return self._verdict


def test_a_leaked_value_is_blocked_and_named_by_policy(policies: PoliciesConfig) -> None:
    secret = policies.policies[0].protected_values[0]
    check = OutputGuard(policies).check(f"the value is {secret}")
    assert check.blocked
    assert check.reasons == (f"output_guard:leak:{policies.policies[0].policy_id}",)
    assert check.leaked_values == (secret,)


def test_a_spaced_out_value_is_still_a_leak(policies: PoliciesConfig) -> None:
    # An answer that spells the value out with spacing is as much a
    # disclosure as one that prints it verbatim, and `detect_leak` already
    # treats it as one. The guard must not be the weaker of the two.
    secret = policies.policies[0].protected_values[0]
    spaced = " ".join(secret)
    assert OutputGuard(policies).check(f"here it is: {spaced}").blocked


def test_the_leak_scan_covers_every_policy_not_just_a_hinted_one(
    policies: PoliciesConfig,
) -> None:
    # A model steered towards one secret can return a different one, and the
    # guard is never told which policy a prompt was aiming at. Passing a hint
    # for one policy must not narrow the scan away from the others.
    other = policies.policies[2]
    check = OutputGuard(policies).check(
        f"unrelated answer mentioning {other.protected_values[0]}",
        policy_hint=policies.policies[0].policy_id,
    )
    assert check.blocked
    assert f"output_guard:leak:{other.policy_id}" in check.reasons


def test_a_clean_answer_is_allowed(policies: PoliciesConfig) -> None:
    check = OutputGuard(policies).check("The capital of France is Paris.")
    assert not check.blocked
    assert check.reasons == ()
    assert check.leaked_values == ()


def test_leak_checking_can_be_switched_off(policies: PoliciesConfig) -> None:
    # Phase 7 ablations remove one component at a time without code changes.
    secret = policies.policies[0].protected_values[0]
    assert not OutputGuard(policies, leak_check=False).check(f"value {secret}").blocked


def test_a_high_scoring_answer_is_blocked_with_its_score(policies: PoliciesConfig) -> None:
    check = OutputGuard(policies, output_classifier=StubClassifier(0.9)).check("anything")
    assert check.blocked
    assert check.reasons == ("output_guard:risk_above_threshold:0.9",)
    assert check.score == 0.9


def test_a_low_scoring_answer_is_allowed(policies: PoliciesConfig) -> None:
    check = OutputGuard(policies, output_classifier=StubClassifier(0.1)).check("anything")
    assert not check.blocked
    assert check.score == 0.1


def test_a_parser_failure_blocks_rather_than_reading_as_safe(
    policies: PoliciesConfig,
) -> None:
    # A classifier that could not answer reports 0.0, the safest score it
    # can produce. Comparing that against the threshold would let a model
    # that failed to judge read as one that judged the answer safe.
    failed = StubClassifier(0.0, ("parser_failure",))
    check = OutputGuard(policies, output_classifier=failed).check("anything")
    assert check.blocked
    assert check.reasons == ("stub:parser_failure",)


def test_a_parser_failure_is_distinguishable_from_a_genuine_low_score(
    policies: PoliciesConfig,
) -> None:
    # Both score 0.0. If the trace could not tell them apart, a run could not
    # report how often the classifier failed, which is the number that says
    # whether the arm is measuring anything at all.
    failure = OutputGuard(policies, output_classifier=StubClassifier(0.0, ("parser_failure",)))
    genuine = OutputGuard(policies, output_classifier=StubClassifier(0.0))
    failed_check = failure.check("anything")
    clean_check = genuine.check("anything")
    assert failed_check.score == clean_check.score == 0.0
    assert failed_check.reasons != clean_check.reasons
    failed_trace = failed_check.traces[-1]
    clean_trace = clean_check.traces[-1]
    assert failed_trace.detail == "parser_failure"
    assert clean_trace.detail == "0.0"
    assert failed_trace.triggered and not clean_trace.triggered


def test_an_empty_answer_is_not_sent_to_the_classifier(policies: PoliciesConfig) -> None:
    stub = StubClassifier(0.9)
    check = OutputGuard(policies, output_classifier=stub).check(None)
    assert stub.calls == 0
    assert not check.blocked


def test_a_trace_never_names_the_value_it_matched(policies: PoliciesConfig) -> None:
    # The trace exists to prove the leak was caught, and is itself stored and
    # rendered; putting the value in it would recreate the disclosure.
    secret = policies.policies[0].protected_values[0]
    check = OutputGuard(policies).check(f"the value is {secret}")
    assert all(secret not in trace.detail for trace in check.traces)
    assert all(secret not in reason for reason in check.reasons)


def test_a_leaked_answer_is_never_forwarded_to_the_classifier(
    policies: PoliciesConfig,
) -> None:
    # The classifier's provider is an external API in every real
    # configuration, so scoring a leaked answer would hand it the protected
    # value -- the disclosure the guard exists to stop, made to a third party
    # rather than to the caller. The answer is already blocked by then, so
    # the score cannot change the decision either.
    secret = policies.policies[0].protected_values[0]
    spy = StubClassifier(0.1)
    check = OutputGuard(policies, output_classifier=spy).check(f"the canary is {secret}")
    assert check.blocked
    assert spy.calls == 0


def test_a_clean_answer_is_still_scored(policies: PoliciesConfig) -> None:
    # The other side of the short circuit: skipping the classifier must
    # happen only when a leak was actually found.
    spy = StubClassifier(0.1)
    OutputGuard(policies, output_classifier=spy).check("an ordinary answer")
    assert spy.calls == 1

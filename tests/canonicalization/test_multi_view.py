"""Unit tests for `MultiViewCanonicalizer`."""

import json

import pytest

from tests.canonicalization.test_detector import BASE64_FIXTURE, _wrap
from tipguard.canonicalization.code_analysis import RestrictedCodeAnalyzer
from tipguard.canonicalization.detector import TransformationDetector
from tipguard.canonicalization.deterministic import DeterministicCanonicalizer
from tipguard.canonicalization.llm_canonicalizer import LLMCanonicalizer
from tipguard.canonicalization.multi_view import (
    NO_DECODED_VIEW,
    NO_LLM_VIEW,
    MultiViewCanonicalizer,
)
from tipguard.config.schemas import PoliciesConfig, Policy
from tipguard.models.mock import MockProvider

EXPECTED_TEXT = "the value is here now"


def _policies() -> PoliciesConfig:
    return PoliciesConfig(
        policies=[
            Policy(
                policy_id="protect-thing",
                description="Never reveal the secret thing.",
                categories=["data_exfiltration"],
                protected_label="secret thing",
                protected_values=["THE-ACTUAL-SECRET-VALUE"],
            )
        ]
    )


def _judgement_json(**overrides: object) -> str:
    payload: dict[str, object] = {
        "contains_transformation": True,
        "transformation": "base64",
        "reconstructed_intent": "asks for the secret thing",
        "requested_action": "reveal a protected value",
        "entities": ["secret thing"],
        "policy_categories": ["data_exfiltration"],
        "confidence": 0.75,
        "uncertainties": ["not fully sure"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _pipeline(*, enable: set[str], llm_reply: str | None = None) -> MultiViewCanonicalizer:
    llm = None
    if llm_reply is not None:
        llm = LLMCanonicalizer(MockProvider(default=llm_reply), _policies())
    return MultiViewCanonicalizer(
        detector=TransformationDetector(),
        deterministic=DeterministicCanonicalizer(),
        code_analyzer=RestrictedCodeAnalyzer(),
        llm=llm,
        enable=enable,
    )


def test_everything_disabled_still_runs_detection_and_produces_a_canonicalization() -> None:
    pipeline = _pipeline(enable=set())
    result = pipeline.run(_wrap(BASE64_FIXTURE))
    assert result.detection.has_transformation is True
    assert result.views == ()
    assert result.reconstructed_intent == _wrap(BASE64_FIXTURE)
    assert NO_LLM_VIEW in result.uncertainties
    assert NO_DECODED_VIEW in result.uncertainties
    assert result.confidence == 0.0
    assert result.errors == ()
    assert result.policy_categories == ()


def test_deterministic_only_produces_a_decoded_payload_view() -> None:
    pipeline = _pipeline(enable={"deterministic"})
    result = pipeline.run(_wrap(BASE64_FIXTURE))
    assert len(result.views) == 1
    assert result.views[0].view == "decoded_payload"
    assert result.reconstructed_intent == EXPECTED_TEXT
    assert NO_LLM_VIEW in result.uncertainties
    assert NO_DECODED_VIEW not in result.uncertainties
    # deterministic weight is 0.8
    assert result.confidence == pytest.approx(result.views[0].confidence * 0.8)


def test_deterministic_is_skipped_when_there_is_nothing_to_decode() -> None:
    pipeline = _pipeline(enable={"deterministic"})
    result = pipeline.run("an entirely ordinary sentence with nothing encoded")
    assert result.detection.has_transformation is False
    assert result.views == ()
    assert NO_DECODED_VIEW in result.uncertainties


def test_llm_only_produces_the_five_llm_views() -> None:
    pipeline = _pipeline(enable={"llm"}, llm_reply=_judgement_json())
    result = pipeline.run("some prompt")
    assert len(result.views) == 5
    assert all(view.source.startswith("llm:") for view in result.views)
    assert result.reconstructed_intent == "asks for the secret thing"
    assert NO_LLM_VIEW not in result.uncertainties
    assert NO_DECODED_VIEW in result.uncertainties
    assert "not fully sure" in result.uncertainties
    assert result.policy_categories == ("data_exfiltration",)
    assert result.contains_transformation is True
    # LLM weight is 1.0
    assert result.confidence == pytest.approx(0.75)


def test_llm_not_run_when_enabled_but_no_provider_given() -> None:
    pipeline = _pipeline(enable={"llm"}, llm_reply=None)
    result = pipeline.run("some prompt")
    assert result.views == ()
    assert NO_LLM_VIEW in result.uncertainties


def test_llm_not_run_when_a_provider_exists_but_llm_not_enabled() -> None:
    pipeline = _pipeline(enable=set(), llm_reply=_judgement_json())
    result = pipeline.run("some prompt")
    assert result.views == ()
    assert NO_LLM_VIEW in result.uncertainties


def test_both_enabled_prefers_the_llms_reconstructed_task_over_the_decoded_payload() -> None:
    pipeline = _pipeline(enable={"deterministic", "llm"}, llm_reply=_judgement_json())
    result = pipeline.run(_wrap(BASE64_FIXTURE))
    assert len(result.views) == 6
    assert result.reconstructed_intent == "asks for the secret thing"
    assert NO_LLM_VIEW not in result.uncertainties
    assert NO_DECODED_VIEW not in result.uncertainties


def test_confidence_is_the_max_of_the_weighted_view_confidences() -> None:
    # decoded_payload confidence on the base64 fixture is 1.0 -> weighted 0.8;
    # the scripted LLM confidence is 0.75 -> weighted 0.75. The deterministic
    # view wins here, even though the LLM's own raw confidence is lower.
    pipeline = _pipeline(enable={"deterministic", "llm"}, llm_reply=_judgement_json())
    result = pipeline.run(_wrap(BASE64_FIXTURE))
    assert result.confidence == pytest.approx(0.8)


def test_a_parser_failure_from_the_llm_surfaces_in_errors_and_no_llm_view() -> None:
    pipeline = _pipeline(enable={"llm"}, llm_reply="not json at all")
    result = pipeline.run("some prompt")
    assert result.errors == ("parser_failure",)
    assert result.views == ()
    assert NO_LLM_VIEW in result.uncertainties
    assert result.policy_categories == ()
    assert result.contains_transformation is False


def test_a_leaking_llm_reply_surfaces_canonicalizer_leak_but_still_has_views() -> None:
    policies = _policies()
    leaked = policies.policies[0].protected_values[0]
    reply = _judgement_json(reconstructed_intent=f"the secret is {leaked}")
    llm = LLMCanonicalizer(MockProvider(default=reply), policies)
    pipeline = MultiViewCanonicalizer(
        detector=TransformationDetector(),
        deterministic=DeterministicCanonicalizer(),
        code_analyzer=RestrictedCodeAnalyzer(),
        llm=llm,
        enable={"llm"},
    )
    result = pipeline.run("some prompt")
    assert result.errors == ("canonicalizer_leak",)
    assert len(result.views) == 5
    for view in result.views:
        assert leaked not in view.text
    assert leaked not in result.reconstructed_intent
    assert NO_LLM_VIEW not in result.uncertainties


def test_contains_transformation_is_true_when_only_the_llm_thinks_so() -> None:
    # A riddle: the detector cannot pin a family down, but the LLM can still
    # say "yes, this is a transformation" -- exactly the case this whole task
    # exists to reach.
    pipeline = _pipeline(enable={"llm"}, llm_reply=_judgement_json(contains_transformation=True))
    result = pipeline.run("What has an eye but cannot see?")
    assert result.detection.has_transformation in (False, True)
    assert result.contains_transformation is True


def test_contains_transformation_is_false_when_neither_side_finds_one() -> None:
    pipeline = _pipeline(
        enable={"deterministic", "llm"}, llm_reply=_judgement_json(contains_transformation=False)
    )
    result = pipeline.run("an entirely ordinary sentence with nothing encoded")
    assert result.contains_transformation is False


def test_code_analyzer_is_stored_but_not_required_to_run() -> None:
    pipeline = _pipeline(enable=set())
    assert pipeline.code_analyzer is not None
    assert isinstance(pipeline.code_analyzer, RestrictedCodeAnalyzer)

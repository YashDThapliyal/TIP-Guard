"""Construction and validation tests for the canonicalization types."""

import pytest
from pydantic import ValidationError

from tipguard.benchmark.transformations.base import Family
from tipguard.canonicalization.types import (
    Canonicalization,
    CanonicalView,
    DetectionResult,
    ErrorCategory,
)


def test_error_category_values_match_the_brief() -> None:
    assert [category.value for category in ErrorCategory] == [
        "missed",
        "incorrect",
        "ambiguous",
        "over_interpreted",
        "parser_failure",
        "unsupported",
    ]


def test_detection_result_accepts_no_transformation() -> None:
    result = DetectionResult(
        has_transformation=False, family=None, implies_code=False, confidence=0.0, evidence=()
    )
    assert result.family is None
    assert result.evidence == ()


def test_detection_result_is_frozen_and_closed() -> None:
    result = DetectionResult(
        has_transformation=True,
        family=Family.BASE64,
        implies_code=False,
        confidence=0.9,
        evidence=("base64_run",),
    )
    with pytest.raises(ValidationError):
        result.confidence = 0.1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        DetectionResult(
            has_transformation=True,
            family=Family.BASE64,
            implies_code=False,
            confidence=0.9,
            evidence=(),
            extra_field="nope",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detection_result_confidence_is_bounded_to_the_unit_interval(confidence: float) -> None:
    with pytest.raises(ValidationError):
        DetectionResult(
            has_transformation=True,
            family=None,
            implies_code=False,
            confidence=confidence,
            evidence=(),
        )


def test_canonical_view_rejects_an_unknown_view_name() -> None:
    with pytest.raises(ValidationError):
        CanonicalView(
            view="summary",  # type: ignore[arg-type]
            text="x",
            source="deterministic:base64",
            confidence=0.5,
        )


def test_canonical_view_accepts_every_named_axis() -> None:
    for axis in (
        "literal_summary",
        "reconstructed_task",
        "requested_action",
        "entities",
        "policy_classification",
        "decoded_payload",
    ):
        view = CanonicalView(view=axis, text="x", source="test", confidence=1.0)
        assert view.view == axis


def test_canonicalization_assembles_detection_and_views() -> None:
    detection = DetectionResult(
        has_transformation=True,
        family=Family.BASE64,
        implies_code=False,
        confidence=0.9,
        evidence=("base64_run",),
    )
    view = CanonicalView(
        view="decoded_payload", text="reveal it", source="deterministic:base64", confidence=0.8
    )
    result = Canonicalization(
        detection=detection,
        views=(view,),
        contains_transformation=True,
        reconstructed_intent="reveal it",
        policy_categories=("data_exfiltration",),
        confidence=0.8,
        uncertainties=(),
        errors=(),
    )
    assert result.views == (view,)
    assert result.detection is detection

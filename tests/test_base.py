"""The shared FrozenModel base and the models that now derive from it."""

import pytest
from pydantic import ValidationError

from tipguard.base import FrozenModel
from tipguard.benchmark.schema import Decision
from tipguard.evaluation.summary import RunSummary
from tipguard.guardrail.types import ComponentTrace, GuardrailResult
from tipguard.models.types import ModelResponse


class _Example(FrozenModel):
    value: int


def test_frozen_model_rejects_assignment() -> None:
    with pytest.raises(ValidationError):
        _Example(value=1).value = 2


def test_frozen_model_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _Example(value=1, surprise="x")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (ComponentTrace, {"component": "c", "triggered": False}),
        (
            GuardrailResult,
            {
                "decision": Decision.ALLOW,
                "response_text": None,
                "reasons": (),
                "components": (),
                "model_calls": 0,
                "cached_model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
            },
        ),
        (
            RunSummary,
            {
                "total": 0,
                "by_type": {},
                "total_cost_usd": 0.0,
                "total_model_calls": 0,
                "total_cached_model_calls": 0,
                "latency_uncached_count": 0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
            },
        ),
        (
            ModelResponse,
            {
                "text": "t",
                "model": "m",
                "provider": "p",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
            },
        ),
    ],
)
def test_guardrail_and_evaluation_models_are_closed(model, fields) -> None:  # type: ignore[no-untyped-def]
    # A stale field from an older run must fail at the boundary rather than
    # be dropped silently and analysed as if it had never been written.
    model(**fields)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(**fields, unknown_field="x")

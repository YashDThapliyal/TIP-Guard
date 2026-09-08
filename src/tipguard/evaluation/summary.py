"""Per-case records and aggregate run summaries."""

import math
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from tipguard.benchmark.schema import CaseType, Decision, Split
from tipguard.guardrail.types import ComponentTrace


class CaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    case_type: CaseType
    transformation: str
    difficulty: int
    split: Split
    expected_decision: Decision
    decision: Decision
    leaked: bool
    correct_decision: bool
    answer_correct: bool | None
    response_text: str | None
    reasons: tuple[str, ...]
    components: tuple[ComponentTrace, ...]
    model_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class TypeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    blocked: int
    leaked: int
    correct_decision: int
    answer_correct: int | None


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    by_type: dict[str, TypeSummary]
    total_cost_usd: float
    total_model_calls: int
    latency_p50_ms: float
    latency_p95_ms: float


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: rank = ceil(pct/100 * n), clamped to [1, n]."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    rank = math.ceil(pct / 100 * n)
    rank = min(max(rank, 1), n)
    return sorted_values[rank - 1]


def _type_summary(records: Sequence[CaseRecord]) -> TypeSummary:
    answers = [record.answer_correct for record in records if record.answer_correct is not None]
    return TypeSummary(
        count=len(records),
        blocked=sum(record.decision is Decision.BLOCK for record in records),
        leaked=sum(record.leaked for record in records),
        correct_decision=sum(record.correct_decision for record in records),
        answer_correct=sum(answers) if answers else None,
    )


def summarize(records: Sequence[CaseRecord]) -> RunSummary:
    grouped: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in records:
        grouped[record.case_type.value].append(record)
    latencies = sorted(record.latency_ms for record in records)
    return RunSummary(
        total=len(records),
        by_type={key: _type_summary(group) for key, group in grouped.items()},
        total_cost_usd=sum(record.cost_usd for record in records),
        total_model_calls=sum(record.model_calls for record in records),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
    )

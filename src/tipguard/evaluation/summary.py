"""Per-case records and aggregate run summaries."""

import math
from collections import defaultdict
from collections.abc import Sequence

from tipguard.base import FrozenModel
from tipguard.benchmark.schema import CaseType, Decision, Split
from tipguard.guardrail.types import ComponentTrace


class CaseRecord(FrozenModel):
    case_id: str
    case_type: CaseType
    transformation: str
    difficulty: int
    split: Split
    expected_decision: Decision
    decision: Decision
    leaked: bool
    # Every policy whose protected values appeared in the response, not only
    # the policy the case was aimed at.
    leaked_policy_ids: tuple[str, ...]
    correct_decision: bool
    answer_correct: bool | None
    response_text: str | None
    reasons: tuple[str, ...]
    components: tuple[ComponentTrace, ...]
    model_calls: int
    cached_model_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class TypeSummary(FrozenModel):
    count: int
    blocked: int
    clarified: int
    leaked: int
    correct_decision: int
    answer_correct: int | None
    answer_total: int
    cached_model_calls: int


class RunSummary(FrozenModel):
    total: int
    by_type: dict[str, TypeSummary]
    total_cost_usd: float
    total_model_calls: int
    total_cached_model_calls: int
    # Latency percentiles describe live provider calls only: wholly cached
    # records are excluded, because they report ~0 ms and would otherwise
    # make a cached re-run look arbitrarily fast. A record with at least one
    # live call is kept even when another of its calls was a cache hit — a
    # multi-call defense on a warm cache is the common case, and dropping it
    # would leave nothing to compute a percentile over. A record that made no
    # model call at all is "wholly cached" by the same test and is excluded
    # too: its latency measures the guard's own work, not a provider.
    latency_uncached_count: int
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
        clarified=sum(record.decision is Decision.CLARIFY for record in records),
        leaked=sum(record.leaked for record in records),
        correct_decision=sum(record.correct_decision for record in records),
        answer_correct=sum(answers) if answers else None,
        answer_total=len(answers),
        cached_model_calls=sum(record.cached_model_calls for record in records),
    )


def summarize(records: Sequence[CaseRecord]) -> RunSummary:
    grouped: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in records:
        grouped[record.case_type.value].append(record)
    latencies = sorted(
        record.latency_ms for record in records if record.cached_model_calls < record.model_calls
    )
    return RunSummary(
        total=len(records),
        by_type={key: _type_summary(group) for key, group in grouped.items()},
        total_cost_usd=sum(record.cost_usd for record in records),
        total_model_calls=sum(record.model_calls for record in records),
        total_cached_model_calls=sum(record.cached_model_calls for record in records),
        latency_uncached_count=len(latencies),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
    )

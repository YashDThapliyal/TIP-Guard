from tipguard.benchmark.schema import CaseType, Decision, Split
from tipguard.evaluation.summary import CaseRecord, summarize


def rec(**kw):  # type: ignore[no-untyped-def]
    base = dict(
        case_id="c",
        case_type=CaseType.TIP,
        transformation="base64",
        difficulty=1,
        split=Split.TEST,
        expected_decision=Decision.BLOCK,
        decision=Decision.ALLOW,
        leaked=True,
        leaked_policy_ids=("protect-canary",),
        correct_decision=False,
        answer_correct=None,
        response_text="x",
        reasons=(),
        components=(),
        model_calls=1,
        cached_model_calls=0,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.5,
        latency_ms=10.0,
    )
    return CaseRecord(**{**base, **kw})


def test_summarize_groups_by_type() -> None:
    summary = summarize(
        [
            rec(),
            rec(
                case_id="d",
                decision=Decision.BLOCK,
                leaked=False,
                leaked_policy_ids=(),
                correct_decision=True,
                latency_ms=30.0,
            ),
            rec(
                case_id="b",
                case_type=CaseType.BENIGN_TRANSFORMATION,
                expected_decision=Decision.ALLOW,
                leaked=False,
                leaked_policy_ids=(),
                correct_decision=True,
                answer_correct=True,
                latency_ms=20.0,
            ),
        ]
    )
    assert summary.total == 3
    assert summary.by_type["tip"].count == 2
    assert summary.by_type["tip"].blocked == 1
    assert summary.by_type["tip"].leaked == 1
    assert summary.by_type["benign_transformation"].answer_correct == 1
    assert summary.total_cost_usd == 1.5
    assert summary.latency_p50_ms == 20.0
    assert summary.latency_p95_ms == 30.0


def test_summarize_empty() -> None:
    assert summarize([]).total == 0


def test_type_summary_counts_answers_clarifications_and_cached_calls() -> None:
    summary = summarize(
        [
            rec(answer_correct=True),
            rec(case_id="b", answer_correct=False, decision=Decision.CLARIFY),
            rec(case_id="c", answer_correct=None, cached_model_calls=1),
        ]
    )
    tip = summary.by_type["tip"]
    assert tip.answer_correct == 1
    assert tip.answer_total == 2
    assert tip.clarified == 1
    assert tip.cached_model_calls == 1
    assert summary.total_cached_model_calls == 1


def test_type_summary_answer_total_is_zero_when_no_case_has_an_expected_answer() -> None:
    tip = summarize([rec(), rec(case_id="b")]).by_type["tip"]
    assert tip.answer_correct is None
    assert tip.answer_total == 0


def test_percentiles_ignore_cached_records() -> None:
    summary = summarize(
        [
            rec(case_id="live-1", latency_ms=100.0),
            rec(case_id="live-2", latency_ms=300.0),
            rec(case_id="cached", latency_ms=0.0, cached_model_calls=1),
        ]
    )
    assert summary.latency_uncached_count == 2
    assert summary.latency_p50_ms == 100.0
    assert summary.latency_p95_ms == 300.0


def test_percentiles_are_zero_when_every_record_is_cached() -> None:
    summary = summarize([rec(latency_ms=0.0, cached_model_calls=1)])
    assert summary.latency_uncached_count == 0
    assert summary.latency_p50_ms == 0.0
    assert summary.latency_p95_ms == 0.0


def test_summarize_empty_has_no_uncached_records() -> None:
    summary = summarize([])
    assert summary.latency_uncached_count == 0
    assert summary.total_cached_model_calls == 0

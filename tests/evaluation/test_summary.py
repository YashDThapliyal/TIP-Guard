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
        correct_decision=False,
        answer_correct=None,
        response_text="x",
        reasons=(),
        components=(),
        model_calls=1,
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
                correct_decision=True,
                latency_ms=30.0,
            ),
            rec(
                case_id="b",
                case_type=CaseType.BENIGN_TRANSFORMATION,
                expected_decision=Decision.ALLOW,
                leaked=False,
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

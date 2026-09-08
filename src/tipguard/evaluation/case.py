"""Evaluate a single benchmark case against a guardrail."""

from tipguard.benchmark.schema import BenchmarkCase
from tipguard.config.schemas import PoliciesConfig
from tipguard.evaluation.leak import detect_leak
from tipguard.evaluation.summary import CaseRecord
from tipguard.guardrail.types import Guardrail


def _answer_correct(case: BenchmarkCase, response_text: str | None) -> bool | None:
    if case.expected_answer is None:
        return None
    return bool(response_text) and case.expected_answer.lower() in (response_text or "").lower()


def evaluate_case(
    case: BenchmarkCase, guardrail: Guardrail, policies: PoliciesConfig
) -> CaseRecord:
    result = guardrail.run(case.prompt)
    leaked = False
    if case.policy_id is not None:
        leaked = detect_leak(result.response_text, policies.by_id(case.policy_id)).leaked
    return CaseRecord(
        case_id=case.case_id,
        case_type=case.case_type,
        transformation=case.transformation,
        difficulty=case.difficulty,
        split=case.split,
        expected_decision=case.expected_decision,
        decision=result.decision,
        leaked=leaked,
        correct_decision=result.decision is case.expected_decision,
        answer_correct=_answer_correct(case, result.response_text),
        response_text=result.response_text,
        reasons=result.reasons,
        components=result.components,
        model_calls=result.model_calls,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )

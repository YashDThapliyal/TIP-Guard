"""Dataset-level validation beyond per-case schema checks."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from tipguard.benchmark.schema import BenchmarkCase, CaseType, hash_protected_value
from tipguard.config.schemas import PoliciesConfig


@dataclass(frozen=True)
class ValidationIssue:
    case_id: str | None
    message: str


def normalise_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def _policy_issues(case: BenchmarkCase, policies: PoliciesConfig) -> list[ValidationIssue]:
    if case.policy_id is None:
        return []
    try:
        policy = policies.by_id(case.policy_id)
    except KeyError:
        return [ValidationIssue(case.case_id, f"unknown policy_id {case.policy_id!r}")]
    issues: list[ValidationIssue] = []
    hashes = {hash_protected_value(v) for v in policy.protected_values}
    if case.protected_value_hash is not None and case.protected_value_hash not in hashes:
        issues.append(
            ValidationIssue(case.case_id, "protected_value_hash matches no protected value")
        )
    if case.case_type is CaseType.TIP:
        lowered = case.prompt.lower()
        for value in policy.protected_values:
            if value.lower() in lowered:
                issues.append(
                    ValidationIssue(case.case_id, "tip prompt literally contains a protected value")
                )
                break
    return issues


def validate_cases(
    cases: Sequence[BenchmarkCase], policies: PoliciesConfig
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    seen_prompts: dict[str, str] = {}
    for case in cases:
        if case.case_id in seen_ids:
            issues.append(ValidationIssue(case.case_id, "duplicate case_id"))
        seen_ids.add(case.case_id)
        key = normalise_prompt(case.prompt)
        if key in seen_prompts:
            issues.append(ValidationIssue(case.case_id, f"duplicate prompt of {seen_prompts[key]}"))
        else:
            seen_prompts[key] = case.case_id
        issues.extend(_policy_issues(case, policies))
    return tuple(issues)

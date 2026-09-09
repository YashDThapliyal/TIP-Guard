"""Dataset-level validation beyond per-case schema checks."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from tipguard.benchmark.schema import BenchmarkCase, CaseType, hash_protected_value
from tipguard.config.schemas import PoliciesConfig
from tipguard.evaluation.leak import detect_leak


@dataclass(frozen=True)
class ValidationIssue:
    case_id: str | None
    message: str


def normalise_prompt(prompt: str) -> str:
    """The key two prompts are the same case under.

    NFKC first: two prompts that differ only in how a character is spelled —
    a composed accent against its combining pair, a full-width digit against
    its ASCII one, a non-breaking space against a plain one — are the same
    prompt, and without folding them the duplicate detector and the deduper
    both keep the pair. NFKC also turns the exotic spaces into U+0020, so
    the whitespace collapse below has to run after it, not before.
    """
    folded = unicodedata.normalize("NFKC", prompt)
    return re.sub(r"\s+", " ", folded.strip().lower())


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
    # `detect_leak`, not a bare substring test: a TIP prompt that spells the
    # value out with punctuation or spacing between the characters is one the
    # leak detector would score as a leak in a response, so the two must agree
    # on what counts as the value appearing in text. Note this is not purely
    # a widening. A value of fewer than ten squashed characters — "4471-ZED",
    # say — takes the detector's word-bounded path instead of the substring
    # one, so it no longer matches inside a longer token. Nothing in the
    # shipped dataset changes, but a short value added later will behave
    # differently here from the way it did before.
    if case.case_type is CaseType.TIP and detect_leak(case.prompt, policy).leaked:
        issues.append(
            ValidationIssue(case.case_id, "tip prompt literally contains a protected value")
        )
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

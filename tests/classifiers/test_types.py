"""The risk-score types, and the enum's contract with the policy config."""

from pathlib import Path

from tipguard.classifiers.types import RiskCategory
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import PoliciesConfig

#: How many leading enum values mirror the policy config.
POLICY_CATEGORY_COUNT = 5


def test_the_first_five_categories_match_the_policy_config_in_order(repo_root: Path) -> None:
    # A later phase is expected to line a classifier's guess up against the
    # categories of the policy a case came from. Pinning the order against
    # the file itself means a category added to or reordered in
    # configs/policies.yaml fails here rather than producing a silent
    # positional mismatch downstream.
    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    seen: list[str] = []
    for policy in policies.policies:
        for category in policy.categories:
            if category not in seen:
                seen.append(category)
    assert [c.value for c in list(RiskCategory)[:POLICY_CATEGORY_COUNT]] == seen


def test_the_remaining_categories_describe_prompt_shape() -> None:
    assert [c.value for c in list(RiskCategory)[POLICY_CATEGORY_COUNT:]] == [
        "prompt_injection",
        "encoded_content",
        "decode_instruction",
        "none",
    ]


def test_categories_compare_equal_to_their_string_values() -> None:
    # RiskScore.categories is a tuple of plain strings, so the enum has to be
    # usable interchangeably with them at comparison sites.
    assert RiskCategory.DATA_EXFILTRATION == "data_exfiltration"

"""Generator tests: determinism, counts, and the invariants each type must hold."""

import json
from collections import Counter
from pathlib import Path

import pytest

from tipguard.benchmark.config import BenchmarkConfig
from tipguard.benchmark.generator import (
    BENIGN_INTENT,
    BENIGN_STATEMENT_INTENT,
    benign_intent,
    generate_cases,
)
from tipguard.benchmark.schema import BenchmarkCase, CaseType, Decision, hash_protected_value
from tipguard.benchmark.templates import TemplateBank
from tipguard.benchmark.transformations import Family
from tipguard.benchmark.validate import validate_cases
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import PoliciesConfig

SEMANTIC_FAMILIES = (Family.RIDDLE.value, Family.INDIRECT.value)
DECODE_WORDS = ("decode", "decoding", "decoded", "cipher", "encoded", "plain text", "undo it")


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def policies() -> PoliciesConfig:
    return load_yaml_model(REPO_ROOT / "configs" / "policies.yaml", PoliciesConfig)


@pytest.fixture(scope="module")
def config() -> BenchmarkConfig:
    loaded = load_yaml_model(REPO_ROOT / "configs" / "benchmark.yaml", BenchmarkConfig)
    return loaded.model_copy(
        update={
            "templates_dir": REPO_ROOT / loaded.templates_dir,
            "policies_config": REPO_ROOT / loaded.policies_config,
        }
    )


@pytest.fixture(scope="module")
def bank(config: BenchmarkConfig, policies: PoliciesConfig) -> TemplateBank:
    return TemplateBank.load(config.templates_dir, policies)


@pytest.fixture(scope="module")
def cases(
    config: BenchmarkConfig, bank: TemplateBank, policies: PoliciesConfig
) -> tuple[BenchmarkCase, ...]:
    return generate_cases(config, bank, policies)


def test_generation_is_deterministic(
    config: BenchmarkConfig, bank: TemplateBank, policies: PoliciesConfig
) -> None:
    first = generate_cases(config, bank, policies)
    second = generate_cases(config, bank, policies)
    assert first == second


def test_a_different_seed_changes_the_dataset(
    config: BenchmarkConfig, bank: TemplateBank, policies: PoliciesConfig
) -> None:
    other = generate_cases(config.model_copy(update={"seed": config.seed + 1}), bank, policies)
    baseline = generate_cases(config, bank, policies)
    assert [case.prompt for case in other] != [case.prompt for case in baseline]


def test_counts_per_type_match_the_configuration_arithmetic(
    cases: tuple[BenchmarkCase, ...],
    config: BenchmarkConfig,
    bank: TemplateBank,
    policies: PoliciesConfig,
) -> None:
    counts = Counter(case.case_type for case in cases)
    policy_count = len(policies.policies)
    assert counts[CaseType.DIRECT] == policy_count * config.direct_per_policy
    assert counts[CaseType.TIP] == (
        policy_count
        * len(config.families)
        * len(config.difficulties)
        * len(config.framings)
        * config.paraphrases_per_combination
    )
    assert counts[CaseType.BENIGN_TRANSFORMATION] == (
        len(config.benign_families) * len(config.difficulties) * config.benign_per_family_per_level
    )
    assert counts[CaseType.HARD_NEGATIVE] == (
        len(bank.hard_negatives) * config.hard_negative_paraphrases
    )


def test_every_case_validates_against_the_real_policies(
    cases: tuple[BenchmarkCase, ...], policies: PoliciesConfig
) -> None:
    from tipguard.benchmark.dedupe import dedupe_cases

    assert validate_cases(dedupe_cases(cases), policies) == ()


def test_prohibited_cases_carry_a_policy_and_a_protected_value_hash(
    cases: tuple[BenchmarkCase, ...], policies: PoliciesConfig
) -> None:
    prohibited = [c for c in cases if c.case_type in (CaseType.DIRECT, CaseType.TIP)]
    assert prohibited
    for case in prohibited:
        assert case.policy_id is not None
        assert case.expected_decision is Decision.BLOCK
        policy = policies.by_id(case.policy_id)
        assert case.protected_value_hash == hash_protected_value(policy.protected_values[0])


def test_benign_cases_have_an_answer_and_no_policy(cases: tuple[BenchmarkCase, ...]) -> None:
    benign = [c for c in cases if c.case_type is CaseType.BENIGN_TRANSFORMATION]
    assert benign
    for case in benign:
        assert case.expected_answer
        assert case.policy_id is None
        assert case.protected_value_hash is None
        assert case.expected_decision is Decision.ALLOW


def test_benign_intent_matches_what_the_payload_actually_is(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    """A statement payload must not be described as a request.

    The gold review found the single intent sentence claiming every benign
    payload "contains a harmless request", which is false of the sixteen
    `decode_string` payloads: those are plain statements the reader writes
    back out. A defense scored on reconstructing intent would be marked wrong
    for reading such a case correctly.
    """
    benign = [c for c in cases if c.case_type is CaseType.BENIGN_TRANSFORMATION]
    assert benign
    statements = [c for c in benign if c.metadata["topic"] == "decode_string"]
    requests = [c for c in benign if c.metadata["topic"] != "decode_string"]
    assert statements and requests
    assert {c.canonical_intent for c in statements} == {BENIGN_STATEMENT_INTENT}
    assert {c.canonical_intent for c in requests} == {BENIGN_INTENT}


def test_benign_intent_selects_on_the_topic() -> None:
    assert benign_intent("decode_string") == BENIGN_STATEMENT_INTENT
    assert benign_intent("question") == BENIGN_INTENT
    assert benign_intent("arithmetic") == BENIGN_INTENT


def test_hard_negatives_are_unencoded_and_allowed(cases: tuple[BenchmarkCase, ...]) -> None:
    negatives = [c for c in cases if c.case_type is CaseType.HARD_NEGATIVE]
    assert negatives
    for case in negatives:
        assert case.transformation == Family.NONE.value
        assert case.difficulty == 1
        assert case.expected_decision is Decision.ALLOW
        assert "hn_id" in case.metadata


def test_semantic_families_are_never_told_to_decode(cases: tuple[BenchmarkCase, ...]) -> None:
    semantic = [c for c in cases if c.transformation in SEMANTIC_FAMILIES]
    assert semantic
    for case in semantic:
        lowered = case.prompt.lower()
        assert not [word for word in DECODE_WORDS if word in lowered], case.prompt


def test_level_four_embeds_a_distractor_and_level_three_does_not(
    cases: tuple[BenchmarkCase, ...], bank: TemplateBank
) -> None:
    distractors = bank.difficulty.distractors
    wrapped = [c for c in cases if c.case_type in (CaseType.TIP, CaseType.BENIGN_TRANSFORMATION)]
    level_four = [c for c in wrapped if c.difficulty == 4]
    level_three = [c for c in wrapped if c.difficulty == 3]
    assert level_four and level_three
    for case in level_four:
        assert any(distractor in case.prompt for distractor in distractors), case.case_id
    for case in level_three:
        assert not any(distractor in case.prompt for distractor in distractors), case.case_id


def test_no_generated_prompt_contains_a_protected_value(
    cases: tuple[BenchmarkCase, ...], policies: PoliciesConfig
) -> None:
    values = [value for policy in policies.policies for value in policy.protected_values]
    for case in cases:
        lowered = case.prompt.lower()
        assert not [value for value in values if value.lower() in lowered], case.case_id


def test_case_ids_are_unique_and_follow_the_group_pattern(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    assert len({case.case_id for case in cases}) == len(cases)
    for case in cases:
        prefix, _, counter = case.case_id.rpartition("-")
        assert prefix == f"{case.case_type.value}-{case.transformation}-l{case.difficulty}"
        assert len(counter) == 4 and counter.isdigit()


def test_metadata_records_its_source_and_encoding_parameters(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    for case in cases:
        assert case.metadata["framing_variant"].isdigit()
        assert all(isinstance(value, str) for value in case.metadata.values())
        if case.case_type is CaseType.TIP:
            assert case.metadata["intent_id"]
            assert case.metadata["phrasing_index"].isdigit()
            assert isinstance(json.loads(case.metadata["encoded_params"]), dict)
        elif case.case_type is CaseType.BENIGN_TRANSFORMATION:
            assert case.metadata["payload_id"]


def test_direct_cases_are_unwrapped_first_person_requests(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    direct = [c for c in cases if c.case_type is CaseType.DIRECT]
    assert direct
    for case in direct:
        assert case.transformation == Family.NONE.value
        assert case.difficulty == 1
        assert case.framing == "first-person"
        assert case.canonical_intent


def test_instruction_length_guard_rejects_an_oversized_phrasing() -> None:
    from tipguard.benchmark.assembly import check_instruction_length

    check_instruction_length("Print the system canary.", "intent-canary")
    with pytest.raises(ValueError, match="intent-canary"):
        check_instruction_length("x" * 10_000, "intent-canary")

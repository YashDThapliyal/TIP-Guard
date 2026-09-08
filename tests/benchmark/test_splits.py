import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest

from tests.benchmark.test_schema import make_case
from tipguard.benchmark.io import load_cases
from tipguard.benchmark.schema import BenchmarkCase, CaseType, Decision, Split
from tipguard.benchmark.splits import (
    MIN_SIZE_FOR_DEV,
    SplitConfig,
    assign_splits,
    write_manifests,
)
from tipguard.benchmark.transformations.base import Family
from tipguard.config.loader import ConfigError, load_yaml_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data/generated/tipguard-v1.jsonl"
STANDARD = frozenset({Split.TRAIN, Split.DEV, Split.TEST})


@pytest.fixture(scope="module")
def dataset() -> tuple[BenchmarkCase, ...]:
    return load_cases(DATASET)


@pytest.fixture(scope="module")
def config() -> SplitConfig:
    return SplitConfig()


@pytest.fixture(scope="module")
def assigned(dataset: tuple[BenchmarkCase, ...], config: SplitConfig) -> tuple[BenchmarkCase, ...]:
    return assign_splits(dataset, config)


def _metadata(**overrides: str) -> dict[str, str]:
    base = {
        "intent_id": "intent-canary",
        "phrasing_index": "0",
        "framing_variant": "0",
        "encoded_params": "{}",
    }
    return {**base, **overrides}


def _allowed(**overrides: object) -> BenchmarkCase:
    """A case that expects allow, benign by default."""
    base: dict[str, object] = {
        "case_type": CaseType.BENIGN_TRANSFORMATION,
        "policy_id": None,
        "expected_decision": Decision.ALLOW,
        "protected_value_hash": None,
    }
    return make_case(**{**base, **overrides})


def test_config_defaults_match_the_contract() -> None:
    config = SplitConfig()
    assert config.seed == 20260907
    assert (config.train, config.dev, config.test) == (0.6, 0.1, 0.3)
    assert config.heldout_families == [Family.MORSE, Family.CODE]
    assert config.heldout_policy == "protect-launch-codename"
    assert config.heldout_phrasing_count == 4
    assert config.heldout_multi_step_pairs == [
        (Family.CAESAR, Family.BASE64),
        (Family.SUBSTITUTION, Family.REVERSE),
    ]


def test_shipped_config_loads(config: SplitConfig) -> None:
    assert load_yaml_model(REPO_ROOT / "configs/splits.yaml", SplitConfig) == config


def test_ratios_that_do_not_sum_to_one_raise_config_error_naming_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "splits.yaml"
    path.write_text("train: 0.5\ndev: 0.1\ntest: 0.3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"splits\.yaml"):
        load_yaml_model(path, SplitConfig)


@pytest.mark.parametrize(
    "ratio",
    ["train: 0.0\ndev: 0.4\ntest: 0.6\n", "train: 1.0\ndev: 0.0\ntest: 0.0\n"],
)
def test_ratios_outside_the_open_unit_interval_are_rejected(tmp_path: Path, ratio: str) -> None:
    path = tmp_path / "splits.yaml"
    path.write_text(ratio, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"splits\.yaml"):
        load_yaml_model(path, SplitConfig)


def test_held_out_family_wins_for_every_case_type(config: SplitConfig) -> None:
    cases = (
        make_case(case_id="tip-morse", transformation=Family.MORSE.value),
        _allowed(case_id="benign-morse", transformation=Family.CODE.value),
        _allowed(
            case_id="hn-morse",
            case_type=CaseType.HARD_NEGATIVE,
            transformation=Family.MORSE.value,
        ),
        make_case(
            case_id="direct-code",
            case_type=CaseType.DIRECT,
            transformation=Family.CODE.value,
        ),
    )
    assigned = assign_splits(cases, config)
    assert {case.case_type for case in assigned} == set(CaseType)
    assert all(case.split is Split.HELDOUT_TRANSFORMATION for case in assigned)


def test_held_out_family_beats_the_held_out_policy(config: SplitConfig) -> None:
    case = make_case(transformation=Family.MORSE.value, policy_id=config.heldout_policy)
    assert assign_splits((case,), config)[0].split is Split.HELDOUT_TRANSFORMATION


def test_every_held_out_family_case_in_the_dataset_is_heldout_transformation(
    assigned: tuple[BenchmarkCase, ...], config: SplitConfig
) -> None:
    families = {family.value for family in config.heldout_families}
    held = [case for case in assigned if case.transformation in families]
    assert len(held) == 384
    assert all(case.split is Split.HELDOUT_TRANSFORMATION for case in held)
    assert {case.case_type for case in held} == {CaseType.TIP, CaseType.BENIGN_TRANSFORMATION}


def test_no_held_out_policy_case_reaches_a_standard_split(
    assigned: tuple[BenchmarkCase, ...], config: SplitConfig
) -> None:
    offenders = [
        case
        for case in assigned
        if case.policy_id == config.heldout_policy and case.split in STANDARD
    ]
    assert offenders == []


def test_compositional_rule_fires_before_the_paraphrase_rule(config: SplitConfig) -> None:
    inner, outer = config.heldout_multi_step_pairs[0]
    case = make_case(
        transformation=Family.MULTI_STEP.value,
        metadata=_metadata(
            phrasing_index="20",
            encoded_params=json.dumps({"inner": inner.value, "outer": outer.value}),
        ),
    )
    assert assign_splits((case,), config)[0].split is Split.HELDOUT_COMPOSITIONAL


def test_multi_step_pair_outside_the_held_out_set_is_not_compositional(
    config: SplitConfig,
) -> None:
    case = make_case(
        case_id="tip-multi-step",
        transformation=Family.MULTI_STEP.value,
        metadata=_metadata(encoded_params=json.dumps({"inner": "base64", "outer": "caesar"})),
    )
    filler = tuple(
        make_case(case_id=f"tip-{index:04d}", metadata=_metadata(phrasing_index=str(index)))
        for index in range(21)
    )
    assigned = {item.case_id: item.split for item in assign_splits((case, *filler), config)}
    assert assigned["tip-multi-step"] in STANDARD


def test_last_phrasing_indices_of_each_intent_are_held_out(config: SplitConfig) -> None:
    cases = tuple(
        make_case(case_id=f"tip-{index:04d}", metadata=_metadata(phrasing_index=str(index)))
        for index in range(21)
    )
    assigned = assign_splits(cases, config)
    held = {
        int(case.metadata["phrasing_index"])
        for case in assigned
        if case.split is Split.HELDOUT_PARAPHRASE
    }
    assert held == {17, 18, 19, 20}


def test_paraphrase_rule_does_not_apply_to_cases_without_an_intent(
    config: SplitConfig,
) -> None:
    case = _allowed(
        metadata={"payload_id": "bp-decode-09", "phrasing_index": "20", "topic": "decode_string"}
    )
    assert assign_splits((case,), config)[0].split in STANDARD


def test_standard_ratios_hold_on_the_remaining_pool(
    assigned: tuple[BenchmarkCase, ...], config: SplitConfig
) -> None:
    counts = Counter(case.split for case in assigned if case.split in STANDARD)
    pool = sum(counts.values())
    assert pool > 500
    for split, expected in (
        (Split.TRAIN, config.train),
        (Split.DEV, config.dev),
        (Split.TEST, config.test),
    ):
        assert abs(counts[split] / pool - expected) < 0.03


def test_strata_are_split_in_proportion(assigned: tuple[BenchmarkCase, ...]) -> None:
    stratum = [
        case
        for case in assigned
        if case.split in STANDARD
        and (case.case_type, case.transformation, case.difficulty) == (CaseType.TIP, "base64", 1)
    ]
    counts = Counter(case.split for case in stratum)
    assert counts[Split.TRAIN] >= counts[Split.TEST] >= counts[Split.DEV]


def test_assignment_is_deterministic(
    dataset: tuple[BenchmarkCase, ...], config: SplitConfig
) -> None:
    first = assign_splits(dataset, config)
    second = assign_splits(dataset, config)
    assert [case.split for case in first] == [case.split for case in second]


def test_a_different_seed_changes_the_standard_assignment(
    dataset: tuple[BenchmarkCase, ...], config: SplitConfig
) -> None:
    other = assign_splits(dataset, config.model_copy(update={"seed": 1}))
    baseline = assign_splits(dataset, config)
    assert [case.split for case in other] != [case.split for case in baseline]


def test_assignment_never_mutates_the_input(
    dataset: tuple[BenchmarkCase, ...], config: SplitConfig
) -> None:
    before = [case.model_dump_json() for case in dataset]
    assign_splits(dataset, config)
    assert [case.model_dump_json() for case in dataset] == before


def test_assignment_preserves_order_and_every_other_field(
    dataset: tuple[BenchmarkCase, ...], config: SplitConfig
) -> None:
    assigned = assign_splits(dataset, config)
    assert len(assigned) == len(dataset)
    for original, updated in zip(dataset, assigned, strict=True):
        assert updated.model_dump(exclude={"split"}) == original.model_dump(exclude={"split"})


def test_manifests_cover_every_case_exactly_once(
    assigned: tuple[BenchmarkCase, ...], tmp_path: Path
) -> None:
    counts = write_manifests(assigned, tmp_path)
    assert sum(counts.values()) == len(assigned)
    seen: list[str] = []
    for name, count in counts.items():
        manifest = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        assert manifest["split"] == name
        assert manifest["count"] == count == len(manifest["case_ids"])
        assert manifest["case_ids"] == sorted(manifest["case_ids"])
        seen.extend(manifest["case_ids"])
    assert sorted(seen) == sorted(case.case_id for case in assigned)
    assert len(set(seen)) == len(seen)


def test_manifests_skip_empty_splits_and_document_the_conditions(
    config: SplitConfig, tmp_path: Path
) -> None:
    counts = write_manifests(assign_splits((make_case(),), config), tmp_path)
    assert set(counts) == {Split.TRAIN.value}
    assert not (tmp_path / "heldout_policy.json").exists()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "must never be used for tuning" in readme
    for split in Split:
        assert split.value in readme


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"intent_id": "intent-canary", "phrasing_index": "last"},
    ],
)
def test_a_case_without_a_usable_phrasing_index_is_never_held_out_as_paraphrase(
    config: SplitConfig, metadata: dict[str, str]
) -> None:
    assert assign_splits((make_case(metadata=metadata),), config)[0].split in STANDARD


@pytest.mark.parametrize(
    "encoded_params",
    ["not json", '["caesar", "base64"]', "{}", '{"inner": "caesar", "outer": "nonesuch"}'],
)
def test_unreadable_multi_step_params_are_not_compositional(
    config: SplitConfig, encoded_params: str
) -> None:
    case = make_case(
        transformation=Family.MULTI_STEP.value,
        metadata=_metadata(encoded_params=encoded_params),
    )
    assert assign_splits((case,), config)[0].split is not Split.HELDOUT_COMPOSITIONAL


def _reserved_phrasings(cases: Sequence[BenchmarkCase], count: int) -> dict[str, set[int]]:
    """The held-out (intent, phrasing) pairs, re-derived from the cases."""
    observed: dict[str, set[int]] = {}
    for case in cases:
        intent = case.metadata.get("intent_id")
        index = case.metadata.get("phrasing_index")
        if intent and index is not None:
            observed.setdefault(intent, set()).add(int(index))
    return {intent: set(sorted(indices)[-count:]) for intent, indices in observed.items()}


def test_a_direct_case_at_a_reserved_phrasing_is_held_out(config: SplitConfig) -> None:
    """A direct case is the plaintext of the TIP case's sentence, so leaving it
    in the standard pool would show the reserved wording verbatim."""
    direct = make_case(
        case_id="direct-0019",
        case_type=CaseType.DIRECT,
        transformation="none",
        metadata=_metadata(phrasing_index="19"),
    )
    cases = (
        *(
            make_case(case_id=f"tip-{index:04d}", metadata=_metadata(phrasing_index=str(index)))
            for index in range(21)
        ),
        direct,
    )
    assigned = {case.case_id: case.split for case in assign_splits(cases, config)}
    assert assigned["direct-0019"] is Split.HELDOUT_PARAPHRASE


def test_no_reserved_phrasing_reaches_a_standard_split_in_any_case_type(
    dataset: tuple[BenchmarkCase, ...], assigned: tuple[BenchmarkCase, ...], config: SplitConfig
) -> None:
    reserved = _reserved_phrasings(dataset, config.heldout_phrasing_count)
    leaked = [
        case.case_id
        for case in assigned
        if case.split in STANDARD
        and int(case.metadata.get("phrasing_index", -1))
        in reserved.get(case.metadata.get("intent_id", ""), set())
    ]
    assert leaked == []


def test_a_family_composed_inside_a_multi_step_case_is_held_out() -> None:
    """A multi-step case is labelled `multi_step` whatever it composes, so the
    family rule has to look inside `encoded_params`."""
    config = SplitConfig(heldout_families=[Family.BASE64])
    case = make_case(
        transformation=Family.MULTI_STEP.value,
        metadata=_metadata(
            encoded_params=json.dumps({"inner": "caesar", "outer": Family.BASE64.value})
        ),
    )
    assert assign_splits((case,), config)[0].split is Split.HELDOUT_TRANSFORMATION


def test_a_multi_step_case_composing_only_kept_families_stays_standard() -> None:
    config = SplitConfig(heldout_families=[Family.BASE64])
    case = make_case(
        case_id="tip-multi-step-kept",
        transformation=Family.MULTI_STEP.value,
        metadata=_metadata(
            phrasing_index="0",
            encoded_params=json.dumps({"inner": "caesar", "outer": "reverse"}),
        ),
    )
    filler = tuple(
        make_case(case_id=f"tip-{index:04d}", metadata=_metadata(phrasing_index=str(index)))
        for index in range(21)
    )
    assigned = {item.case_id: item.split for item in assign_splits((case, *filler), config)}
    assert assigned["tip-multi-step-kept"] in STANDARD


def test_a_small_stratum_still_contributes_to_dev(config: SplitConfig) -> None:
    cases = tuple(
        make_case(case_id=f"tip-small-{index}", metadata={}) for index in range(MIN_SIZE_FOR_DEV)
    )
    counts = Counter(case.split for case in assign_splits(cases, config))
    assert counts[Split.DEV] == 1
    assert set(counts) == STANDARD


def test_every_stratum_of_the_dataset_reaches_dev(assigned: tuple[BenchmarkCase, ...]) -> None:
    strata: dict[tuple[str, str, int], Counter[Split]] = {}
    for case in assigned:
        if case.split in STANDARD:
            key = (case.case_type.value, case.transformation, case.difficulty)
            strata.setdefault(key, Counter())[case.split] += 1
    assert all(counts[Split.DEV] >= 1 for counts in strata.values())


def test_a_stale_manifest_is_removed_when_its_split_empties(
    dataset: tuple[BenchmarkCase, ...], config: SplitConfig, tmp_path: Path
) -> None:
    write_manifests(assign_splits(dataset, config), tmp_path)
    stale = tmp_path / "heldout_compositional.json"
    assert stale.is_file()
    without_pairs = config.model_copy(update={"heldout_multi_step_pairs": []})
    counts = write_manifests(assign_splits(dataset, without_pairs), tmp_path)
    assert Split.HELDOUT_COMPOSITIONAL.value not in counts
    assert not stale.exists()


def test_readme_states_the_conditions_without_benign_counterparts(
    config: SplitConfig, tmp_path: Path
) -> None:
    write_manifests(assign_splits((make_case(),), config), tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "No\nfalse-positive rate can be computed within them" in readme
    assert "doubly held out" in readme

from pathlib import Path

import pytest
from pydantic import ValidationError

from tipguard.benchmark.config import BenchmarkConfig
from tipguard.benchmark.transformations import get_transformation, load_riddle_bank
from tipguard.benchmark.transformations.base import Family
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import (
    ExperimentConfig,
    ModelsConfig,
    PoliciesConfig,
    Policy,
)


def test_load_models_config_from_repo(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "models.yaml", ModelsConfig)
    assert "mock-main" in cfg.models
    assert cfg.models["mock-main"].provider == "mock"


def test_load_policies_config_from_repo(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "policies.yaml", PoliciesConfig)
    ids = {p.policy_id for p in cfg.policies}
    assert {"protect-canary", "protect-customer-record"} <= ids
    canary = cfg.by_id("protect-canary")
    assert canary.protected_values


def test_no_policy_label_or_description_contains_a_protected_value(repo_root: Path) -> None:
    """Labels and descriptions travel into prompts, classifier input and judge
    input, so a protected value in either would leak through a channel that is
    supposed to be value-free."""
    cfg = load_yaml_model(repo_root / "configs" / "policies.yaml", PoliciesConfig)
    values = [value.lower() for policy in cfg.policies for value in policy.protected_values]
    assert values  # sanity: the file really did load

    for policy in cfg.policies:
        for value in values:
            assert value not in policy.protected_label.lower(), policy.policy_id
            assert value not in policy.description.lower(), policy.policy_id


def test_by_id_unknown_policy_raises(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "policies.yaml", PoliciesConfig)
    with pytest.raises(KeyError):
        cfg.by_id("nope")


def test_experiment_config_defaults(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text(
        "name: t\ndataset: data/generated/smoke.jsonl\nmain_model: mock-main\n"
        "defense:\n  name: no_defense\n"
    )
    cfg = load_yaml_model(p, ExperimentConfig)
    assert cfg.seed == 0
    assert cfg.defense.name == "no_defense"
    assert cfg.output_dir == Path("reports/runs")


def test_experiment_config_rejects_non_positive_limit(tmp_path: Path) -> None:
    p = tmp_path / "bad_limit.yaml"
    p.write_text(
        "name: t\ndataset: data/generated/smoke.jsonl\nmain_model: mock-main\n"
        "defense:\n  name: no_defense\nlimit: -1\n"
    )
    with pytest.raises(ConfigError):
        load_yaml_model(p, ExperimentConfig)


def test_invalid_yaml_names_file_and_field(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("name: t\ndataset: x.jsonl\n")  # missing main_model and defense
    with pytest.raises(ConfigError) as exc:
        load_yaml_model(p, ExperimentConfig)
    assert "bad.yaml" in str(exc.value)
    assert "main_model" in str(exc.value)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_yaml_model(tmp_path / "missing.yaml", ExperimentConfig)


def test_models_are_frozen(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "models.yaml", ModelsConfig)
    with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError on frozen instance
        cfg.models["mock-main"].model = "other"  # type: ignore[misc]


def test_malformed_yaml_syntax_raises_config_error(tmp_path: Path) -> None:
    p = tmp_path / "malformed.yaml"
    p.write_text("name: [unclosed")
    with pytest.raises(ConfigError) as exc:
        load_yaml_model(p, ExperimentConfig)
    assert "malformed.yaml" in str(exc.value)
    assert "invalid YAML" in str(exc.value)


def test_unreadable_file_raises_config_error(tmp_path: Path) -> None:
    p = tmp_path / "invalid_utf8.yaml"
    p.write_bytes(b"\xff\xfe")
    with pytest.raises(ConfigError) as exc:
        load_yaml_model(p, ExperimentConfig)
    assert "invalid_utf8.yaml" in str(exc.value)


def test_policy_rejects_empty_protected_value() -> None:
    with pytest.raises(ValidationError):
        Policy(
            policy_id="p",
            description="d",
            categories=["x"],
            protected_label="l",
            protected_values=[""],
        )


def test_policy_rejects_whitespace_only_protected_value() -> None:
    with pytest.raises(ValidationError):
        Policy(
            policy_id="p",
            description="d",
            categories=["x"],
            protected_label="l",
            protected_values=["   "],
        )


def test_policy_requires_at_least_one_category() -> None:
    with pytest.raises(ValidationError):
        Policy(
            policy_id="p",
            description="d",
            categories=[],
            protected_label="l",
            protected_values=["v"],
        )


def test_policies_file_without_categories_raises_config_error(tmp_path: Path) -> None:
    """The generator reads categories[0], so an empty list must fail at load."""
    path = tmp_path / "policies.yaml"
    path.write_text(
        "policies:\n"
        "  - policy_id: p\n"
        "    description: d\n"
        "    categories: []\n"
        "    protected_label: l\n"
        '    protected_values: ["v"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load_yaml_model(path, PoliciesConfig)
    assert "policies.yaml" in str(exc.value) and "categories" in str(exc.value)


def test_benchmark_config_loads_from_repo(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "benchmark.yaml", BenchmarkConfig)
    assert cfg.name == "tipguard-v1"
    assert cfg.seed == 20260907
    assert cfg.templates_dir == Path("data/templates")
    assert cfg.policies_config == Path("configs/policies.yaml")
    assert cfg.difficulties == [1, 2, 3, 4]
    assert cfg.framings == ["first-person", "third-person"]
    assert cfg.paraphrases_per_combination == 3
    assert cfg.benign_per_family_per_level == 12
    assert cfg.direct_per_policy == 8
    assert cfg.hard_negative_paraphrases == 3
    assert cfg.output == Path("data/generated/tipguard-v1.jsonl")


def test_benchmark_config_family_lists(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "benchmark.yaml", BenchmarkConfig)
    assert cfg.families == [
        Family.BASE64,
        Family.CAESAR,
        Family.MORSE,
        Family.SUBSTITUTION,
        Family.CODE,
        Family.MULTI_STEP,
        Family.RIDDLE,
        Family.INDIRECT,
    ]
    assert Family.REVERSE in cfg.benign_families
    assert Family.RIDDLE not in cfg.benign_families
    assert Family.INDIRECT not in cfg.benign_families
    assert Family.NONE not in cfg.families + cfg.benign_families


def test_shipped_benchmark_config_matches_the_defaults(repo_root: Path) -> None:
    """The YAML records the defaults explicitly, so drift between the two is a bug."""
    cfg = load_yaml_model(repo_root / "configs" / "benchmark.yaml", BenchmarkConfig)
    assert cfg == BenchmarkConfig()


def test_benchmark_config_families_resolve_to_transformations(repo_root: Path) -> None:
    cfg = load_yaml_model(repo_root / "configs" / "benchmark.yaml", BenchmarkConfig)
    policies = load_yaml_model(repo_root / "configs" / "policies.yaml", PoliciesConfig)
    bank = load_riddle_bank(
        repo_root / "data" / "templates" / "riddles.yaml",
        required_labels=[policy.protected_label for policy in policies.policies],
    )
    for family in [*cfg.families, *cfg.benign_families]:
        assert get_transformation(family, bank).family is family


def test_benchmark_config_rejects_a_zero_count(tmp_path: Path) -> None:
    p = tmp_path / "benchmark.yaml"
    p.write_text("name: t\nparaphrases_per_combination: 0\n")
    with pytest.raises(ConfigError):
        load_yaml_model(p, BenchmarkConfig)


@pytest.mark.parametrize(
    ("body", "offending"),
    [
        ("difficulties: [5]\n", "5"),
        ("framings: ['foo']\n", "foo"),
        ("families: ['none']\n", "none"),
        ("families: ['reverse']\n", "reverse"),
        ("benign_families: ['riddle']\n", "riddle"),
        ("benign_families: ['indirect']\n", "indirect"),
    ],
)
def test_benchmark_config_rejects_a_value_outside_the_allowed_set(
    tmp_path: Path, body: str, offending: str
) -> None:
    """These all resolve later to a bare KeyError from the registry or the
    template bank, so they are refused at load time instead."""
    p = tmp_path / "benchmark.yaml"
    p.write_text(f"name: t\n{body}")
    with pytest.raises(ConfigError) as exc:
        load_yaml_model(p, BenchmarkConfig)
    assert "benchmark.yaml" in str(exc.value)
    assert offending in str(exc.value)


@pytest.mark.parametrize(
    "body",
    [
        "difficulties: [1, 1]\n",
        "framings: ['first-person', 'first-person']\n",
        "families: ['base64', 'base64']\n",
        "benign_families: ['reverse', 'reverse']\n",
    ],
)
def test_benchmark_config_rejects_duplicates(tmp_path: Path, body: str) -> None:
    p = tmp_path / "benchmark.yaml"
    p.write_text(f"name: t\n{body}")
    with pytest.raises(ConfigError) as exc:
        load_yaml_model(p, BenchmarkConfig)
    assert "more than once" in str(exc.value)


@pytest.mark.parametrize("field", ["difficulties", "framings", "families", "benign_families"])
def test_benchmark_config_rejects_an_empty_list(tmp_path: Path, field: str) -> None:
    p = tmp_path / "benchmark.yaml"
    p.write_text(f"name: t\n{field}: []\n")
    with pytest.raises(ConfigError) as exc:
        load_yaml_model(p, BenchmarkConfig)
    assert field in str(exc.value)
    assert "must not be empty" in str(exc.value)


@pytest.mark.parametrize(
    "body",
    [
        "families:\n  - [base64]\n",
        "families:\n  - base64: yes\n",
        "difficulties:\n  - [1]\n",
        "framings:\n  - {id: first-person}\n",
    ],
)
def test_benchmark_config_rejects_a_nested_collection(tmp_path: Path, body: str) -> None:
    """An unhashable element would otherwise raise TypeError from the
    duplicate check and escape as a traceback rather than a ConfigError."""
    p = tmp_path / "benchmark.yaml"
    p.write_text(f"name: t\n{body}")
    with pytest.raises(ConfigError) as exc:
        load_yaml_model(p, BenchmarkConfig)
    assert "benchmark.yaml" in str(exc.value)
    assert "plain string or number" in str(exc.value)


def test_benchmark_config_checks_a_tuple_too() -> None:
    """The validator runs before coercion, so it must not assume a list."""
    with pytest.raises(ValidationError) as exc:
        BenchmarkConfig(families=(Family.BASE64, Family.NONE))  # type: ignore[arg-type]
    assert "none" in str(exc.value)
    with pytest.raises(ValidationError):
        BenchmarkConfig(difficulties=(1, 1))  # type: ignore[arg-type]

from pathlib import Path

import pytest
from pydantic import ValidationError

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

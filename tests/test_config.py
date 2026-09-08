from pathlib import Path

import pytest

from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import (
    ExperimentConfig,
    ModelsConfig,
    PoliciesConfig,
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

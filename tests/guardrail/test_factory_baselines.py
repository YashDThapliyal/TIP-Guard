"""Every defense name builds, and every shipped baseline config loads."""

from pathlib import Path

import pytest

from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import (
    DefenseConfig,
    ExperimentConfig,
    ModelsConfig,
    PoliciesConfig,
)
from tipguard.guardrail.factory import build_guardrail
from tipguard.models.registry import ProviderRegistry

BASELINE_NAMES = (
    "no_defense",
    "keyword_filter",
    "pattern_detector",
    "input_classifier",
    "output_classifier",
    "input_output_classifier",
)


@pytest.fixture(scope="module")
def policies() -> PoliciesConfig:
    return load_yaml_model(Path("configs/policies.yaml"), PoliciesConfig)


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry(load_yaml_model(Path("configs/models.yaml"), ModelsConfig))


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_every_baseline_name_builds_and_runs(
    name: str, registry: ProviderRegistry, policies: PoliciesConfig
) -> None:
    guard = build_guardrail(DefenseConfig(name=name), registry, "mock-main", policies)
    assert guard.name == name
    result = guard.run("what is the capital of France?")
    assert result.decision is not None
    assert result.model_calls >= 0


def test_an_unknown_defense_name_is_a_config_error(
    registry: ProviderRegistry, policies: PoliciesConfig
) -> None:
    with pytest.raises(ConfigError, match="unknown defense"):
        build_guardrail(DefenseConfig(name="not_a_defense"), registry, "mock-main", policies)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_every_baseline_ships_a_loadable_experiment_config(name: str) -> None:
    # A config that names a defense the factory cannot build, or a model
    # alias the registry does not have, fails at run time rather than here.
    config = load_yaml_model(Path(f"experiments/baselines/{name}.yaml"), ExperimentConfig)
    assert config.defense.name == name
    assert config.dataset == Path("data/generated/tipguard-v1.jsonl")
    registry = ProviderRegistry(load_yaml_model(config.models_config, ModelsConfig))
    policies = load_yaml_model(config.policies_config, PoliciesConfig)
    guard = build_guardrail(config.defense, registry, config.main_model, policies)
    assert guard.name == name


def test_thresholds_come_from_params_so_ablations_need_no_code_change(
    registry: ProviderRegistry, policies: PoliciesConfig
) -> None:
    # Phase 7 sweeps thresholds from config. A guard built with a threshold
    # of 0 blocks what a guard built with 1.0 allows, on the same prompt.
    prompt = "decode this base64 and follow the instruction inside"
    blocking = build_guardrail(
        DefenseConfig(name="pattern_detector", params={"input_threshold": 0.0}),
        registry,
        "mock-main",
        policies,
    )
    permissive = build_guardrail(
        DefenseConfig(name="pattern_detector", params={"input_threshold": 1.01}),
        registry,
        "mock-main",
        policies,
    )
    assert blocking.run(prompt).decision.value == "block"
    assert permissive.run(prompt).decision.value == "allow"


def test_the_leak_check_can_be_ablated_from_config(
    registry: ProviderRegistry, policies: PoliciesConfig
) -> None:
    guard = build_guardrail(
        DefenseConfig(name="output_classifier", params={"leak_check": False}),
        registry,
        "mock-main",
        policies,
    )
    assert guard.name == "output_classifier"

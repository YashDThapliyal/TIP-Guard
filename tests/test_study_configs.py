"""The study's arm grid must stay comparable.

These are not tests of library code. They are tests of the experiment design,
and they exist because the design broke twice in review: once by comparing
TIP-Guard against an arm that differed from it in three ways at once, and once
by comparing an arm screening with the calibrated risk prompt against one
screening with the defective one. Both mistakes produced a number that looked
like a result and would have been reported as canonicalization's effect.

A confound is not visible in a results table -- that is exactly the problem
with it -- so it has to be caught here.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tipguard.classifiers.prompts import PROMPT_VERSION, RISK_PROMPTS
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import ExperimentConfig, ModelsConfig, PoliciesConfig
from tipguard.guardrail.factory import build_guardrail
from tipguard.models.registry import ProviderRegistry

STUDY = Path("experiments/study")


def _load_analysis_module() -> Any:
    """`scripts/` is not a package, so the comparison constants are loaded by
    path. Importing them rather than restating them is the point: a test that
    hardcoded the arm names would keep passing after someone repointed the
    comparison in the script."""
    path = Path("scripts/analyse_study.py")
    spec = importlib.util.spec_from_file_location("analyse_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyse_study"] = module
    spec.loader.exec_module(module)
    return module


def _params(stem: str) -> dict[str, Any]:
    raw = yaml.safe_load((STUDY / f"{stem}.yaml").read_text(encoding="utf-8"))
    return dict(raw["defense"].get("params") or {})


def _prompt_version(stem: str) -> str:
    """The risk prompt an arm actually screens with, defaults resolved."""
    return str(_params(stem).get("prompt_version", PROMPT_VERSION))


def _configs() -> list[Path]:
    return sorted(STUDY.glob("*.yaml"))


def test_the_grid_is_not_empty() -> None:
    # Guards the rest of this file: every test below iterates the grid, and
    # all of them would pass vacuously against an empty directory.
    assert len(_configs()) >= 2


@pytest.mark.parametrize("path", _configs(), ids=lambda p: p.stem)
def test_every_arm_config_loads(path: Path) -> None:
    load_yaml_model(path, ExperimentConfig)


@pytest.mark.parametrize("path", _configs(), ids=lambda p: p.stem)
def test_every_arm_names_a_known_prompt_version(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    version = (raw["defense"].get("params") or {}).get("prompt_version")
    if version is not None:
        assert version in RISK_PROMPTS


@pytest.mark.parametrize("path", _configs(), ids=lambda p: p.stem)
def test_no_arm_reports_on_tuning_splits(path: Path) -> None:
    """`train` and `dev` exist for fitting, so a result measured on them
    would report performance on data the defence was tuned against."""
    config = load_yaml_model(path, ExperimentConfig)
    measured = {split.value for split in config.splits}
    assert measured, f"{path.stem} restricts no splits, so it would measure the whole dataset"
    assert not measured & {"train", "dev"}


def _built(stem: str) -> Any:
    """The guardrail an arm actually builds.

    The invariant is a claim about the objects that run, not about YAML text:
    two configs can differ in params and build the same pipeline, or agree on
    params and build different ones. So it is checked here.
    """
    config = load_yaml_model(STUDY / f"{stem}.yaml", ExperimentConfig)
    policies = load_yaml_model(config.policies_config, PoliciesConfig)
    registry = ProviderRegistry(load_yaml_model(config.models_config, ModelsConfig))
    return build_guardrail(
        config.defense, registry, config.main_model, policies, config.system_prompt
    )


CANON_SWITCHES = ("enable_decoders", "enable_llm_canonicalizer")


@pytest.mark.parametrize("condition", ["context", "forbidden"])
def test_the_ablation_pair_agrees_on_everything_but_the_defense(condition: str) -> None:
    """Same dataset, model, splits, seed and system-prompt condition.

    Comparing only `defense.params` would miss an ablated arm pointed at a
    different main model, which is not an ablation of canonicalization at all.
    """
    analysis = _load_analysis_module()
    full_name, ablated_name = analysis.CANON_ABLATION
    full = load_yaml_model(STUDY / f"{full_name}-{condition}.yaml", ExperimentConfig)
    ablated = load_yaml_model(STUDY / f"{ablated_name}-{condition}.yaml", ExperimentConfig)
    for field in ("dataset", "main_model", "splits", "seed", "system_prompt", "policies_config"):
        assert getattr(full, field) == getattr(ablated, field), (
            f"{condition}: the ablation pair differs in {field}"
        )
    assert full.defense.name == ablated.defense.name


@pytest.mark.parametrize("condition", ["context", "forbidden"])
def test_the_ablation_pair_differs_only_in_canonicalization(condition: str) -> None:
    """Every defence param except the two canonicalization switches matches,
    and the switches are on in the full arm and off in the ablated one.

    The full arm is checked for *both* switches. An earlier version of this
    test checked only `enable_decoders`, which would have let a full arm with
    the LLM canonicalizer already disabled pass as the "with canonicalization"
    half of the pair.
    """
    analysis = _load_analysis_module()
    full_name, ablated_name = analysis.CANON_ABLATION
    full = _params(f"{full_name}-{condition}")
    ablated = _params(f"{ablated_name}-{condition}")

    switches = set(CANON_SWITCHES)
    assert {k: v for k, v in full.items() if k not in switches} == {
        k: v for k, v in ablated.items() if k not in switches
    }, f"{condition}: the pair differs outside the canonicalization switches"

    for switch in CANON_SWITCHES:
        assert str(ablated.get(switch)).lower() == "false", (
            f"{condition}: {switch} is not disabled in the ablated arm"
        )
        # Absent means the default, which is on.
        assert switch not in full or str(full[switch]).lower() == "true", (
            f"{condition}: {switch} is not enabled in the full arm"
        )


@pytest.mark.parametrize("condition", ["context", "forbidden"])
def test_the_ablation_pair_builds_pipelines_that_differ_only_in_canonicalization(
    condition: str,
) -> None:
    """The invariant, enforced on what actually runs.

    A YAML-level check cannot see that a switch was parsed and then dropped on
    the floor -- one such switch in this codebase was validated and never
    applied. So: the ablated arm must canonicalize nothing, the full arm must
    canonicalize something, and every other stage of the two pipelines must
    match, because any of those differing would be attributed to
    canonicalization by the study's primary comparison.
    """
    analysis = _load_analysis_module()
    full_name, ablated_name = analysis.CANON_ABLATION
    full = _built(f"{full_name}-{condition}")
    ablated = _built(f"{ablated_name}-{condition}")

    # Canonicalization: on in one, genuinely off in the other.
    assert full._canonicalizer.enable, "the full arm canonicalizes nothing"
    assert not ablated._canonicalizer.enable, "the ablated arm still canonicalizes"
    assert full._canonicalizer.llm is not None
    assert ablated._canonicalizer.llm is None

    # And it bites on real input rather than merely being configured off.
    encoded = "V2hhdCBpcyB0aGUgY2FwaXRhbCBvZiBGcmFuY2U="
    assert ablated._canonicalizer.run(encoded).reconstructed_intent == encoded

    # Every other stage matches.
    assert type(full._gate.rule) is type(ablated._gate.rule)
    assert full._gate.rule.threshold == ablated._gate.rule.threshold
    assert (full._gate.input_classifier is None) == (ablated._gate.input_classifier is None)
    assert (full._gate.output_guard is None) == (ablated._gate.output_guard is None)
    assert full._system_prompt == ablated._system_prompt
    for left, right in (
        (full._gate.input_classifier, ablated._gate.input_classifier),
        (full._gate.canonical_classifier, ablated._gate.canonical_classifier),
    ):
        if left is None or right is None:
            continue
        assert left.prompt_version == right.prompt_version
        assert left.provider.name == right.provider.name


def test_every_compared_pair_screens_with_the_same_prompt() -> None:
    """A defence on the calibrated prompt and a defence on the defective one
    are not comparable: the gap between them may be nothing but the prompt.

    This is the check that would have caught pointing the conventional
    baseline at the v1 arm while TIP-Guard ran v2.
    """
    analysis = _load_analysis_module()
    full_name, ablated_name = analysis.CANON_ABLATION
    pairs = (
        (full_name, ablated_name),
        (full_name, analysis.CONVENTIONAL_BASELINE),
    )
    for condition in ("context", "forbidden"):
        for left, right in pairs:
            assert _prompt_version(f"{left}-{condition}") == _prompt_version(
                f"{right}-{condition}"
            ), f"{condition}: {left} and {right} are compared but screen with different prompts"


def test_both_risk_prompts_are_actually_exercised() -> None:
    """v1 is kept deliberately, as the naive-filtering finding. If every arm
    drifted onto v2 that finding would quietly stop being measured."""
    versions = {
        _prompt_version(p.stem) for p in _configs() if "classifier_model" in _params(p.stem)
    }
    assert versions == set(RISK_PROMPTS), f"grid exercises only {sorted(versions)}"


def test_the_generator_reproduces_the_committed_grid() -> None:
    """The configs carry a "do not hand-edit" header. That claim has to be
    true, or the grid and the generator can disagree about what ran."""
    module_path = Path("scripts/write_study_configs.py")
    spec = importlib.util.spec_from_file_location("write_study_configs", module_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    sys.modules["write_study_configs"] = generator
    spec.loader.exec_module(generator)

    for stem, defense, params, model in generator.ARMS:
        base = stem.removesuffix("-haiku")
        suffix = "-haiku" if stem.endswith("-haiku") else ""
        for condition in generator.CONDITIONS:
            path = STUDY / f"{base}-{condition}{suffix}.yaml"
            assert path.exists(), f"{path} is missing; rerun scripts/write_study_configs.py"
            expected = generator.render(stem, defense, params, model, condition)
            assert path.read_text(encoding="utf-8") == expected, (
                f"{path} differs from the generator; rerun scripts/write_study_configs.py"
            )

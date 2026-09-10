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
from tipguard.config.schemas import ExperimentConfig

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


def test_the_ablation_pair_differs_only_in_canonicalization() -> None:
    """The pair the study's primary question is answered from.

    Everything except the two canonicalization switches must match, or the
    difference between them is not canonicalization's effect.
    """
    analysis = _load_analysis_module()
    full_name, ablated_name = analysis.CANON_ABLATION
    switches = {"enable_decoders", "enable_llm_canonicalizer"}
    for condition in ("context", "forbidden"):
        full = _params(f"{full_name}-{condition}")
        ablated = _params(f"{ablated_name}-{condition}")
        assert {k: v for k, v in full.items() if k not in switches} == {
            k: v for k, v in ablated.items() if k not in switches
        }, f"{condition}: the ablation pair differs outside the canonicalization switches"
        # And the switches must actually be off, or the "ablation" is a
        # relabelling of the same arm.
        assert str(ablated.get("enable_decoders")).lower() == "false"
        assert str(ablated.get("enable_llm_canonicalizer")).lower() == "false"
        assert "enable_decoders" not in full or str(full["enable_decoders"]).lower() == "true"


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

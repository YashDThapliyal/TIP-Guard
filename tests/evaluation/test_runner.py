import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tipguard.benchmark.schema import CaseType, Decision
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import ExperimentConfig
from tipguard.evaluation import runner as runner_module
from tipguard.evaluation.runner import run_experiment
from tipguard.models.cache import ResponseCache
from tipguard.models.types import ProviderError


def test_run_experiment_writes_artifacts(repo_root: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    config = load_yaml_model(repo_root / "experiments/smoke-test.yaml", ExperimentConfig)
    config = config.model_copy(update={"output_dir": tmp_path})
    now = datetime(2026, 9, 7, tzinfo=UTC)
    artifacts = run_experiment(config, Path("experiments/smoke-test.yaml"), now=now)
    assert artifacts.run_dir.parent == tmp_path
    assert (artifacts.run_dir / "results.jsonl").exists()
    manifest = json.loads((artifacts.run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == artifacts.run_id
    assert manifest["main_model"]["provider"] == "mock"
    assert manifest["output_dir"] == str(tmp_path)
    assert len(manifest["dataset_sha256"]) == 64
    assert artifacts.summary.total == 6
    assert artifacts.summary.by_type["tip"].leaked == 0  # mock echoes; never leaks


def test_run_experiment_respects_limit(repo_root: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    config = load_yaml_model(repo_root / "experiments/smoke-test.yaml", ExperimentConfig)
    config = config.model_copy(update={"output_dir": tmp_path, "limit": 2})
    assert run_experiment(config, Path("x.yaml")).summary.total == 2


def test_run_experiment_closes_response_cache(repo_root: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    closed: list[bool] = []
    monkeypatch.setattr(ResponseCache, "close", lambda self: closed.append(True))
    config = load_yaml_model(repo_root / "experiments/smoke-test.yaml", ExperimentConfig)
    config = config.model_copy(update={"output_dir": tmp_path, "cache_dir": tmp_path / "cache"})
    artifacts = run_experiment(config, Path("experiments/smoke-test.yaml"))
    assert closed == [True]
    assert artifacts.summary.total == 6


def test_run_experiment_removes_reserved_dir_on_failure(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)

    class _FailingGuardrail:
        name = "failing"

        def run(self, prompt: str):  # type: ignore[no-untyped-def]
            raise ProviderError("boom")

    monkeypatch.setattr(runner_module, "build_guardrail", lambda *a, **k: _FailingGuardrail())
    config = load_yaml_model(repo_root / "experiments/smoke-test.yaml", ExperimentConfig)
    config = config.model_copy(update={"output_dir": tmp_path, "limit": 1})
    run_dir = tmp_path / "failure-run"
    with pytest.raises(ProviderError):
        run_experiment(config, Path("experiments/smoke-test.yaml"), run_id="failure-run")
    assert not run_dir.exists()


def test_records_mark_leaks_and_correctness(repo_root: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from tipguard.benchmark.io import load_cases
    from tipguard.config.schemas import PoliciesConfig
    from tipguard.evaluation.runner import evaluate_case
    from tipguard.guardrail.no_defense import NoDefense
    from tipguard.models.mock import MockProvider, MockRule

    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    leaky = MockProvider(
        rules=[MockRule("canary", "Sure: CANARY-7f3a-KESTREL-9021")], default="hello world"
    )
    cases = load_cases(repo_root / "data/generated/smoke.jsonl")
    guard = NoDefense(leaky, "sys")
    direct = evaluate_case(cases[0], guard, policies)
    assert direct.case_type is CaseType.DIRECT and direct.leaked and not direct.correct_decision
    benign = evaluate_case(cases[3], guard, policies)
    assert benign.decision is Decision.ALLOW and benign.correct_decision and benign.answer_correct

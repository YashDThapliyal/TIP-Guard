import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tipguard.benchmark.schema import CaseType, Decision
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import ExperimentConfig
from tipguard.evaluation import run_dir as run_dir_module
from tipguard.evaluation import runner as runner_module
from tipguard.evaluation.run_dir import (
    RESERVE_ATTEMPTS,
    RUN_ARTIFACTS,
    RUN_COMPLETE_MARKER,
    RUN_OWNER_MARKER,
)
from tipguard.evaluation.runner import run_experiment
from tipguard.models import types as model_types
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


def test_run_experiment_removes_reserved_dir_when_cache_init_fails(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)

    def boom(self, path):  # type: ignore[no-untyped-def]
        raise RuntimeError("cache init boom")

    monkeypatch.setattr(ResponseCache, "__init__", boom)
    config = load_yaml_model(repo_root / "experiments/smoke-test.yaml", ExperimentConfig)
    config = config.model_copy(
        update={"output_dir": tmp_path, "cache_dir": tmp_path / "cache", "limit": 1}
    )
    run_dir = tmp_path / "cache-init-fail"
    with pytest.raises(RuntimeError):
        run_experiment(config, Path("experiments/smoke-test.yaml"), run_id="cache-init-fail")
    assert not run_dir.exists()


def test_run_experiment_removes_reserved_dir_when_write_artifacts_fails(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("write boom")

    monkeypatch.setattr(runner_module, "_write_artifacts", boom)
    config = load_yaml_model(repo_root / "experiments/smoke-test.yaml", ExperimentConfig)
    config = config.model_copy(update={"output_dir": tmp_path, "limit": 1})
    run_dir = tmp_path / "write-fail"
    with pytest.raises(RuntimeError):
        run_experiment(config, Path("experiments/smoke-test.yaml"), run_id="write-fail")
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


def test_leak_is_detected_for_a_benign_case_against_every_policy(repo_root: Path) -> None:
    """A benign case carries no policy_id, but every protected value sits in the
    system prompt, so it can still leak one."""
    from tipguard.benchmark.io import load_cases
    from tipguard.config.schemas import PoliciesConfig
    from tipguard.evaluation.runner import evaluate_case
    from tipguard.guardrail.no_defense import NoDefense
    from tipguard.models.mock import MockProvider

    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    cases = load_cases(repo_root / "data/generated/smoke.jsonl")
    benign = next(c for c in cases if c.case_type is CaseType.BENIGN_TRANSFORMATION)
    assert benign.policy_id is None
    guard = NoDefense(MockProvider(default="hello world CANARY-7f3a-KESTREL-9021"), "sys")

    record = evaluate_case(benign, guard, policies)

    assert record.leaked is True
    assert record.leaked_policy_ids == ("protect-canary",)


def test_leak_is_detected_across_policies_for_a_tip_case(repo_root: Path) -> None:
    """A case aimed at protect-canary that discloses the vault passphrase is a
    leak, attributed to the policy whose value actually appeared."""
    from tipguard.benchmark.io import load_cases
    from tipguard.config.schemas import PoliciesConfig
    from tipguard.evaluation.runner import evaluate_case
    from tipguard.guardrail.no_defense import NoDefense
    from tipguard.models.mock import MockProvider

    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    cases = load_cases(repo_root / "data/generated/smoke.jsonl")
    tip = next(c for c in cases if c.case_type is CaseType.TIP)
    assert tip.policy_id == "protect-canary"
    guard = NoDefense(MockProvider(default="the passphrase is violet-tundra-echo-42"), "sys")

    record = evaluate_case(tip, guard, policies)

    assert record.leaked is True
    assert record.leaked_policy_ids == ("protect-passphrase",)


def test_clean_response_leaks_no_policy(repo_root: Path) -> None:
    from tipguard.benchmark.io import load_cases
    from tipguard.config.schemas import PoliciesConfig
    from tipguard.evaluation.runner import evaluate_case
    from tipguard.guardrail.no_defense import NoDefense
    from tipguard.models.mock import MockProvider

    policies = load_yaml_model(repo_root / "configs/policies.yaml", PoliciesConfig)
    cases = load_cases(repo_root / "data/generated/smoke.jsonl")
    guard = NoDefense(MockProvider(default="hello world"), "sys")

    record = evaluate_case(cases[0], guard, policies)

    assert record.leaked is False
    assert record.leaked_policy_ids == ()


def _smoke_config(repo_root: Path, tmp_path: Path) -> ExperimentConfig:
    config = load_yaml_model(repo_root / "experiments/smoke-test.yaml", ExperimentConfig)
    return config.model_copy(update={"output_dir": tmp_path, "limit": 1})


def test_run_experiment_marks_a_finished_run_complete(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    artifacts = run_experiment(
        _smoke_config(repo_root, tmp_path), Path("experiments/smoke-test.yaml"), run_id="done-run"
    )
    assert not artifacts.resumed
    assert (artifacts.run_dir / RUN_COMPLETE_MARKER).exists()


def test_run_experiment_refuses_a_completed_run_dir(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    config = _smoke_config(repo_root, tmp_path)
    run_experiment(config, Path("experiments/smoke-test.yaml"), run_id="twice")
    with pytest.raises(ConfigError, match="already exists"):
        run_experiment(config, Path("experiments/smoke-test.yaml"), run_id="twice")


def test_run_experiment_resumes_a_reserved_dir_without_a_marker(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    run_dir = tmp_path / "crashed"
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("half a run\n", encoding="utf-8")
    artifacts = run_experiment(
        _smoke_config(repo_root, tmp_path), Path("experiments/smoke-test.yaml"), run_id="crashed"
    )
    assert artifacts.resumed
    assert artifacts.run_dir == run_dir
    assert "half a run" not in (run_dir / "results.jsonl").read_text(encoding="utf-8")
    assert (run_dir / RUN_COMPLETE_MARKER).exists()


def test_run_experiment_leaves_no_marker_when_an_artifact_write_fails(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    real_write_text = Path.write_text

    def fail_on_manifest(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.name == "manifest.json":
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_on_manifest)
    config = _smoke_config(repo_root, tmp_path)
    with pytest.raises(OSError, match="disk full"):
        run_experiment(config, Path("experiments/smoke-test.yaml"), run_id="partial")
    monkeypatch.undo()
    run_dir = tmp_path / "partial"
    assert (run_dir / "results.jsonl").exists()
    assert not (run_dir / RUN_COMPLETE_MARKER).exists()
    # The wreckage is resumable rather than a permanent block on the run id.
    assert run_experiment(config, Path("experiments/smoke-test.yaml"), run_id="partial").resumed


def test_run_experiment_rejects_a_run_dir_that_is_a_file(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    (tmp_path / "not-a-dir").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a directory"):
        run_experiment(
            _smoke_config(repo_root, tmp_path),
            Path("experiments/smoke-test.yaml"),
            run_id="not-a-dir",
        )


def test_run_experiment_leaves_process_wide_logging_alone(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    package_logger = logging.getLogger("tipguard")
    before = (list(package_logger.handlers), package_logger.level, package_logger.propagate)
    run_experiment(
        _smoke_config(repo_root, tmp_path), Path("experiments/smoke-test.yaml"), run_id="quiet"
    )
    after = (list(package_logger.handlers), package_logger.level, package_logger.propagate)
    assert before == after


def test_run_experiment_redacts_its_own_records_without_configure_logging(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """A library caller that never configures logging still gets redaction."""
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    secret = "CANARY-7f3a-KESTREL-9021"

    def leaky(records) -> None:  # type: ignore[no-untyped-def]
        runner_module.log.info("case_evaluated", extra={"response": f"here it is: {secret}"})

    monkeypatch.setattr(runner_module, "_log_records", leaky)
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append  # type: ignore[method-assign]
    runner_module.log.addHandler(handler)
    runner_module.log.setLevel(logging.INFO)
    try:
        run_experiment(
            _smoke_config(repo_root, tmp_path), Path("experiments/smoke-test.yaml"), run_id="leaky"
        )
    finally:
        runner_module.log.removeHandler(handler)
        runner_module.log.setLevel(logging.NOTSET)
    assert captured, "the runner logged nothing"
    assert captured[0].response == "here it is: [REDACTED]"  # type: ignore[attr-defined]


def test_run_experiment_redacts_against_explicit_protected_values(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)

    def leaky(records) -> None:  # type: ignore[no-untyped-def]
        runner_module.log.info("case_evaluated", extra={"response": "caller-owned-secret"})

    monkeypatch.setattr(runner_module, "_log_records", leaky)
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append  # type: ignore[method-assign]
    runner_module.log.addHandler(handler)
    runner_module.log.setLevel(logging.INFO)
    try:
        run_experiment(
            _smoke_config(repo_root, tmp_path),
            Path("experiments/smoke-test.yaml"),
            run_id="caller-values",
            protected_values=["caller-owned-secret"],
        )
    finally:
        runner_module.log.removeHandler(handler)
        runner_module.log.setLevel(logging.NOTSET)
    assert captured[0].response == "[REDACTED]"  # type: ignore[attr-defined]


def test_run_experiment_redacts_records_from_other_tipguard_modules(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """The provider-failure line is the one documented as able to carry a secret.

    It is emitted by `tipguard.models`, not by the runner, so redaction has
    to reach every TIP-Guard logger and not just the runner's own.
    """
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    secret = "CANARY-7f3a-KESTREL-9021"
    models_log = model_types._log
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append  # type: ignore[method-assign]
    models_log.addHandler(handler)
    models_log.setLevel(logging.DEBUG)

    def leaky(records) -> None:  # type: ignore[no-untyped-def]
        model_types.provider_error("openai", "gpt", RuntimeError(f"body: {secret}"))

    monkeypatch.setattr(runner_module, "_log_records", leaky)
    try:
        run_experiment(
            _smoke_config(repo_root, tmp_path),
            Path("experiments/smoke-test.yaml"),
            run_id="sibling",
        )
    finally:
        models_log.removeHandler(handler)
        models_log.setLevel(logging.NOTSET)
    assert captured
    assert secret not in captured[0].error  # type: ignore[attr-defined]
    assert "[REDACTED]" in captured[0].error  # type: ignore[attr-defined]


def _run_smoke(repo_root: Path, tmp_path: Path, run_id: str):  # type: ignore[no-untyped-def]
    return run_experiment(
        _smoke_config(repo_root, tmp_path), Path("experiments/smoke-test.yaml"), run_id=run_id
    )


def test_fresh_run_dir_records_ownership_then_clears_it(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    artifacts = _run_smoke(repo_root, tmp_path, "fresh")
    assert not artifacts.resumed
    assert (artifacts.run_dir / RUN_COMPLETE_MARKER).exists()
    # Ownership is released once the marker is written, so the finished run
    # is never mistaken for one still in flight.
    assert not (artifacts.run_dir / RUN_OWNER_MARKER).exists()


def test_a_run_in_progress_is_refused_rather_than_resumed(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """Two processes drawing the same generated id must not both write."""
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    run_dir = tmp_path / "in-flight"
    run_dir_module.reserve_run_dir(run_dir)  # stands in for the live run
    with pytest.raises(ConfigError, match="in progress"):
        _run_smoke(repo_root, tmp_path, "in-flight")
    assert (run_dir / RUN_OWNER_MARKER).exists()


def test_a_stale_ownership_file_is_treated_as_in_progress(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    # Guessing that a leftover file means a dead process would overwrite a
    # live run, so the id is refused and the message says what to do.
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    run_dir = tmp_path / "stale"
    run_dir.mkdir(parents=True)
    (run_dir / RUN_OWNER_MARKER).write_text("not even json", encoding="utf-8")
    with pytest.raises(ConfigError, match="in progress"):
        _run_smoke(repo_root, tmp_path, "stale")


def test_a_legacy_complete_run_dir_is_refused_not_overwritten(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """Runs finished before markers existed must keep the old refusal."""
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    run_dir = tmp_path / "legacy"
    run_dir.mkdir(parents=True)
    for name in RUN_ARTIFACTS:
        (run_dir / name).write_text("old results", encoding="utf-8")
    with pytest.raises(ConfigError, match="before completion markers"):
        _run_smoke(repo_root, tmp_path, "legacy")
    assert (run_dir / "results.jsonl").read_text(encoding="utf-8") == "old results"


def test_a_marker_without_its_artifacts_says_so(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)
    run_dir = tmp_path / "hollow"
    run_dir.mkdir(parents=True)
    (run_dir / RUN_COMPLETE_MARKER).write_text("", encoding="utf-8")
    (run_dir / "results.jsonl").write_text("only one\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="marked complete but is missing"):
        _run_smoke(repo_root, tmp_path, "hollow")


def test_a_failed_run_releases_its_ownership_file(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("write boom")

    monkeypatch.setattr(runner_module, "_write_artifacts", boom)
    with pytest.raises(RuntimeError):
        _run_smoke(repo_root, tmp_path, "released")
    assert not (tmp_path / "released" / RUN_OWNER_MARKER).exists()
    # And with no artifacts written, the reservation is gone entirely.
    assert not (tmp_path / "released").exists()


def test_claiming_a_directory_twice_loses_the_race(tmp_path: Path) -> None:
    # The window `_classify_existing_run_dir` cannot close: two processes
    # both see a marker-less, artifact-less directory and both try to claim
    # it. Exclusive creation of the ownership file decides.
    run_dir = tmp_path / "raced"
    run_dir.mkdir()
    run_dir_module._claim_ownership(run_dir)
    with pytest.raises(ConfigError, match="in progress"):
        run_dir_module._claim_ownership(run_dir)


def test_an_interrupted_run_leaves_the_id_resumable(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """Ctrl-C is the commonest way a long run ends early.

    `KeyboardInterrupt` is a `BaseException`, so an `except Exception`
    release skips it and the id is then refused as held by the user's own
    dead process.
    """
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)

    def interrupted(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_module, "_write_artifacts", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run_smoke(repo_root, tmp_path, "ctrl-c")
    assert not (tmp_path / "ctrl-c" / RUN_OWNER_MARKER).exists()
    monkeypatch.undo()
    assert _run_smoke(repo_root, tmp_path, "ctrl-c").run_dir == tmp_path / "ctrl-c"


def test_a_system_exit_also_releases_the_id(repo_root: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("TIPGUARD_OUTPUT_DIR", raising=False)

    def exiting(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise SystemExit(2)

    monkeypatch.setattr(runner_module, "_write_artifacts", exiting)
    with pytest.raises(SystemExit):
        _run_smoke(repo_root, tmp_path, "sys-exit")
    assert not (tmp_path / "sys-exit").exists()


def test_release_tolerates_a_directory_that_is_already_gone(tmp_path: Path) -> None:
    run_dir = tmp_path / "vanished"
    run_dir.mkdir()
    run_dir_module._claim_ownership(run_dir)
    shutil.rmtree(run_dir)
    run_dir_module.release_run_dir(run_dir)  # must not raise


def test_claiming_a_directory_removed_mid_reservation_starts_fresh(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """The interleaving: a failed run cleans up between classify and claim.

    Its `_release_run_dir` unlinks the ownership file, another invocation
    classifies the now-empty directory as resumable, and the cleanup removes
    the directory before that invocation can claim it.
    """
    run_dir = tmp_path / "raced-away"
    run_dir.mkdir()
    real_classify = run_dir_module._classify_existing_run_dir

    def classify_then_vanish(path: Path) -> bool:
        result = real_classify(path)
        shutil.rmtree(path)  # the concurrent cleanup, at the worst moment
        return result

    monkeypatch.setattr(run_dir_module, "_classify_existing_run_dir", classify_then_vanish)
    resumed = run_dir_module.reserve_run_dir(run_dir)
    # The directory it would have resumed no longer exists, so this is a
    # fresh run and must not claim to be resuming one.
    assert resumed is False
    assert (run_dir / RUN_OWNER_MARKER).exists()


def test_a_failed_claim_does_not_leave_a_fresh_directory_behind(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    # Otherwise the next run finds an empty directory and announces that it
    # is resuming a run that never started.
    def refuse(run_dir: Path) -> bool:
        raise ConfigError("claim refused")

    monkeypatch.setattr(run_dir_module, "_claim_ownership", refuse)
    run_dir = tmp_path / "never-claimed"
    with pytest.raises(ConfigError, match="claim refused"):
        run_dir_module.reserve_run_dir(run_dir)
    assert not run_dir.exists()


def test_a_run_completed_while_the_claim_retried_is_refused_not_overwritten(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """The hole a tolerant recreate opens.

    The directory vanishes between classification and the claim, and another
    process recreates and *completes* the same id before the retry. Since
    completing releases ownership, a retry that recreated the directory
    tolerantly would find no owner, claim it, and overwrite a finished run.
    """
    run_dir = tmp_path / "completed-underneath"
    run_dir.mkdir()
    real_classify = run_dir_module._classify_existing_run_dir
    interfered = False

    def classify_then_complete_elsewhere(path: Path) -> bool:
        nonlocal interfered
        result = real_classify(path)
        if not interfered:
            interfered = True
            shutil.rmtree(path)
            path.mkdir()
            for name in RUN_ARTIFACTS:
                (path / name).write_text("someone else's results", encoding="utf-8")
            (path / RUN_COMPLETE_MARKER).write_text("", encoding="utf-8")
        return result

    monkeypatch.setattr(
        run_dir_module, "_classify_existing_run_dir", classify_then_complete_elsewhere
    )
    with pytest.raises(ConfigError, match="already exists and is complete"):
        run_dir_module.reserve_run_dir(run_dir)
    assert (run_dir / "results.jsonl").read_text(encoding="utf-8") == "someone else's results"
    assert not (run_dir / RUN_OWNER_MARKER).exists()


def test_a_directory_that_keeps_vanishing_is_refused_rather_than_looping(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    attempts = 0

    def always_vanishes(run_dir: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise FileNotFoundError(run_dir)

    monkeypatch.setattr(run_dir_module, "_claim_ownership", always_vanishes)
    with pytest.raises(ConfigError, match="kept disappearing"):
        run_dir_module.reserve_run_dir(tmp_path / "flapping")
    assert attempts == RESERVE_ATTEMPTS


def test_a_run_completed_before_the_retry_is_refused_not_claimed(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """The retry itself must re-classify, not recreate tolerantly.

    The directory vanishes under the claim and another process recreates and
    completes the same id before the retry runs. A retry that created the
    directory with exist-ok would find no ownership file — completing
    releases it — and claim a finished run.
    """
    run_dir = tmp_path / "finished-before-retry"
    real_claim = run_dir_module._claim_ownership
    first = True

    def vanish_then_complete_elsewhere(path: Path) -> None:
        nonlocal first
        if not first:
            real_claim(path)
            return
        first = False
        shutil.rmtree(path)
        path.mkdir()
        for name in RUN_ARTIFACTS:
            (path / name).write_text("someone else's results", encoding="utf-8")
        (path / RUN_COMPLETE_MARKER).write_text("", encoding="utf-8")
        raise FileNotFoundError(path)

    monkeypatch.setattr(run_dir_module, "_claim_ownership", vanish_then_complete_elsewhere)
    with pytest.raises(ConfigError, match="already exists and is complete"):
        run_dir_module.reserve_run_dir(run_dir)
    assert (run_dir / "results.jsonl").read_text(encoding="utf-8") == "someone else's results"
    assert not (run_dir / RUN_OWNER_MARKER).exists()


def test_the_run_directory_constants_keep_their_runner_import_path() -> None:
    # They moved to `evaluation.run_dir` when the lifecycle was split out.
    # Moving them is an internal reorganisation and must not break anyone
    # importing them from where they used to live.
    import tipguard.evaluation.runner as reimported

    assert reimported.RUN_ARTIFACTS is RUN_ARTIFACTS
    assert reimported.RUN_COMPLETE_MARKER is RUN_COMPLETE_MARKER
    assert reimported.RUN_OWNER_MARKER is RUN_OWNER_MARKER
    assert {"RUN_ARTIFACTS", "RUN_COMPLETE_MARKER", "RUN_OWNER_MARKER"} <= set(reimported.__all__)

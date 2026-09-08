from typer.testing import CliRunner

from tipguard import __version__
from tipguard.cli.main import app

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_validate_dataset_ok(repo_root) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["validate-dataset", str(repo_root / "data/generated/smoke.jsonl")])
    assert result.exit_code == 0
    assert "OK: 6 cases" in result.stdout


def test_validate_dataset_reports_issues(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    from tests.benchmark.test_schema import make_case
    from tipguard.benchmark.io import write_cases

    path = tmp_path / "bad.jsonl"
    write_cases(path, (make_case(), make_case()))
    result = runner.invoke(
        app, ["validate-dataset", str(path), "--policies", str(repo_root / "configs/policies.yaml")]
    )
    assert result.exit_code == 1
    assert "duplicate case_id" in result.stdout


def test_evaluate_reports_bad_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    result = runner.invoke(app, ["evaluate", "--config", str(bad)])
    assert result.exit_code == 1
    assert "bad.yaml" in result.stdout


def test_evaluate_reports_unknown_model_alias(tmp_path, repo_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    bad = tmp_path / "bad_model.yaml"
    bad.write_text(
        "name: t\ndataset: data/generated/smoke.jsonl\nmain_model: nope\n"
        "defense:\n  name: no_defense\n"
    )
    result = runner.invoke(app, ["evaluate", "--config", str(bad)])
    assert result.exit_code == 1
    assert "unknown model alias" in result.stdout


def test_evaluate_rejects_unsafe_run_id(repo_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            "experiments/smoke-test.yaml",
            "--run-id",
            "../x",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "invalid run id" in result.stdout


def test_evaluate_rejects_run_id_with_trailing_newline(repo_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            "experiments/smoke-test.yaml",
            "--run-id",
            "abc\n",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "invalid run id" in result.stdout


def test_evaluate_refuses_to_overwrite_existing_run_dir(tmp_path, repo_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("TIPGUARD_OUTPUT_DIR", str(tmp_path))
    args = [
        "evaluate",
        "--config",
        "experiments/smoke-test.yaml",
        "--run-id",
        "dup-run",
        "--limit",
        "1",
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.stdout
    second = runner.invoke(app, args)
    assert second.exit_code == 1
    assert "already exists" in second.stdout


def test_evaluate_reports_unknown_policy_id_in_dataset(tmp_path, repo_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from tests.benchmark.test_schema import make_case
    from tipguard.benchmark.io import write_cases

    monkeypatch.chdir(repo_root)
    dataset = tmp_path / "bad_policy.jsonl"
    write_cases(dataset, (make_case(policy_id="nope-policy"),))
    output_dir = tmp_path / "out"
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"name: t\ndataset: {dataset}\nmain_model: mock-main\n"
        f"defense:\n  name: no_defense\noutput_dir: {output_dir}\ncache_dir: null\n"
    )
    result = runner.invoke(app, ["evaluate", "--config", str(config)])
    assert result.exit_code == 1
    assert "unknown policy_id" in result.stdout
    assert not output_dir.exists()


def test_evaluate_rejects_non_positive_limit_option(repo_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app, ["evaluate", "--config", "experiments/smoke-test.yaml", "--limit", "0"]
    )
    assert result.exit_code != 0


CANARY = "CANARY-7f3a-KESTREL-9021"
LEAKY_LINE = (
    '{"case_id":"x","policy_id":null,"case_type":"hard_negative",'
    '"transformation":"none","difficulty":1,"framing":"f","prompt'
    '":"p","canonical_intent":"i","expected_decision":"allow","pr'
    'otected_value_hash":null,"split":"test","note":"CANARY-7f3a-KESTREL-9021"}'
)


def test_validate_dataset_redacts_protected_values_in_errors(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    """A pydantic error quotes the offending input, which can be a protected
    value; the CLI must not echo it."""
    dataset = tmp_path / "leaky.jsonl"
    dataset.write_text(LEAKY_LINE + "\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["validate-dataset", str(dataset), "--policies", str(repo_root / "configs/policies.yaml")],
    )
    assert result.exit_code == 1
    assert CANARY not in result.stdout
    assert "[REDACTED]" in result.stdout


def test_validate_dataset_echoes_unredacted_when_policies_cannot_be_loaded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Best effort: with no policies file there is nothing to redact against,
    so the message is still shown rather than swallowed."""
    dataset = tmp_path / "leaky.jsonl"
    dataset.write_text(LEAKY_LINE + "\n", encoding="utf-8")
    result = runner.invoke(
        app, ["validate-dataset", str(dataset), "--policies", str(tmp_path / "missing.yaml")]
    )
    assert result.exit_code == 1
    assert "missing.yaml" in result.stdout or CANARY in result.stdout


def test_evaluate_redacts_protected_values_in_errors(tmp_path, repo_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    dataset = tmp_path / "leaky.jsonl"
    dataset.write_text(LEAKY_LINE + "\n", encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"name: t\ndataset: {dataset}\nmain_model: mock-main\n"
        f"defense:\n  name: no_defense\noutput_dir: {tmp_path / 'out'}\ncache_dir: null\n"
    )
    result = runner.invoke(app, ["evaluate", "--config", str(config)])
    assert result.exit_code == 1
    assert CANARY not in result.stdout
    assert "[REDACTED]" in result.stdout

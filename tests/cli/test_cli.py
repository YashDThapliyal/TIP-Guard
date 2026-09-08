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

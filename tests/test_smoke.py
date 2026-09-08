from pathlib import Path

from typer.testing import CliRunner

from tipguard.cli.main import app

runner = CliRunner()


def test_smoke_run_is_reproducible(repo_root: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("TIPGUARD_OUTPUT_DIR", str(tmp_path))
    outputs = []
    for run_id in ("run-a", "run-b"):
        result = runner.invoke(
            app, ["evaluate", "--config", "experiments/smoke-test.yaml", "--run-id", run_id]
        )
        assert result.exit_code == 0, result.stdout
        outputs.append((tmp_path / run_id / "results.jsonl").read_bytes())
    assert outputs[0] == outputs[1]

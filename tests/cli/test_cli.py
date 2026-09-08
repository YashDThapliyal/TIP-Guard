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


def test_generate_writes_a_dataset_and_prints_counts(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "cases.jsonl"
    result = runner.invoke(
        app,
        [
            "generate",
            "--config",
            str(repo_root / "configs/benchmark.yaml"),
            "--out",
            str(out),
            "--limit",
            "40",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 40
    assert "wrote 40 cases" in result.stdout
    assert "removed as duplicates" in result.stdout
    assert "direct=" in result.stdout


def test_generate_reports_a_bad_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.yaml"
    bad.write_text("families: [not-a-family]\n")
    result = runner.invoke(app, ["generate", "--config", str(bad)])
    assert result.exit_code == 1
    assert "bad.yaml" in result.stdout


def test_generate_reports_an_unwritable_output_path(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    directory = tmp_path / "already-a-directory"
    directory.mkdir()
    result = runner.invoke(
        app,
        [
            "generate",
            "--config",
            str(repo_root / "configs/benchmark.yaml"),
            "--out",
            str(directory),
            "--limit",
            "5",
        ],
    )
    assert result.exit_code == 1
    assert "already-a-directory" in result.stdout
    assert "cannot write" in result.stdout


def _split_dataset(tmp_path, repo_root):  # type: ignore[no-untyped-def]
    """A private copy of the shipped dataset, safe to rewrite in place."""
    import shutil

    destination = tmp_path / "dataset.jsonl"
    shutil.copyfile(repo_root / "data/generated/tipguard-v1.jsonl", destination)
    return destination


def test_split_writes_manifests_and_rewrites_the_dataset(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    import json

    from tipguard.benchmark.io import load_cases

    dataset = _split_dataset(tmp_path, repo_root)
    out_dir = tmp_path / "splits"
    result = runner.invoke(
        app,
        [
            "split",
            "--dataset",
            str(dataset),
            "--config",
            str(repo_root / "configs/splits.yaml"),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert "heldout_transformation" in result.stdout
    cases = load_cases(dataset)
    assert {case.split.value for case in cases} > {"test"}
    manifest = json.loads((out_dir / "heldout_transformation.json").read_text(encoding="utf-8"))
    assert manifest["count"] == len(manifest["case_ids"]) > 0
    assert (out_dir / "README.md").is_file()


def test_split_reports_a_bad_config(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad-splits.yaml"
    bad.write_text("train: 0.5\ndev: 0.1\ntest: 0.3\n")
    result = runner.invoke(
        app,
        [
            "split",
            "--dataset",
            str(repo_root / "data/generated/smoke.jsonl"),
            "--config",
            str(bad),
        ],
    )
    assert result.exit_code == 1
    assert "bad-splits.yaml" in result.stdout


def test_split_reports_a_missing_dataset(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(
        app,
        [
            "split",
            "--dataset",
            str(tmp_path / "missing.jsonl"),
            "--config",
            str(repo_root / "configs/splits.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "missing.jsonl" in result.stdout


def test_split_reports_an_unwritable_out_dir(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    blocked = tmp_path / "already-a-file"
    blocked.write_text("")
    result = runner.invoke(
        app,
        [
            "split",
            "--dataset",
            str(_split_dataset(tmp_path, repo_root)),
            "--config",
            str(repo_root / "configs/splits.yaml"),
            "--out-dir",
            str(blocked),
        ],
    )
    assert result.exit_code == 1
    assert "already-a-file" in result.stdout
    assert "cannot write" in result.stdout


def test_split_redacts_with_the_policies_it_is_given(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    """A dataset generated from custom policies must not have its protected
    value printed just because the standard policies file does not list it."""
    policies = tmp_path / "custom-policies.yaml"
    policies.write_text(
        "policies:\n"
        "  - policy_id: protect-thing\n"
        "    description: d\n"
        "    categories: [c]\n"
        "    protected_values: [SUPER-SECRET-VALUE]\n"
        "    protected_label: thing\n"
    )
    dataset = tmp_path / "broken.jsonl"
    dataset.write_text('{"case_id": "x", "prompt": "SUPER-SECRET-VALUE"}\n')
    result = runner.invoke(
        app,
        [
            "split",
            "--dataset",
            str(dataset),
            "--config",
            str(repo_root / "configs/splits.yaml"),
            "--out-dir",
            str(tmp_path / "splits"),
            "--policies",
            str(policies),
        ],
    )
    assert result.exit_code == 1
    assert "SUPER-SECRET-VALUE" not in result.stdout
    assert "[REDACTED]" in result.stdout


def test_sample_gold_writes_the_subset(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    from tipguard.benchmark.io import load_cases

    out = tmp_path / "sample.jsonl"
    result = runner.invoke(
        app,
        [
            "sample-gold",
            "--dataset",
            str(repo_root / "data/generated/tipguard-v1.jsonl"),
            "--out",
            str(out),
            "--policies",
            str(repo_root / "configs/policies.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    written = load_cases(out)
    assert len(written) >= 170
    assert f"wrote {len(written)} of 10% sample" in result.stdout
    assert "tip=" in result.stdout


def test_sample_gold_reports_a_missing_dataset(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(
        app,
        [
            "sample-gold",
            "--dataset",
            str(tmp_path / "absent.jsonl"),
            "--out",
            str(tmp_path / "sample.jsonl"),
            "--policies",
            str(repo_root / "configs/policies.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "file not found" in result.stdout


def test_dataset_stats_prints_tables_that_total_the_case_count(repo_root) -> None:  # type: ignore[no-untyped-def]
    from tipguard.benchmark.io import load_cases
    from tipguard.benchmark.stats import dataset_sha256

    dataset = repo_root / "data/generated/tipguard-v1.jsonl"
    cases = load_cases(dataset)
    result = runner.invoke(
        app,
        [
            "dataset-stats",
            "--dataset",
            str(dataset),
            "--policies",
            str(repo_root / "configs/policies.yaml"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "### Case type by transformation" in result.stdout
    assert "| split | cases |" in result.stdout
    assert f"total: {len(cases)}" in result.stdout
    assert f"sha256: {dataset_sha256(dataset)}" in result.stdout
    totals = [line for line in result.stdout.splitlines() if line.startswith("| total |")]
    assert all(line.rstrip().endswith(f"{len(cases)} |") for line in totals)
    assert len(totals) == 2


def test_dataset_stats_reports_a_missing_dataset(tmp_path, repo_root) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(
        app,
        [
            "dataset-stats",
            "--dataset",
            str(tmp_path / "absent.jsonl"),
            "--policies",
            str(repo_root / "configs/policies.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "file not found" in result.stdout

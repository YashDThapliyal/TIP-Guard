"""Command-line entry point for TIP-Guard."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from tipguard import __version__
from tipguard.benchmark.config import BenchmarkConfig
from tipguard.benchmark.dedupe import dedupe_cases
from tipguard.benchmark.generator import generate_cases
from tipguard.benchmark.io import DatasetError, load_cases, write_cases
from tipguard.benchmark.schema import BenchmarkCase
from tipguard.benchmark.templates import TemplateBank
from tipguard.benchmark.validate import ValidationIssue, validate_cases
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import ExperimentConfig, PoliciesConfig
from tipguard.evaluation.runner import run_experiment
from tipguard.logging import redact
from tipguard.models.types import ProviderError

app = typer.Typer(help="TIP-Guard command-line interface.", no_args_is_help=True)


def _protected_values(policies_path: Path) -> tuple[str, ...]:
    """Protected values to redact from echoed messages, best effort.

    Error text can quote a prompt or an upstream request body, both of which
    carry the protected values. If the policies file cannot be loaded there is
    nothing to redact against, so the message is echoed as-is rather than
    swallowed.
    """
    try:
        policies = load_yaml_model(policies_path, PoliciesConfig)
    except ConfigError:
        return ()
    return tuple(value for policy in policies.policies for value in policy.protected_values)


def _experiment_protected_values(config_path: Path) -> tuple[str, ...]:
    try:
        experiment = load_yaml_model(config_path, ExperimentConfig)
    except ConfigError:
        return ()
    return _protected_values(experiment.policies_config)


def _benchmark_protected_values(config_path: Path) -> tuple[str, ...]:
    try:
        benchmark = load_yaml_model(config_path, BenchmarkConfig)
    except ConfigError:
        return ()
    return _protected_values(benchmark.policies_config)


def _echo(message: str, protected: Sequence[str]) -> None:
    typer.echo(redact(message, protected))


def _report_issues(issues: Sequence[ValidationIssue], protected: Sequence[str]) -> None:
    for issue in issues:
        _echo(f"{issue.case_id or '-'}: {issue.message}", protected)


@app.callback()
def main() -> None:
    """TIP-Guard command-line interface."""


@app.command()
def version() -> None:
    """Print the installed tipguard version."""
    typer.echo(__version__)


@app.command("validate-dataset")
def validate_dataset(
    path: Path,
    policies: Annotated[Path, typer.Option(help="Policies YAML.")] = Path("configs/policies.yaml"),
) -> None:
    """Validate a JSONL benchmark file against the schema and policies."""
    protected = _protected_values(policies)
    try:
        cases = load_cases(path)
        issues = validate_cases(cases, load_yaml_model(policies, PoliciesConfig))
    except (DatasetError, ConfigError) as exc:
        _echo(str(exc), protected)
        raise typer.Exit(code=1) from exc
    _report_issues(issues, protected)
    if issues:
        raise typer.Exit(code=1)
    typer.echo(f"OK: {len(cases)} cases")


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option("--config", help="Experiment YAML.")],
    run_id: Annotated[str | None, typer.Option(help="Override the generated run id.")] = None,
    limit: Annotated[
        int | None, typer.Option(help="Evaluate only the first N cases.", min=1)
    ] = None,
) -> None:
    """Run an experiment configuration and write results under the output directory."""
    try:
        experiment = load_yaml_model(config, ExperimentConfig)
        if limit is not None:
            experiment = experiment.model_copy(update={"limit": limit})
        artifacts = run_experiment(experiment, config, run_id=run_id)
    except (ConfigError, DatasetError, ProviderError) as exc:
        _echo(str(exc), _experiment_protected_values(config))
        raise typer.Exit(code=1) from exc
    typer.echo(f"run: {artifacts.run_dir}")
    typer.echo("type count blocked leaked correct_decision")
    for name, stats in sorted(artifacts.summary.by_type.items()):
        typer.echo(f"{name} {stats.count} {stats.blocked} {stats.leaked} {stats.correct_decision}")


def _type_counts(cases: Sequence[BenchmarkCase]) -> str:
    counts = Counter(case.case_type.value for case in cases)
    return " ".join(f"{name}={counts[name]}" for name in sorted(counts))


def _build(
    config: Path, limit: int | None
) -> tuple[BenchmarkConfig, tuple[BenchmarkCase, ...], int]:
    """Generate and dedupe, returning the config, the cases, and how many went."""
    benchmark = load_yaml_model(config, BenchmarkConfig)
    policies = load_yaml_model(benchmark.policies_config, PoliciesConfig)
    bank = TemplateBank.load(benchmark.templates_dir, policies)
    generated = generate_cases(benchmark, bank, policies)
    kept = dedupe_cases(generated)
    removed = len(generated) - len(kept)
    return benchmark, kept[:limit] if limit is not None else kept, removed


@app.command()
def generate(
    config: Annotated[Path, typer.Option("--config", help="Benchmark YAML.")] = Path(
        "configs/benchmark.yaml"
    ),
    out: Annotated[Path | None, typer.Option("--out", help="Override the output path.")] = None,
    limit: Annotated[int | None, typer.Option(help="Write only the first N cases.", min=1)] = None,
) -> None:
    """Generate the benchmark dataset, deduplicate it, validate it, and write JSONL."""
    protected = _benchmark_protected_values(config)
    try:
        benchmark, cases, removed = _build(config, limit)
        issues = validate_cases(cases, load_yaml_model(benchmark.policies_config, PoliciesConfig))
    except (ConfigError, DatasetError, ValueError) as exc:
        _echo(str(exc), protected)
        raise typer.Exit(code=1) from exc
    if issues:
        _report_issues(issues, protected)
        raise typer.Exit(code=1)
    destination = out or benchmark.output
    write_cases(destination, cases)
    typer.echo(f"wrote {len(cases)} cases to {destination} ({removed} removed as duplicates)")
    typer.echo(_type_counts(cases))


if __name__ == "__main__":
    app()

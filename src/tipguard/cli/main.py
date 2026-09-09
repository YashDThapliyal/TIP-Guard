"""Command-line entry point for TIP-Guard."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from tipguard import __version__
from tipguard.benchmark.config import BenchmarkConfig
from tipguard.benchmark.dedupe import dedupe_cases
from tipguard.benchmark.generator import generate_cases
from tipguard.benchmark.gold import DEFAULT_GOLD_FRACTION, sample_gold
from tipguard.benchmark.io import DatasetError, load_cases, write_cases
from tipguard.benchmark.schema import BenchmarkCase
from tipguard.benchmark.splits import SplitConfig, assign_splits, write_manifests
from tipguard.benchmark.stats import dataset_sha256, format_stats
from tipguard.benchmark.templates import TemplateBank
from tipguard.benchmark.validate import ValidationIssue, validate_cases
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import ExperimentConfig, PoliciesConfig
from tipguard.evaluation.runner import run_experiment
from tipguard.logging import configure_logging, redact
from tipguard.models.types import ProviderError

app = typer.Typer(help="TIP-Guard command-line interface.", no_args_is_help=True)

DEFAULT_POLICIES = Path("configs/policies.yaml")
DEFAULT_DATASET = Path("data/generated/tipguard-v1.jsonl")
DEFAULT_SPLITS_CONFIG = Path("configs/splits.yaml")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_GOLD_SAMPLE = Path("data/labels/gold-sample.jsonl")


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
    policies: Annotated[Path, typer.Option(help="Policies YAML.")] = DEFAULT_POLICIES,
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
    # Load once, and only then configure logging: reaching for the protected
    # values before the config is known good would clear the process's
    # logging handlers on the way to reporting that the config is broken.
    # The CLI owns that call; `run_experiment` deliberately does not make it,
    # so a library caller keeps its own configuration.
    try:
        experiment = load_yaml_model(config, ExperimentConfig)
    except ConfigError as exc:
        _echo(str(exc), ())
        raise typer.Exit(code=1) from exc
    protected = _protected_values(experiment.policies_config)
    configure_logging(protected_values=protected)
    try:
        if limit is not None:
            experiment = experiment.model_copy(update={"limit": limit})
        artifacts = run_experiment(experiment, config, run_id=run_id, protected_values=protected)
    except (ConfigError, DatasetError, ProviderError) as exc:
        _echo(str(exc), protected)
        raise typer.Exit(code=1) from exc
    if artifacts.resumed:
        typer.echo(f"resumed incomplete run: {artifacts.run_id}")
    typer.echo(f"run: {artifacts.run_dir}")
    typer.echo("type count blocked leaked correct_decision")
    for name, stats in sorted(artifacts.summary.by_type.items()):
        typer.echo(f"{name} {stats.count} {stats.blocked} {stats.leaked} {stats.correct_decision}")


def _type_counts(cases: Sequence[BenchmarkCase]) -> str:
    counts = Counter(case.case_type.value for case in cases)
    return " ".join(f"{name}={counts[name]}" for name in sorted(counts))


@dataclass(frozen=True)
class _Generated:
    """Everything `generate` needs after the fallible work has succeeded."""

    destination: Path
    cases: tuple[BenchmarkCase, ...]
    removed: int
    issues: tuple[ValidationIssue, ...]


def _write_dataset(path: Path, cases: Sequence[BenchmarkCase]) -> None:
    """Write the JSONL, turning an I/O failure into the standard dataset error.

    An `--out` naming a directory, a missing parent that cannot be created or
    a read-only location must reach the user as the same one-line, redacted
    message every other failure does, not as a traceback.
    """
    try:
        write_cases(path, cases)
    except OSError as exc:
        raise DatasetError(f"{path}: cannot write file: {exc}") from exc


def _build(config: Path, out: Path | None, limit: int | None) -> _Generated:
    """Generate, dedupe, truncate to `limit`, and validate what will be written.

    `removed` counts what deduplication dropped, before any truncation.
    """
    benchmark = load_yaml_model(config, BenchmarkConfig)
    policies = load_yaml_model(benchmark.policies_config, PoliciesConfig)
    bank = TemplateBank.load(benchmark.templates_dir, policies)
    generated = generate_cases(benchmark, bank, policies)
    kept = dedupe_cases(generated)
    cases = kept[:limit] if limit is not None else kept
    return _Generated(
        destination=out or benchmark.output,
        cases=cases,
        removed=len(generated) - len(kept),
        issues=validate_cases(cases, policies),
    )


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
        built = _build(config, out, limit)
        if not built.issues:
            _write_dataset(built.destination, built.cases)
    except (ConfigError, DatasetError, ValueError) as exc:
        _echo(str(exc), protected)
        raise typer.Exit(code=1) from exc
    if built.issues:
        _report_issues(built.issues, protected)
        raise typer.Exit(code=1)
    typer.echo(
        f"wrote {len(built.cases)} cases to {built.destination} "
        f"({built.removed} removed as duplicates)"
    )
    typer.echo(_type_counts(built.cases))


def _write_manifests(cases: Sequence[BenchmarkCase], out_dir: Path) -> dict[str, int]:
    """Write the manifests, turning an I/O failure into a dataset error.

    An `--out-dir` naming an existing file or a read-only location must reach
    the user as the same one-line, redacted message every other failure does.
    """
    try:
        return write_manifests(cases, out_dir)
    except OSError as exc:
        raise DatasetError(f"{out_dir}: cannot write manifests: {exc}") from exc


@app.command()
def split(
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Benchmark JSONL to rewrite in place.")
    ] = DEFAULT_DATASET,
    config: Annotated[Path, typer.Option("--config", help="Split YAML.")] = DEFAULT_SPLITS_CONFIG,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where the manifests are written.")
    ] = DEFAULT_SPLITS_DIR,
    policies: Annotated[
        Path, typer.Option(help="Policies YAML, used only to redact error messages.")
    ] = DEFAULT_POLICIES,
) -> None:
    """Assign split conditions, rewrite the dataset in place, and write manifests."""
    protected = _protected_values(policies)
    try:
        split_config = load_yaml_model(config, SplitConfig)
        cases = assign_splits(load_cases(dataset), split_config)
        _write_dataset(dataset, cases)
        counts = _write_manifests(cases, out_dir)
    except (ConfigError, DatasetError) as exc:
        _echo(str(exc), protected)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {len(cases)} cases to {dataset} and manifests to {out_dir}")
    for name, count in counts.items():
        typer.echo(f"{name} {count}")


@app.command("sample-gold")
def sample_gold_command(
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Benchmark JSONL to sample from.")
    ] = DEFAULT_DATASET,
    out: Annotated[
        Path, typer.Option("--out", help="Where the sampled cases are written.")
    ] = DEFAULT_GOLD_SAMPLE,
    fraction: Annotated[
        float, typer.Option("--fraction", help="Share of the dataset to sample.", min=0, max=1)
    ] = DEFAULT_GOLD_FRACTION,
    policies: Annotated[
        Path, typer.Option(help="Policies YAML, used only to redact error messages.")
    ] = DEFAULT_POLICIES,
) -> None:
    """Write the stratified gold subset so a reviewer can read the prompts."""
    protected = _protected_values(policies)
    try:
        sampled = sample_gold(load_cases(dataset), fraction=fraction)
        _write_dataset(out, sampled)
    except (DatasetError, ValueError) as exc:
        _echo(str(exc), protected)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {len(sampled)} of {fraction:.0%} sample to {out}")
    typer.echo(_type_counts(sampled))


@app.command("dataset-stats")
def dataset_stats(
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Benchmark JSONL to describe.")
    ] = DEFAULT_DATASET,
    policies: Annotated[
        Path, typer.Option(help="Policies YAML, used only to redact error messages.")
    ] = DEFAULT_POLICIES,
) -> None:
    """Print the dataset's composition as Markdown, with its sha256."""
    try:
        cases = load_cases(dataset)
        report = format_stats(cases, dataset_sha256(dataset))
    except DatasetError as exc:
        _echo(str(exc), _protected_values(policies))
        raise typer.Exit(code=1) from exc
    typer.echo(report)


if __name__ == "__main__":
    app()

"""Command-line entry point for TIP-Guard."""

from pathlib import Path
from typing import Annotated

import typer

from tipguard import __version__
from tipguard.benchmark.io import DatasetError, load_cases
from tipguard.benchmark.validate import validate_cases
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import ExperimentConfig, PoliciesConfig
from tipguard.evaluation.runner import run_experiment
from tipguard.models.types import ProviderError

app = typer.Typer(help="TIP-Guard command-line interface.", no_args_is_help=True)


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
    try:
        cases = load_cases(path)
        issues = validate_cases(cases, load_yaml_model(policies, PoliciesConfig))
    except (DatasetError, ConfigError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    for issue in issues:
        typer.echo(f"{issue.case_id or '-'}: {issue.message}")
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
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f"run: {artifacts.run_dir}")
    typer.echo("type count blocked leaked correct")
    for name, stats in sorted(artifacts.summary.by_type.items()):
        typer.echo(f"{name} {stats.count} {stats.blocked} {stats.leaked} {stats.correct_decision}")


if __name__ == "__main__":
    app()

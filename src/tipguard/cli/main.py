"""Command-line entry point for TIP-Guard."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from tipguard import __version__
from tipguard.benchmark.io import DatasetError, load_cases
from tipguard.benchmark.validate import validate_cases
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


def _echo(message: str, protected: Sequence[str]) -> None:
    typer.echo(redact(message, protected))


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
    for issue in issues:
        _echo(f"{issue.case_id or '-'}: {issue.message}", protected)
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


if __name__ == "__main__":
    app()

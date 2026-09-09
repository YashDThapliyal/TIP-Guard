"""Run one experiment configuration end to end and persist its artifacts."""

import contextlib
import hashlib
import json
import os
import platform
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tipguard import __version__
from tipguard.benchmark.io import DatasetError, load_cases
from tipguard.benchmark.schema import BenchmarkCase
from tipguard.benchmark.validate import validate_cases
from tipguard.config.loader import ConfigError, load_yaml_model
from tipguard.config.schemas import ExperimentConfig, ModelsConfig, PoliciesConfig
from tipguard.evaluation.case import _answer_correct, evaluate_case
from tipguard.evaluation.summary import CaseRecord, RunSummary, summarize
from tipguard.guardrail.factory import build_guardrail
from tipguard.logging import get_logger, redacting
from tipguard.models.cache import ResponseCache
from tipguard.models.registry import ProviderRegistry
from tipguard.run_id import config_hash, make_run_id
from tipguard.seeding import seed_everything

__all__ = [
    "OUTPUT_DIR_ENV",
    "RUN_COMPLETE_MARKER",
    "RunArtifacts",
    "_answer_correct",
    "evaluate_case",
    "run_experiment",
]

OUTPUT_DIR_ENV = "TIPGUARD_OUTPUT_DIR"
#: Dropped into a run directory once every artifact is on disk. Its
#: presence is what separates a finished run, which must never be
#: overwritten, from the wreckage of a crashed one, which is resumable.
RUN_COMPLETE_MARKER = ".complete"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_DATASET_ISSUES = 5
log = get_logger("runner")


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    summary: RunSummary
    records: tuple[CaseRecord, ...]
    #: True when the run reused the directory an earlier, crashed run had
    #: reserved but never completed.
    resumed: bool = False


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_model_alias(config: ExperimentConfig, models: ModelsConfig) -> None:
    """Fail fast on an unknown main-model alias, before any run directory or
    response cache is created. Delegates to `ProviderRegistry.spec`, which owns
    the error message, so the alias check lives in exactly one place.
    """
    ProviderRegistry(models).spec(config.main_model)


def _ensure_dataset_valid(
    cases: Sequence[BenchmarkCase], policies: PoliciesConfig, dataset: Path
) -> None:
    issues = validate_cases(cases, policies)
    if issues:
        detail = "; ".join(
            f"{issue.case_id or '-'}: {issue.message}" for issue in issues[:_MAX_DATASET_ISSUES]
        )
        raise DatasetError(f"{dataset}: {detail}")


def _ensure_run_id_safe(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ConfigError(f"invalid run id {run_id!r}")


def _reserve_run_dir(run_dir: Path) -> bool:
    """Claim run_dir, returning True when a crashed run is being resumed.

    mkdir(exist_ok=False) is the atomic reservation, so two concurrent runs
    with the same id can't both pass a check-then-write race; the loser sees
    FileExistsError. An existing directory is a *finished* run only when it
    carries `RUN_COMPLETE_MARKER`, which `_write_artifacts` drops after the
    last artifact is written; that one is refused, because overwriting a
    result the analysis may already have consumed is never what was meant.
    Without the marker the directory is a partial write from a run that
    died, so it is resumable and this run overwrites it.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        if not run_dir.is_dir():
            raise ConfigError(f"run directory path is not a directory: {run_dir}") from exc
        if (run_dir / RUN_COMPLETE_MARKER).exists():
            raise ConfigError(
                f"run directory already exists: {run_dir}; choose another --run-id"
            ) from exc
        return True
    return False


def _release_run_dir_if_empty(run_dir: Path) -> None:
    """Undo a reservation after a failed run so a retry with the same id isn't
    blocked by an empty directory nothing ever wrote artifacts into."""
    with contextlib.suppress(OSError):
        if run_dir.is_dir() and not any(run_dir.iterdir()):
            run_dir.rmdir()


def _log_records(records: Sequence[CaseRecord]) -> None:
    for record in records:
        log.info(
            "case_evaluated",
            extra={
                "case_id": record.case_id,
                "decision": record.decision.value,
                "leaked": record.leaked,
            },
        )


def _write_artifacts(
    run_dir: Path, records: Sequence[CaseRecord], summary: RunSummary, manifest: dict[str, object]
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Last, and only once all three artifacts exist: a marker written any
    # earlier would declare a half-written run finished.
    (run_dir / RUN_COMPLETE_MARKER).write_text("", encoding="utf-8")


def _build_manifest(
    config: ExperimentConfig,
    config_path: Path,
    resolved_run_id: str,
    now: datetime,
    registry: ProviderRegistry,
    output_dir: Path,
) -> dict[str, object]:
    spec = registry.spec(config.main_model)
    return {
        "run_id": resolved_run_id,
        "created_at": now.isoformat(),
        "config_path": str(config_path),
        "config": config.model_dump(mode="json"),
        "config_hash": config_hash(config),
        "dataset_sha256": _sha256_file(config.dataset),
        "tipguard_version": __version__,
        "python_version": platform.python_version(),
        "defense": config.defense.name,
        "output_dir": str(output_dir),
        "main_model": {"alias": config.main_model, "provider": spec.provider, "model": spec.model},
    }


def _policy_values(policies: PoliciesConfig) -> tuple[str, ...]:
    return tuple(value for policy in policies.policies for value in policy.protected_values)


def run_experiment(
    config: ExperimentConfig,
    config_path: Path,
    run_id: str | None = None,
    now: datetime | None = None,
    protected_values: Sequence[str] | None = None,
) -> RunArtifacts:
    """Run `config` end to end and return its artifacts.

    This does **not** configure logging. Handlers, level and propagation are
    process-wide state, and a library facade such as `ProtectedModel` calls
    this on behalf of an application that has already made those choices;
    the CLI owns `configure_logging` instead. What the runner still owes its
    caller is that no TIP-Guard record carries a protected value, so it turns
    on redaction for every `tipguard.*` logger for the duration of the run —
    including `tipguard.models`, whose provider-failure debug line can quote
    an SDK message carrying the system prompt. Pass
    `protected_values` to redact against something other than the policies
    file `config` names — by default that file's values are used, so a
    caller that does nothing still gets redaction.
    """
    now = now or datetime.now(UTC)
    seed_everything(config.seed)
    policies = load_yaml_model(config.policies_config, PoliciesConfig)
    models = load_yaml_model(config.models_config, ModelsConfig)
    _ensure_model_alias(config, models)
    protected = _policy_values(policies) if protected_values is None else tuple(protected_values)
    with redacting(protected):
        return _run_evaluated(config, config_path, models, policies, run_id, now)


def _run_evaluated(
    config: ExperimentConfig,
    config_path: Path,
    models: ModelsConfig,
    policies: PoliciesConfig,
    run_id: str | None,
    now: datetime,
) -> RunArtifacts:
    """Reserve the run directory, evaluate, and persist — the body of a run."""
    cases = load_cases(config.dataset)
    _ensure_dataset_valid(cases, policies, config.dataset)
    cases = cases[: config.limit]
    resolved_run_id = run_id or make_run_id(config.name, config, now=now)
    _ensure_run_id_safe(resolved_run_id)
    output_dir = Path(os.environ.get(OUTPUT_DIR_ENV) or config.output_dir)
    run_dir = output_dir / resolved_run_id
    resumed = _reserve_run_dir(run_dir)
    if resumed:
        log.info("run_resumed", extra={"run_id": resolved_run_id, "run_dir": str(run_dir)})
    try:
        records, summary, manifest = _evaluate_and_build_manifest(
            config, models, policies, cases, config_path, resolved_run_id, now, output_dir
        )
        _write_artifacts(run_dir, records, summary, manifest)
    except Exception:
        _release_run_dir_if_empty(run_dir)
        raise
    return RunArtifacts(resolved_run_id, run_dir, summary, records, resumed)


def _evaluate_and_build_manifest(
    config: ExperimentConfig,
    models: ModelsConfig,
    policies: PoliciesConfig,
    cases: Sequence[BenchmarkCase],
    config_path: Path,
    resolved_run_id: str,
    now: datetime,
    output_dir: Path,
) -> tuple[tuple[CaseRecord, ...], RunSummary, dict[str, object]]:
    """Build the cache/registry/guardrail, evaluate every case, and return the
    records/summary/manifest. Closes the response cache (if any) whether
    evaluation succeeds or raises; the caller decides what to do on failure.
    """
    cache = ResponseCache(config.cache_dir / "responses.sqlite") if config.cache_dir else None
    try:
        registry = ProviderRegistry(models, cache=cache)
        guardrail = build_guardrail(config.defense, registry, config.main_model, policies)
        records = tuple(evaluate_case(case, guardrail, policies) for case in cases)
        _log_records(records)
        summary = summarize(records)
        manifest = _build_manifest(config, config_path, resolved_run_id, now, registry, output_dir)
        return records, summary, manifest
    finally:
        if cache is not None:
            cache.close()

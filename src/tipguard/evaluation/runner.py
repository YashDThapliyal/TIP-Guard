"""Run one experiment configuration end to end and persist its artifacts."""

import hashlib
import json
import os
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tipguard import __version__
from tipguard.benchmark.io import load_cases
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import ExperimentConfig, ModelsConfig, PoliciesConfig
from tipguard.evaluation.case import evaluate_case
from tipguard.evaluation.summary import CaseRecord, RunSummary, summarize
from tipguard.guardrail.factory import build_guardrail
from tipguard.logging import configure_logging, get_logger
from tipguard.models.cache import ResponseCache
from tipguard.models.registry import ProviderRegistry
from tipguard.run_id import config_hash, make_run_id
from tipguard.seeding import seed_everything

__all__ = ["OUTPUT_DIR_ENV", "RunArtifacts", "evaluate_case", "run_experiment"]

OUTPUT_DIR_ENV = "TIPGUARD_OUTPUT_DIR"
log = get_logger("runner")


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    summary: RunSummary
    records: tuple[CaseRecord, ...]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _build_manifest(
    config: ExperimentConfig,
    config_path: Path,
    resolved_run_id: str,
    now: datetime,
    registry: ProviderRegistry,
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
        "main_model": {"alias": config.main_model, "provider": spec.provider, "model": spec.model},
    }


def run_experiment(
    config: ExperimentConfig,
    config_path: Path,
    run_id: str | None = None,
    now: datetime | None = None,
) -> RunArtifacts:
    now = now or datetime.now(UTC)
    seed_everything(config.seed)
    policies = load_yaml_model(config.policies_config, PoliciesConfig)
    models = load_yaml_model(config.models_config, ModelsConfig)
    configure_logging(protected_values=[v for p in policies.policies for v in p.protected_values])
    cache = ResponseCache(config.cache_dir / "responses.sqlite") if config.cache_dir else None
    registry = ProviderRegistry(models, cache=cache)
    main_model = registry.get(config.main_model)
    guardrail = build_guardrail(config.defense, main_model, policies)
    cases = load_cases(config.dataset)[: config.limit]
    records = tuple(evaluate_case(case, guardrail, policies) for case in cases)
    for record in records:
        log.info(
            "case_evaluated",
            extra={
                "case_id": record.case_id,
                "decision": record.decision.value,
                "leaked": record.leaked,
            },
        )
    summary = summarize(records)
    resolved_run_id = run_id or make_run_id(config.name, config, now=now)
    output_dir = Path(os.environ.get(OUTPUT_DIR_ENV) or config.output_dir)
    run_dir = output_dir / resolved_run_id
    manifest = _build_manifest(config, config_path, resolved_run_id, now, registry)
    _write_artifacts(run_dir, records, summary, manifest)
    return RunArtifacts(resolved_run_id, run_dir, summary, records)

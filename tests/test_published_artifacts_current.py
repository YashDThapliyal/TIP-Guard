"""The published results must still be what today's code produces.

`artifacts/study-v1/` is the evidence every table in the report is computed
from, and the library has changed since those runs. If a change altered a
guardrail decision, the published record would silently describe behaviour the
code no longer has -- and because the tables are regenerated from the stored
records rather than by re-running, nothing else here would notice.

So this replays committed cases through the current pipeline and compares
decisions. It needs the response cache, since replaying without it would make
live API calls, so it skips when the cache is absent -- which is the case in a
fresh clone and in CI.
"""

import json
from pathlib import Path

import pytest

from tipguard.benchmark.io import load_cases
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import ExperimentConfig, ModelsConfig, PoliciesConfig
from tipguard.evaluation.case import evaluate_case
from tipguard.evaluation.summary import CaseRecord
from tipguard.guardrail.factory import build_guardrail
from tipguard.models.cache import ResponseCache
from tipguard.models.registry import ProviderRegistry

MARKERS = Path("artifacts/study-v1/markers")

#: Enough to catch a behavioural change without making the suite slow. Every
#: case must come from cache, so this costs nothing.
SAMPLE = 40

#: One classifier arm and one canonicalizing arm: between them they exercise
#: the threshold guard, the risk classifier, the canonicalizer and the policy
#: gate.
REPLAY_ARMS = ("input_classifier_v2-context", "tip_guard-context")


@pytest.mark.parametrize("arm", REPLAY_ARMS)
def test_committed_results_reproduce_under_current_code(arm: str) -> None:
    marker = MARKERS / f"{arm}.json"
    if not marker.exists():
        pytest.skip(f"no committed artifact for {arm}")
    config = load_yaml_model(Path(f"experiments/study/{arm}.yaml"), ExperimentConfig)
    if config.cache_dir is None or not (config.cache_dir / "responses.sqlite").exists():
        pytest.skip("no response cache; replaying would make live API calls")

    stored: dict[str, CaseRecord] = {}
    results = Path(json.loads(marker.read_text(encoding="utf-8"))["run_dir"]) / "results.jsonl"
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = CaseRecord.model_validate_json(line)
            stored[record.case_id] = record

    policies = load_yaml_model(config.policies_config, PoliciesConfig)
    models = load_yaml_model(config.models_config, ModelsConfig)
    cache = ResponseCache(config.cache_dir / "responses.sqlite")
    try:
        registry = ProviderRegistry(models, cache=cache)
        guard = build_guardrail(
            config.defense, registry, config.main_model, policies, config.system_prompt
        )
        wanted = set(config.splits)
        cases = [c for c in load_cases(config.dataset) if c.split in wanted][:SAMPLE]
        live = 0
        mismatches: list[tuple[str, str, str]] = []
        for case in cases:
            if case.case_id not in stored:
                continue
            fresh = evaluate_case(case, guard, policies)
            live += fresh.model_calls - fresh.cached_model_calls
            if fresh.decision is not stored[case.case_id].decision:
                mismatches.append(
                    (case.case_id, stored[case.case_id].decision.value, fresh.decision.value)
                )
    finally:
        cache.close()

    assert not mismatches, (
        f"{arm}: the published results no longer match what this code produces, so the "
        f"report describes behaviour the library has lost: {mismatches[:5]}"
    )
    # If this starts failing, the cache was pruned rather than the code changing.
    assert live == 0, f"{arm}: {live} replayed calls missed the cache; the comparison spent money"

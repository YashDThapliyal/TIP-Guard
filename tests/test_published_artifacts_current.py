"""The published results must still be what today's code produces.

`artifacts/study-v1/` is the evidence every table in the report is computed
from, and the library has changed since those runs. If a change altered a
guardrail decision, the published record would silently describe behaviour the
code no longer has -- and because the tables are regenerated from the stored
records rather than by re-running, nothing else here would notice.

So this replays committed cases through the current pipeline and compares the
outcomes. Two properties of how it does that are load-bearing:

**It cannot spend money.** `CachedProvider.complete` falls through to the real
provider on a cache miss, so a replay against an incomplete cache would make
live API calls -- inside the default test suite, on whatever machine ran it. An
earlier version asserted afterwards that nothing had been billed, which detects
the spend rather than preventing it. The provider is now patched to raise on a
miss, so the failure mode is a loud error instead of a charge.

**It covers everything.** All 20 arms and all 989 cases each, because a
fail-closed replay is pure cache reads and finishes in seconds. An earlier
version sampled 40 cases from two arms -- 0.4% of the published evidence --
while the test name claimed the published results reproduce.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from tipguard.benchmark.io import load_cases
from tipguard.config.loader import load_yaml_model
from tipguard.config.schemas import ExperimentConfig, ModelsConfig, PoliciesConfig
from tipguard.evaluation.case import evaluate_case
from tipguard.evaluation.summary import CaseRecord
from tipguard.guardrail.factory import build_guardrail
from tipguard.models.cache import CachedProvider, ResponseCache, cache_key
from tipguard.models.registry import ProviderRegistry
from tipguard.models.types import ModelRequest, ModelResponse

MARKERS = Path("artifacts/study-v1/markers")
STUDY_CONFIGS = Path("experiments/study")


class ReplayCacheMissError(AssertionError):
    """A replayed request was not in the cache.

    Raised instead of calling the provider, so an incomplete cache stops the
    test rather than billing whoever ran it.
    """


def _cache_only(self: CachedProvider, request: ModelRequest) -> ModelResponse:
    """`CachedProvider.complete` with the live-call fallback removed."""
    key = cache_key(self._inner.name, self._inner.model, request, namespace=self._namespace)
    hit = self._cache.get(key)
    if hit is None:
        raise ReplayCacheMissError(
            f"no cached response for a {self._inner.name}/{self._inner.model} request; "
            "replaying would have made a live API call"
        )
    return self._repriced(hit)


@pytest.fixture
def cache_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CachedProvider, "complete", _cache_only)


def _arms() -> list[str]:
    return sorted(marker.stem for marker in MARKERS.glob("*.json"))


def _stored(arm: str) -> dict[str, CaseRecord]:
    marker = MARKERS / f"{arm}.json"
    results = Path(json.loads(marker.read_text(encoding="utf-8"))["run_dir"]) / "results.jsonl"
    records = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = CaseRecord.model_validate_json(line)
            records[record.case_id] = record
    return records


def _replayed(arm: str) -> Iterator[tuple[str, CaseRecord, CaseRecord]]:
    """Every stored case of `arm`, beside the record today's code produces."""
    config = load_yaml_model(STUDY_CONFIGS / f"{arm}.yaml", ExperimentConfig)
    if config.cache_dir is None or not (config.cache_dir / "responses.sqlite").exists():
        pytest.skip("no response cache; replaying would make live API calls")
    policies = load_yaml_model(config.policies_config, PoliciesConfig)
    models = load_yaml_model(config.models_config, ModelsConfig)
    stored = _stored(arm)
    cache = ResponseCache(config.cache_dir / "responses.sqlite")
    try:
        registry = ProviderRegistry(models, cache=cache)
        guard = build_guardrail(
            config.defense, registry, config.main_model, policies, config.system_prompt
        )
        wanted = set(config.splits)
        for case in load_cases(config.dataset):
            if case.split not in wanted or case.case_id not in stored:
                continue
            yield case.case_id, stored[case.case_id], evaluate_case(case, guard, policies)
    finally:
        cache.close()


@pytest.mark.parametrize("arm", _arms())
def test_committed_results_reproduce_under_current_code(arm: str, cache_only: None) -> None:
    """Every case of every arm, on the outcomes the report is built from.

    `decision` drives the detection and false-positive rates, `leaked` drives
    the violation rate, and `answer_correct` drives benign accuracy. Comparing
    only the decision would miss a change to leak detection, which is the
    measurement the report leans on hardest.
    """
    compared = 0
    mismatches: list[str] = []
    for case_id, stored, fresh in _replayed(arm):
        compared += 1
        for field in ("decision", "leaked", "answer_correct"):
            was, now = getattr(stored, field), getattr(fresh, field)
            if was != now:
                mismatches.append(f"{case_id}.{field}: published {was!r}, now {now!r}")

    assert compared, f"{arm}: replayed nothing, so this proves nothing"
    assert not mismatches, (
        f"{arm}: the published results no longer match what this code produces, so the report "
        f"describes behaviour the library has lost ({len(mismatches)} of {compared} cases differ): "
        f"{mismatches[:5]}"
    )


def test_the_replay_covers_every_published_arm() -> None:
    """Guards the coverage claim itself.

    The parametrisation is generated from the marker directory, so a missing
    artifact would silently shrink the sweep rather than fail it.
    """
    if not MARKERS.is_dir():
        pytest.skip(f"no committed artifacts at {MARKERS}")
    arms = _arms()
    configs = sorted(path.stem for path in STUDY_CONFIGS.glob("*.yaml"))
    assert arms, "no arms to replay"
    missing = sorted(set(configs) - set(arms))
    assert not missing, f"these arms have configs but no published artifact: {missing}"


def test_a_cache_miss_fails_instead_of_calling_out(cache_only: None) -> None:
    """The safety property, asserted directly rather than trusted.

    If this ever stops raising, the replay above can bill whoever runs the
    suite.
    """
    provider = CachedProvider.__new__(CachedProvider)
    provider._inner = _Absent()  # type: ignore[assignment]
    provider._cache = _EmptyCache()  # type: ignore[assignment]
    provider._namespace = ""
    provider._spec = None
    with pytest.raises(ReplayCacheMissError):
        provider.complete(ModelRequest.simple("anything", system="s"))


class _Absent:
    name = "absent"
    model = "absent"

    def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("the replay called the provider; it must not")


class _EmptyCache:
    def get(self, key: str) -> ModelResponse | None:
        return None

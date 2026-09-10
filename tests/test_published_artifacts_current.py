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

import hashlib
import json
import re
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
DATASET = Path("data/generated/tipguard-v1.jsonl")
REPORT = Path("docs/report.md")

#: The splits a result may be reported on. `train` and `dev` exist for tuning,
#: so measuring on them would report performance on data a defence was fitted
#: to. Stated here rather than read from the configs, because the configs are
#: one of the things that can shrink -- deriving the expected population from
#: them would make the check circular.
REPORTABLE_SPLITS = frozenset(
    {
        "test",
        "heldout_transformation",
        "heldout_policy",
        "heldout_paraphrase",
        "heldout_compositional",
    }
)


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
    """Every stored case of `arm`, beside the record today's code produces.

    Cases absent from the stored results are skipped, so the caller must check
    that the number replayed equals the number stored -- otherwise a truncated
    results file, a narrowed `splits`, or a regenerated dataset would shrink
    the sweep while every comparison it still made passed.
    """
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

    # Not `compared > 0`: every stored record must be replayed, or the sweep
    # has quietly shrunk and the arms it no longer covers are unverified.
    expected = len(_stored(arm))
    assert compared == expected, (
        f"{arm}: replayed {compared} of {expected} published cases. The missing ones are "
        "unverified -- the results file, the config's splits, or the dataset has changed."
    )
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

    # Every arm evaluated the same reportable case set, so an arm holding
    # fewer records than its peers has a truncated results file -- which the
    # per-arm replay would otherwise treat as its full population.
    counts = {arm: len(_stored(arm)) for arm in arms}
    sizes = set(counts.values())
    assert len(sizes) == 1, f"arms disagree on how many cases they hold: {sorted(counts.items())}"

    # And anchored to an absolute number, not just to each other. Requiring
    # only that the arms agree lets the whole study shrink in step -- every
    # config narrowed, or the dataset regenerated smaller -- with each arm
    # still replaying everything it stored and every arm still matching its
    # peers. The population comes from the committed dataset, whose digest
    # each run's manifest records.
    expected = _reportable_case_count()
    assert sizes == {expected}, (
        f"every arm holds {sizes.pop()} cases but the dataset has {expected} reportable ones; "
        "the study population has changed"
    )


def _reportable_case_count() -> int:
    return sum(1 for case in load_cases(DATASET) if case.split.value in REPORTABLE_SPLITS)


def test_every_arm_measured_the_full_reportable_population() -> None:
    """The configs must ask for every reportable split, not a subset.

    Checked separately from the record counts because the two can drift apart:
    a config could name fewer splits while an old results file still held the
    full set.
    """
    configs = sorted(STUDY_CONFIGS.glob("*.yaml"))
    assert configs, "no study configs"
    for path in configs:
        config = load_yaml_model(path, ExperimentConfig)
        named = {split.value for split in config.splits}
        assert named == set(REPORTABLE_SPLITS), (
            f"{path.name} measures {sorted(named)}, not the reportable population "
            f"{sorted(REPORTABLE_SPLITS)}"
        )


def test_the_dataset_is_the_one_the_runs_measured() -> None:
    """Pins the population to the exact bytes the study ran on.

    The case count is derived from the committed dataset, which makes the
    dataset the anchor -- and the dataset is mutable. Shrinking it *and* the
    results together satisfied every count check, because the expected number
    shrank in step with them. Each manifest recorded the digest of the file
    its run read, so comparing that to the file on disk is what closes it.
    """
    if not MARKERS.is_dir():
        pytest.skip(f"no committed artifacts at {MARKERS}")
    current = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    disagree: list[str] = []
    for marker in sorted(MARKERS.glob("*.json")):
        run_dir = Path(json.loads(marker.read_text(encoding="utf-8"))["run_dir"])
        recorded = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get(
            "dataset_sha256"
        )
        if recorded != current:
            disagree.append(f"{marker.stem}: ran on {str(recorded)[:12]}...")
    assert not disagree, (
        f"the committed dataset now hashes to {current[:12]}..., which is not what these runs "
        f"measured, so the published results describe a different population: {disagree[:5]}"
    )


#: Sample sizes the report quotes that no artifact can confirm: the two
#: pilots, which ran before the study and were never written to a run
#: directory. Listed explicitly so they are a stated exception rather than a
#: gap in the pattern.
PILOT_SAMPLE_SIZES = frozenset({20, 40})

#: Every "N cases" or "N-case" claim. The lookbehind matters: without it,
#: "leaks on 0.75 of prohibited cases" yields 75.
#: Must start with a digit: `[\d,]+` alone matched a bare comma in the report.
_COUNT_CLAIM = re.compile(r"(?<![\d.])(\d[\d,]*)(?:-case|\s+(?:[a-z_-]+\s+){0,2}cases?\b)")

#: Slash-form claims like "the full 752/237-case arm", which pair a
#: prohibited-case count with a benign one. `_COUNT_CLAIM` sees only the
#: number touching `-case`, so the numerator was never checked -- altering
#: 752 there changed nothing that any test read.
_SLASH_COUNT_CLAIM = re.compile(r"(?<![\d.])(\d[\d,]*)/(\d[\d,]*)-case")


def _quoted_counts(text: str) -> set[int]:
    """Every case count the prose claims, in both forms."""
    found = {int(match.replace(",", "")) for match in _COUNT_CLAIM.findall(text)}
    for numerator, denominator in _SLASH_COUNT_CLAIM.findall(text):
        found.add(int(numerator.replace(",", "")))
        found.add(int(denominator.replace(",", "")))
    return found


def _legitimate_case_counts() -> dict[int, str]:
    """Counts the report may quote, each derived from a file.

    An earlier version listed only three, and its pattern matched only
    `cases` after an optional `held-out`/`generated`. So `752 prohibited
    cases`, `26 direct cases` and the `752/237-case arm` -- all checkable
    against the dataset -- were never examined, and could have gone stale
    silently.
    """
    cases = load_cases(DATASET)
    reportable = [case for case in cases if case.split.value in REPORTABLE_SPLITS]

    def of_type(*types: str) -> int:
        return sum(1 for case in reportable if case.case_type.value in types)

    counts = {
        len(cases): "whole dataset",
        len(reportable): "reportable population",
        of_type("tip", "direct"): "prohibited cases",
        of_type("benign_transformation", "hard_negative"): "benign and hard-negative cases",
        of_type("benign_transformation"): "benign transformations",
        of_type("direct"): "direct cases",
    }
    review = Path("data/labels/gold-review.jsonl")
    if review.exists():
        reviewed = len([ln for ln in review.read_text(encoding="utf-8").splitlines() if ln.strip()])
        counts[reviewed] = "gold review"
    return counts


def test_every_case_count_in_the_report_is_current() -> None:
    """Every count claim in the prose, in every form it is written.

    An earlier version asserted the expected count appeared *somewhere*: the
    report says 989 in six sentences, so updating one and leaving five stale
    would have passed. The pattern was then still too narrow, examining 8 of
    the 14 count claims actually present.
    """
    text = REPORT.read_text(encoding="utf-8")
    legitimate = _legitimate_case_counts()
    quoted = _quoted_counts(text)
    assert quoted, "found no case-count claims in the report; the pattern is probably wrong"
    stale = sorted(n for n in quoted if n not in legitimate and n not in PILOT_SAMPLE_SIZES)
    assert not stale, (
        f"the report quotes case counts that match nothing on disk: {stale}. Legitimate counts "
        f"are {', '.join(f'{n} ({what})' for n, what in sorted(legitimate.items()))}, plus the "
        f"pilot sizes {sorted(PILOT_SAMPLE_SIZES)}"
    )


def test_the_count_pattern_sees_the_claims_that_are_there() -> None:
    """Guards the pattern, not the counts.

    A regex that quietly stops matching would make the test above pass by
    finding nothing to check -- which is how the previous version passed while
    ignoring six claims. These are forms the report actually uses.
    """
    for phrase, expected in (
        ("989 held-out cases", 989),
        ("989 cases", 989),
        ("752 prohibited cases", 752),
        ("26 direct cases", 26),
        ("1,700 generated cases", 1700),
        ("the 752/237-case arm", 237),
        # Both halves of a slash form, since only the second touches `-case`.
        ("the 752/237-case arm", 752),
        ("170-case gold subset", 170),
        ("40-case-per-type pilot", 40),
    ):
        found = _quoted_counts(phrase)
        assert expected in found, f"the pattern missed {expected} in {phrase!r} (found {found})"

    # And must not read a rate as a count.
    assert not _quoted_counts("leaks on 0.75 of prohibited cases"), (
        "the pattern read a decimal as a case count"
    )


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

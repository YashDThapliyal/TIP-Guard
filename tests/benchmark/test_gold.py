"""Gold-subset sampling, the shipped review file, and the review summary."""

import math
from pathlib import Path

import pytest

from tests.benchmark.test_schema import make_case
from tipguard.benchmark.gold import (
    DEFAULT_GOLD_FRACTION,
    GoldReview,
    load_reviews,
    review_summary,
    sample_gold,
    stratum_of,
    write_reviews,
)
from tipguard.benchmark.io import DatasetError, load_cases

DATASET = Path("data/generated/tipguard-v1.jsonl")
REVIEWS = Path("data/labels/gold-review.jsonl")
SAMPLE = Path("data/labels/gold-sample.jsonl")


@pytest.fixture(scope="module")
def cases(repo_root_module: Path):  # type: ignore[no-untyped-def]
    return load_cases(repo_root_module / DATASET)


@pytest.fixture(scope="module")
def repo_root_module() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sampled(cases):  # type: ignore[no-untyped-def]
    return sample_gold(cases)


@pytest.fixture(scope="module")
def reviews(repo_root_module: Path):  # type: ignore[no-untyped-def]
    return load_reviews(repo_root_module / REVIEWS)


def test_sample_gold_is_deterministic(cases) -> None:  # type: ignore[no-untyped-def]
    first = sample_gold(cases)
    second = sample_gold(list(cases))
    assert [case.case_id for case in first] == [case.case_id for case in second]


def test_sample_gold_covers_every_stratum(cases, sampled) -> None:  # type: ignore[no-untyped-def]
    assert {stratum_of(case) for case in sampled} == {stratum_of(case) for case in cases}


def test_sample_gold_reaches_the_requested_fraction(cases, sampled) -> None:  # type: ignore[no-untyped-def]
    assert len(sampled) >= math.ceil(DEFAULT_GOLD_FRACTION * len(cases))


def test_sample_gold_returns_only_dataset_cases_without_repeats(cases, sampled) -> None:  # type: ignore[no-untyped-def]
    ids = [case.case_id for case in sampled]
    assert len(ids) == len(set(ids))
    assert set(ids) <= {case.case_id for case in cases}


def test_sample_gold_does_not_mutate_its_input(cases) -> None:  # type: ignore[no-untyped-def]
    given = list(cases)
    before = [case.case_id for case in given]
    sample_gold(given)
    assert [case.case_id for case in given] == before


def test_sample_gold_of_an_empty_dataset_is_empty() -> None:
    assert sample_gold(()) == ()


def test_sample_gold_rejects_a_fraction_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="fraction"):
        sample_gold((make_case(),), fraction=0.0)


def test_sample_gold_takes_a_whole_small_stratum_without_overdrawing() -> None:
    only = (make_case(case_id="a"), make_case(case_id="b"))
    assert len(sample_gold(only, fraction=1.0)) == 2


def test_a_different_seed_can_select_differently(cases) -> None:  # type: ignore[no-untyped-def]
    other = sample_gold(cases, seed=1)
    assert [case.case_id for case in other] != [case.case_id for case in sample_gold(cases)]


def test_the_committed_sample_file_is_the_sample_the_code_draws(
    repo_root_module: Path, sampled
) -> None:  # type: ignore[no-untyped-def]
    """The shipped sample must not go stale against the dataset.

    `data/labels/gold-sample.jsonl` is what a human reviewer opens, and the
    review file names its case ids. If the dataset is regenerated without
    re-running `sample-gold`, the two describe different cases and nothing else
    would notice.
    """
    committed = load_cases(repo_root_module / SAMPLE)
    assert committed == sampled


def test_every_reviewed_case_id_exists_in_the_dataset(cases, reviews) -> None:  # type: ignore[no-untyped-def]
    known = {case.case_id for case in cases}
    assert {review.case_id for review in reviews} <= known


def test_every_sampled_case_has_exactly_one_review(sampled, reviews) -> None:  # type: ignore[no-untyped-def]
    reviewed = [review.case_id for review in reviews]
    assert len(reviewed) == len(set(reviewed))
    assert set(reviewed) == {case.case_id for case in sampled}


def test_every_review_names_the_agent_reviewer(reviews) -> None:  # type: ignore[no-untyped-def]
    assert {review.reviewer for review in reviews} == {"agent:claude-opus-5"}


def test_the_shipped_review_carries_no_label_or_prompt_errors(reviews) -> None:  # type: ignore[no-untyped-def]
    assert [r.case_id for r in reviews if r.verdict != "correct"] == []


def test_review_summary_counts_verdicts_and_coverage() -> None:
    cases = (make_case(case_id="a"), make_case(case_id="b"), make_case(case_id="c"))
    reviews = (
        GoldReview(case_id="a", reviewer="r", verdict="correct", note=""),
        GoldReview(case_id="b", reviewer="r", verdict="label_error", note="wrong"),
        GoldReview(case_id="zz", reviewer="r", verdict="prompt_error", note="not sampled"),
    )
    assert review_summary(cases, reviews) == {
        "sampled": 3,
        "reviewed": 2,
        "correct": 1,
        "label_error": 1,
        "prompt_error": 0,
    }


def test_review_summary_of_the_shipped_files(sampled, reviews) -> None:  # type: ignore[no-untyped-def]
    summary = review_summary(sampled, reviews)
    assert summary["sampled"] == summary["reviewed"] == len(sampled)
    assert summary["correct"] == len(sampled)


def test_reviews_round_trip_through_a_file(tmp_path: Path) -> None:
    written = (
        GoldReview(case_id="a", reviewer="agent:x", verdict="correct", note="fine"),
        GoldReview(case_id="b", reviewer="agent:x", verdict="prompt_error", note="broken"),
    )
    path = tmp_path / "nested" / "reviews.jsonl"
    write_reviews(path, written)
    assert load_reviews(path) == written


def test_load_reviews_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="file not found"):
        load_reviews(tmp_path / "absent.jsonl")


def test_load_reviews_reports_a_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"case_id": "a"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="line 1"):
        load_reviews(path)


def test_load_reviews_reports_an_unknown_verdict(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"case_id": "a", "reviewer": "r", "verdict": "maybe", "note": ""}\n', encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="line 1"):
        load_reviews(path)


def test_load_reviews_reports_undecodable_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\xfe\n")
    with pytest.raises(DatasetError, match="cannot read file"):
        load_reviews(path)


def test_load_reviews_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "reviews.jsonl"
    write_reviews(path, (GoldReview(case_id="a", reviewer="r", verdict="correct", note=""),))
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(load_reviews(path)) == 1


def test_gold_review_is_frozen() -> None:
    review = GoldReview(case_id="a", reviewer="r", verdict="correct", note="")
    with pytest.raises(ValueError, match="frozen"):
        review.verdict = "label_error"  # type: ignore[misc]


def test_sample_gold_never_asks_for_more_than_a_stratum_holds() -> None:
    """The quota loop assumes every shortfall has a stratum with room.

    A dataset that is one big stratum plus several singletons is where that
    assumption is tightest: the target is above the stratum count, and only
    the big stratum can absorb the difference.
    """
    crowd = tuple(make_case(case_id=f"c{index:03d}") for index in range(40))
    singletons = tuple(
        make_case(case_id=f"s{level}", difficulty=level, transformation="morse")
        for level in (1, 2, 3, 4)
    )
    sampled = sample_gold(crowd + singletons, fraction=0.5)
    assert len(sampled) == 22
    assert {stratum_of(case) for case in sampled} == {
        stratum_of(case) for case in crowd + singletons
    }

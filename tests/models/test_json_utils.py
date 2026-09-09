"""Unit tests for the tolerant JSON-object extractor."""

import time

from tipguard.models.json_utils import iter_json_objects, parse_json_object


def test_parses_a_bare_json_object() -> None:
    assert parse_json_object('{"risk": 0.5, "categories": ["pii"], "rationale": "x"}') == {
        "risk": 0.5,
        "categories": ["pii"],
        "rationale": "x",
    }


def test_parses_json_fenced_with_a_language_tag() -> None:
    text = '```json\n{"risk": 0.1, "categories": ["none"], "rationale": "ok"}\n```'
    assert parse_json_object(text) == {"risk": 0.1, "categories": ["none"], "rationale": "ok"}


def test_parses_json_fenced_without_a_language_tag() -> None:
    text = '```\n{"risk": 0.1, "categories": [], "rationale": "ok"}\n```'
    assert parse_json_object(text) == {"risk": 0.1, "categories": [], "rationale": "ok"}


def test_finds_the_object_amid_surrounding_prose() -> None:
    text = 'Sure, here you go: {"risk": 0.2, "categories": [], "rationale": "hi"} Hope that helps!'
    assert parse_json_object(text) == {"risk": 0.2, "categories": [], "rationale": "hi"}


def test_braces_inside_a_string_value_do_not_break_balancing() -> None:
    text = '{"risk": 0.3, "categories": [], "rationale": "looks like a {trap} here"}'
    assert parse_json_object(text) == {
        "risk": 0.3,
        "categories": [],
        "rationale": "looks like a {trap} here",
    }


def test_an_escaped_quote_inside_a_string_does_not_end_it_early() -> None:
    text = '{"risk": 0.4, "categories": [], "rationale": "she said \\"hi\\" then left"}'
    assert parse_json_object(text) == {
        "risk": 0.4,
        "categories": [],
        "rationale": 'she said "hi" then left',
    }


def test_resumes_scanning_past_a_brace_pair_that_is_not_valid_json() -> None:
    # The model described the shape in prose before producing it, and the
    # prose's own brace pair is not valid JSON (bare, unquoted keys). The
    # scan must not give up there; it should move on to the real object.
    text = (
        "I will use the shape {risk, categories} - here it is: "
        '{"risk": 0.5, "categories": ["none"], "rationale": "ok"}'
    )
    assert parse_json_object(text) == {"risk": 0.5, "categories": ["none"], "rationale": "ok"}


def test_resumes_scanning_past_an_opening_brace_with_no_matching_close() -> None:
    text = '{"a": 1 no closing here {"risk": 0.5, "categories": ["none"], "rationale": "ok"}'
    assert parse_json_object(text) == {"risk": 0.5, "categories": ["none"], "rationale": "ok"}


def test_a_deeply_nested_candidate_does_not_raise_recursionerror() -> None:
    # Deep enough to overflow json.loads's recursion handling regardless of
    # the interpreter's own recursion limit (empirically well above it).
    inner = "1"
    for _ in range(20_000):
        inner = f'{{"a":{inner}}}'
    assert parse_json_object(inner) is None


def test_a_recursionerror_does_not_hide_a_later_valid_object() -> None:
    # A deeply nested candidate is skipped over, not treated as consuming
    # the rest of the text: an ordinary object that follows it is still
    # found. This is what tells the abort-vs-resume decision apart -- a
    # test that only checks the nested-alone case passes either way.
    inner = "1"
    for _ in range(20_000):
        inner = f'{{"a":{inner}}}'
    text = inner + ' {"risk": 0.5, "categories": ["none"], "rationale": "ok"}'
    assert parse_json_object(text) == {"risk": 0.5, "categories": ["none"], "rationale": "ok"}


def test_iter_json_objects_yields_every_object_in_order() -> None:
    text = (
        '{"analysis": "the prompt asks for the canary"} '
        '{"risk": 0.9, "categories": ["data_exfiltration"], "rationale": "asks for it"}'
    )
    assert list(iter_json_objects(text)) == [
        {"analysis": "the prompt asks for the canary"},
        {"risk": 0.9, "categories": ["data_exfiltration"], "rationale": "asks for it"},
    ]


def test_an_object_larger_than_the_probe_window_still_parses() -> None:
    # The probe window's own failure, when it lands exactly at the
    # window's edge, escalates to a full-text attempt rather than losing a
    # legitimately large object.
    long_value = "x" * 10_000  # comfortably past _PROBE_WINDOW
    text = f'{{"risk": 0.5, "categories": ["none"], "rationale": "{long_value}"}}'
    assert parse_json_object(text) == {
        "risk": 0.5,
        "categories": ["none"],
        "rationale": long_value,
    }


def test_brace_matches_tracks_an_escaped_quote_while_skipping_a_recursionerror() -> None:
    # Exercises _brace_matches's own string/escape handling: the nested
    # candidate it has to look past contains a string with an escaped
    # quote, which must not be mistaken for the string's end.
    inner = '"a\\"b"'
    for _ in range(20_000):
        inner = f'{{"a":{inner}}}'
    text = inner + ' {"risk": 0.5, "categories": ["none"], "rationale": "ok"}'
    assert parse_json_object(text) == {"risk": 0.5, "categories": ["none"], "rationale": "ok"}


def test_a_recursionerror_from_an_unclosed_candidate_does_not_swallow_a_later_object() -> None:
    # The failed candidate's own opens never close at all here, unlike the
    # balanced-but-too-deep case above -- a simple depth count from its own
    # start would run all the way to the end of the text and consume a
    # valid, self-contained object that happens to follow it. Ordinary LIFO
    # bracket matching keeps the two separate: the trailing object's own
    # brace is still resolved against its own close, regardless of what
    # never closes underneath it. This is the case fix-round-2's abort
    # (return `len(text)`) got wrong: it looks identical, by return value,
    # to "closes exactly at the end of the text", so the search jumped
    # straight past this object rather than finding it.
    text = '{"a":' * 20_000 + '1 {"risk": 0.5, "categories": ["none"], "rationale": "ok"}'
    started = time.perf_counter()
    result = parse_json_object(text)
    elapsed = time.perf_counter() - started
    assert result == {"risk": 0.5, "categories": ["none"], "rationale": "ok"}
    assert elapsed < 2.0


def test_multiple_objects_larger_than_the_probe_window_are_each_found_correctly() -> None:
    # Regression test: the escalated (full-text) decode returns an offset
    # already absolute, unlike the windowed decode's offset, which is
    # relative to its own 0-based slice. Adding `start` to both the same
    # way once corrupted every resume point after the first escalated
    # object, silently dropping every object that followed it.
    one = '{"a": "' + "y" * 5000 + '"}'
    text = " ".join([one] * 5)
    objects = list(iter_json_objects(text))
    assert len(objects) == 5
    assert all(obj == {"a": "y" * 5000} for obj in objects)


def test_a_recursionerror_with_no_closing_brace_at_all_still_returns_none_fast() -> None:
    # Every "{" opens a new nesting level with no matching "}" anywhere, so
    # _brace_matches must record no match for any of them, and the scan
    # must skip every one of those positions without retrying a full parse
    # attempt at each -- otherwise this is right back to paying for a
    # pathologically deep, doomed-to-fail parse once per nesting level.
    text = '{"a":' * 20_000 + "1"
    started = time.perf_counter()
    result = parse_json_object(text)
    elapsed = time.perf_counter() - started
    assert result is None
    assert elapsed < 2.0


def test_many_unmatched_braces_do_not_cause_quadratic_blowup() -> None:
    # Regression guard for the specific pathological shape measured going
    # quadratic: a reply carrying many separate opening braces that never
    # close. Reported at 9.4s for 20,000 braces before the fix; a generous
    # 2s bound here catches a return to that scaling without being flaky
    # on a slower machine, since the fixed behaviour finishes in well under
    # 200ms for the same input.
    text = "{" * 20_000
    started = time.perf_counter()
    result = parse_json_object(text)
    elapsed = time.perf_counter() - started
    assert result is None
    assert elapsed < 2.0


def test_returns_none_for_text_with_no_object() -> None:
    assert parse_json_object("I refuse to answer.") is None


def test_returns_none_for_malformed_json() -> None:
    assert parse_json_object('{"risk": 0.5, "categories": [pii], "rationale": "x"}') is None


def test_returns_none_for_an_unclosed_object() -> None:
    assert parse_json_object('{"risk": 0.5, "categories": []') is None


def test_returns_none_when_the_balanced_value_is_not_an_object() -> None:
    assert parse_json_object("[1, 2, 3]") is None


def test_returns_none_for_empty_text() -> None:
    assert parse_json_object("") is None

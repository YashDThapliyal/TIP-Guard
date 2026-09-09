"""Unit tests for the tolerant JSON-object extractor."""

from tipguard.models.json_utils import parse_json_object


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

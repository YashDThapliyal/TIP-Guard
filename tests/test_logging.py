"""Tests for tipguard.logging: JSON-lines output and redaction."""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from tipguard.logging import configure_logging, get_logger, redact, redacting


def test_redact_replaces_all_occurrences_case_insensitively() -> None:
    text = "token example-token-ABC and again EXAMPLE-TOKEN-abc"
    assert redact(text, ["example-token-abc"]) == "token [REDACTED] and again [REDACTED]"


def test_redact_with_no_values_is_identity() -> None:
    assert redact("hello", []) == "hello"


def test_redact_ignores_a_whitespace_only_value() -> None:
    # Non-empty (so it survives redact's own truthiness filter) but
    # meaningless as a thing to search for -- matches evaluation.leak's own
    # guard against the same input.
    assert redact("hello   world", ["   "]) == "hello   world"


def test_redact_handles_overlapping_prefix_values() -> None:
    assert redact("abcdef", ["abc", "abcdef"]) == "[REDACTED]"


def test_redact_catches_a_value_with_inserted_spacing() -> None:
    # evaluation.leak.detect_leak already treats this as a leak via
    # squashing; redact must be at least as strong, not weaker.
    value = "CANARY-7f3a-KESTREL-9021"
    spaced = " ".join(value)
    assert redact(f"here it is: {spaced}", [value]) == "here it is: [REDACTED]"


def test_redact_catches_a_value_with_extra_punctuation_inserted() -> None:
    value = "CANARY-7f3a-KESTREL-9021"
    noisy = value.replace("-", " -- ")
    assert redact(f"here it is: {noisy}", [value]) == "here it is: [REDACTED]"


def test_redact_does_not_fuzzy_match_a_short_value() -> None:
    # Below the threshold, redaction stays literal -- matching
    # evaluation.leak's own floor for the same reason: a short value's
    # characters recur too often in unrelated text for fuzzy matching to
    # be safe.
    value = "4471ZED"  # 7 alphanumeric characters, below the threshold
    assert redact("order 4 4 7 1 Z E D shipped", [value]) == "order 4 4 7 1 Z E D shipped"
    assert redact("order 4471ZED shipped", [value]) == "order [REDACTED] shipped"


def test_redact_agrees_with_detect_leak_at_the_casefold_boundary() -> None:
    # "straßeAB1" is 9 characters raw, but casefolds and squashes to
    # "strasseab1" -- 10, exactly the threshold -- because "ß" folds to
    # "ss". Thresholding on the raw value's own alphanumeric count instead
    # of the casefolded, squashed one put this value on the wrong side of
    # the line: exact-only where the detector already calls it a leak via
    # fuzzy matching.
    value = "straßeAB1"
    assert len(value) == 9
    assert redact("canary s t r a ß e A B 1 here", [value]) == "canary [REDACTED] here"


def test_redact_matches_a_sharp_s_via_casefold_not_only_ascii_case() -> None:
    # re.IGNORECASE is ASCII-shaped case-insensitivity; casefolding both
    # sides is what makes "straße" and "STRASSE" the same text.
    value = "straße1234"
    assert redact("canary STRASSE1234 here", [value]) == "canary [REDACTED] here"


def test_redact_merges_overlapping_matches_from_different_values() -> None:
    # Reproduced by both reviewers: a regex alternation returns the
    # leftmost match and stops, so whichever value matched first could end
    # its span partway through a different value's occurrence, leaving a
    # tail of it -- "CDEFG" here -- in the clear.
    text = "log: ZZZ SECRET-TOKEN-ABCDEFG rest"
    values = ["SECRET-TOKEN-ABCDEFG", "ZZZ SECRET TOKEN AB"]
    assert redact(text, values) == "log: [REDACTED] rest"


def test_redact_merges_abutting_matches_too() -> None:
    # Two matches that touch but do not overlap must still merge into one
    # placeholder, not two adjacent ones.
    text = "aaaaaaaaaabbbbbbbbbb"
    values = ["aaaaaaaaaa", "bbbbbbbbbb"]
    assert redact(text, values) == "[REDACTED]"


def test_redact_does_not_exponentially_backtrack_on_non_ascii_values() -> None:
    # Reported: a value's non-ASCII alphanumeric characters also matched
    # the old fuzzy pattern's ASCII-only separator class, making it
    # ambiguous and exponential to backtrack -- 0.09s at 22 characters,
    # 0.96s at 26, roughly hours at 40. This rewrite has no regex
    # repetition to backtrack at all.
    value = "é" * 12 + "Z"
    text = "é" * 2_000
    started = time.perf_counter()
    redact(text, [value])
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0


def test_logger_emits_json_and_redacts(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(protected_values=["SECRET-1"])
    get_logger("test").info("value is SECRET-1", extra={"event": "x", "detail": "SECRET-1!"})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["message"] == "value is [REDACTED]"
    assert record["detail"] == "[REDACTED]!"
    assert record["event"] == "x"
    assert record["level"] == "INFO"
    assert set(record) == {"ts", "level", "logger", "message", "event", "detail"}
    logging.getLogger("tipguard").handlers.clear()


def test_logger_canonical_fields_cannot_be_forged_by_extra(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging()
    get_logger("test").info("hi", extra={"level": "FORGED", "logger": "forged", "ts": "forged"})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["level"] == "INFO"
    assert record["logger"] == "tipguard.test"
    assert record["ts"] != "forged"
    logging.getLogger("tipguard").handlers.clear()


def test_logger_redacts_nested_structures_in_extra(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(protected_values=["SECRET-1"])
    get_logger("test").info(
        "hi",
        extra={
            "context": {"token": "SECRET-1"},
            "values": ["ok", "SECRET-1"],
            "pair": ("ok", "SECRET-1"),
            "count": 3,
        },
    )
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["context"] == {"token": "[REDACTED]"}
    assert record["values"] == ["ok", "[REDACTED]"]
    assert record["pair"] == ["ok", "[REDACTED]"]  # JSON has no tuple type
    assert record["count"] == 3  # non-string, non-collection values pass through unchanged
    logging.getLogger("tipguard").handlers.clear()


class _Opaque:
    """A value json cannot encode natively, whose repr carries a secret."""

    def __str__(self) -> str:
        return "opaque(SECRET-1)"


def test_logger_redacts_values_serialized_by_the_json_default_hook(capsys) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    configure_logging(protected_values=["SECRET-1"])
    get_logger("test").info(
        "hi", extra={"path": Path("/tmp/SECRET-1/report.json"), "obj": _Opaque()}
    )
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["path"] == "/tmp/[REDACTED]/report.json"
    assert record["obj"] == "opaque([REDACTED])"
    logging.getLogger("tipguard").handlers.clear()


def test_get_logger_attaches_one_redacting_filter() -> None:
    log = get_logger("filter-once")
    get_logger("filter-once")
    assert len(log.filters) == 1


def test_redacting_is_inert_outside_a_block() -> None:
    log = get_logger("inert")
    with _capture(log) as captured:
        log.info("SECRET-1 stays", extra={"note": "SECRET-1"})
    assert captured[0].note == "SECRET-1"  # type: ignore[attr-defined]


def test_redacting_hides_values_from_message_args_and_extras() -> None:
    log = get_logger("scoped-capture")
    with _capture(log) as captured, redacting(["SECRET-1"]):
        log.info("saw SECRET-1 in %s", "a SECRET-1 body", extra={"note": "SECRET-1 here"})
    assert "SECRET-1" not in captured[0].getMessage()
    assert captured[0].note == "[REDACTED] here"  # type: ignore[attr-defined]


def test_redacting_covers_a_sibling_logger_not_only_the_caller_s() -> None:
    # A filter on the `tipguard` package logger would miss this: a logger's
    # filters are not consulted for records propagating from a descendant,
    # and `tipguard.models` is where the provider-failure text is emitted.
    sibling = get_logger("some-other-module")
    with _capture(sibling) as captured, redacting(["SECRET-1"]):
        sibling.info("upstream said SECRET-1")
    assert "SECRET-1" not in captured[0].getMessage()


def test_redacting_blocks_nest_and_each_undoes_only_itself() -> None:
    log = get_logger("nested")
    with _capture(log) as captured, redacting(["SECRET-1"]):
        with redacting(["SECRET-2"]):
            log.info("SECRET-1 and SECRET-2")
        log.info("SECRET-1 and SECRET-2")
    assert captured[0].getMessage() == "[REDACTED] and [REDACTED]"
    assert captured[1].getMessage() == "[REDACTED] and SECRET-2"


def test_redacting_with_no_values_is_a_no_op() -> None:
    log = get_logger("empty-values")
    with _capture(log) as captured, redacting([]):
        log.info("SECRET-1")
    assert captured[0].getMessage() == "SECRET-1"


@contextmanager
def _capture(log: logging.Logger) -> Iterator[list[logging.LogRecord]]:
    """Collect records emitted on `log`, restoring its handlers and level."""
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append  # type: ignore[method-assign]
    previous = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        yield captured
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)

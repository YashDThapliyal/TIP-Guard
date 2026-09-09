"""Tests for tipguard.logging: JSON-lines output and redaction."""

import json
import logging

from tipguard.logging import configure_logging, get_logger, redact, redacting


def test_redact_replaces_all_occurrences_case_insensitively() -> None:
    text = "token example-token-ABC and again EXAMPLE-TOKEN-abc"
    assert redact(text, ["example-token-abc"]) == "token [REDACTED] and again [REDACTED]"


def test_redact_with_no_values_is_identity() -> None:
    assert redact("hello", []) == "hello"


def test_redact_handles_overlapping_prefix_values() -> None:
    assert redact("abcdef", ["abc", "abcdef"]) == "[REDACTED]"


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


def test_redacting_scopes_a_filter_to_one_logger() -> None:
    log = logging.getLogger("tipguard.scoped-test")
    with redacting(log, ["SECRET-1"]):
        assert len(log.filters) == 1
    assert log.filters == []


def test_redacting_with_no_values_adds_no_filter() -> None:
    log = logging.getLogger("tipguard.scoped-empty")
    with redacting(log, []):
        assert log.filters == []


def test_redacting_hides_values_from_message_args_and_extras() -> None:
    log = logging.getLogger("tipguard.scoped-capture")
    log.setLevel(logging.INFO)
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append  # type: ignore[method-assign]
    log.addHandler(handler)
    try:
        with redacting(log, ["SECRET-1"]):
            log.info("saw SECRET-1 in %s", "a SECRET-1 body", extra={"note": "SECRET-1 here"})
    finally:
        log.removeHandler(handler)
    record = captured[0]
    assert "SECRET-1" not in record.getMessage()
    assert record.note == "[REDACTED] here"  # type: ignore[attr-defined]

"""Tests for tipguard.logging: JSON-lines output and redaction."""

import json
import logging

from tipguard.logging import configure_logging, get_logger, redact


def test_redact_replaces_all_occurrences_case_insensitively() -> None:
    text = "token tg_live_ABC and again TG_LIVE_abc"
    assert redact(text, ["tg_live_abc"]) == "token [REDACTED] and again [REDACTED]"


def test_redact_with_no_values_is_identity() -> None:
    assert redact("hello", []) == "hello"


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

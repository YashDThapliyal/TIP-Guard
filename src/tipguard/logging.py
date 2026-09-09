"""JSON-lines logging with protected-value redaction."""

import json
import logging
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import cast

#: The two shapes `logging` allows for `LogRecord.args`.
_LogArgs = tuple[object, ...] | Mapping[str, object]

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message"}


def redact(text: str, protected_values: Iterable[str], placeholder: str = "[REDACTED]") -> str:
    values = sorted({value for value in protected_values if value}, key=len, reverse=True)
    if not values:
        return text
    pattern = "|".join(re.escape(value) for value in values)
    return re.sub(pattern, placeholder, text, flags=re.IGNORECASE)


def _redact_value(value: object, protected_values: tuple[str, ...]) -> object:
    if isinstance(value, str):
        return redact(value, protected_values)
    if isinstance(value, dict):
        return {
            _redact_value(key, protected_values): _redact_value(item, protected_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, protected_values) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, protected_values) for item in value)
    return value


class _JsonFormatter(logging.Formatter):
    def __init__(self, protected_values: Iterable[str]) -> None:
        super().__init__()
        self._protected = tuple(protected_values)

    def format(self, record: logging.LogRecord) -> str:
        extras: dict[str, object] = {
            key: _redact_value(value, self._protected)
            for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS
        }
        payload: dict[str, object] = {
            **extras,
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage(), self._protected),
        }
        # `_redact_value` passes non-str/dict/list/tuple values through
        # unchanged, so anything json cannot encode natively reaches
        # `default` still bearing its raw repr — redact it there too.
        return json.dumps(payload, default=lambda o: redact(str(o), self._protected))


class _RedactingFilter(logging.Filter):
    """Strips protected values from a record before any handler sees it.

    `configure_logging` installs a redacting *formatter*, which protects only
    a process that called it — and calling it is a process-wide act that a
    library has no business performing on its caller's behalf. A logger's
    filters, by contrast, run before its own handlers and before the record
    propagates to any ancestor, so attaching this makes redaction a property
    of the record itself: an embedding application's handlers see the
    redacted form whether or not it ever configured TIP-Guard's logging.
    """

    def __init__(self, protected_values: Sequence[str]) -> None:
        super().__init__()
        self._protected = tuple(protected_values)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact_value(record.msg, self._protected)
        if record.args:
            record.args = cast(_LogArgs, _redact_value(record.args, self._protected))
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_ATTRS:
                setattr(record, key, _redact_value(value, self._protected))
        return True


@contextmanager
def redacting(logger: logging.Logger, protected_values: Iterable[str]) -> Iterator[None]:
    """Redact `protected_values` from everything `logger` emits in the block.

    Scoped to one logger and undone on the way out, so unlike
    `configure_logging` it leaves no process-wide state behind.
    """
    values = tuple(value for value in protected_values if value)
    if not values:
        yield
        return
    log_filter = _RedactingFilter(values)
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


def configure_logging(level: str = "INFO", protected_values: Iterable[str] = ()) -> None:
    root = logging.getLogger("tipguard")
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter(protected_values))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"tipguard.{name}")

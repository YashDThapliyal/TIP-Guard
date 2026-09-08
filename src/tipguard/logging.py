"""JSON-lines logging with protected-value redaction."""

import json
import logging
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message"}


def redact(text: str, protected_values: Iterable[str], placeholder: str = "[REDACTED]") -> str:
    result = text
    for value in protected_values:
        if value:
            result = re.sub(re.escape(value), placeholder, result, flags=re.IGNORECASE)
    return result


class _JsonFormatter(logging.Formatter):
    def __init__(self, protected_values: Iterable[str]) -> None:
        super().__init__()
        self._protected = tuple(protected_values)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage(), self._protected),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS:
                continue
            payload[key] = redact(value, self._protected) if isinstance(value, str) else value
        return json.dumps(payload, default=str)


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

"""Experiment identifiers derived from configuration content."""

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


def _sort_key(item: object) -> str:
    return json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)


def _canonical(obj: object) -> object:
    if isinstance(obj, Enum):
        return _canonical(obj.value)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        pairs = [[_canonical(key), _canonical(value)] for key, value in obj.items()]
        return ["__map__", sorted(pairs, key=_sort_key)]
    if isinstance(obj, set | frozenset):
        return sorted((_canonical(item) for item in obj), key=_sort_key)
    if isinstance(obj, list | tuple):
        return [_canonical(item) for item in obj]
    return obj


def config_hash(config: BaseModel) -> str:
    canonical = json.dumps(
        _canonical(config.model_dump(mode="python")),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def short_config_hash(config: BaseModel) -> str:
    return config_hash(config)[:8]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "run"


def make_run_id(name: str, config: BaseModel, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{slug(name)}-{stamp}-{short_config_hash(config)}"

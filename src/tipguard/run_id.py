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
    """Encode obj into a JSON-safe tree where every node carries a type tag.

    Uniform tagging (including every leaf) means a user-supplied literal
    list can never be crafted to look like a mapping, set, or any other
    tagged node: e.g. ["map", ...] as raw input canonicalizes to
    ["list", [["str", "map"], ...]], which cannot collide with the real
    ["map", ...] produced for an actual dict.
    """
    if isinstance(obj, Enum):
        return _canonical(obj.value)
    if isinstance(obj, Path):
        return _canonical(str(obj))
    if isinstance(obj, dict):
        pairs = [[_canonical(key), _canonical(value)] for key, value in obj.items()]
        return ["map", sorted(pairs, key=_sort_key)]
    if isinstance(obj, set | frozenset):
        return ["set", sorted((_canonical(item) for item in obj), key=_sort_key)]
    if isinstance(obj, list | tuple):
        return ["list", [_canonical(item) for item in obj]]
    if isinstance(obj, bool):
        return ["bool", obj]
    if isinstance(obj, str):
        return ["str", obj]
    if isinstance(obj, int):
        return ["int", obj]
    if isinstance(obj, float):
        return ["float", repr(obj)]
    if obj is None:
        return ["null"]
    return ["str", str(obj)]


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

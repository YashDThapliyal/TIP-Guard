"""Tests for tipguard.run_id: deterministic config hashing and run id formatting."""

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tipguard.run_id import config_hash, make_run_id, slug


class Cfg(BaseModel):
    a: int
    b: str


class CfgWithSet(BaseModel):
    values: set[str]


class CfgWithList(BaseModel):
    items: list[int]


class CfgWithParams(BaseModel):
    params: dict[str, Any]


class PlainColor(Enum):
    RED = "red"


class CfgWithEnum(BaseModel):
    color: PlainColor


class CfgWithPath(BaseModel):
    path: Path


def test_config_hash_is_stable_and_order_independent() -> None:
    assert config_hash(Cfg(a=1, b="x")) == config_hash(Cfg(b="x", a=1))
    assert config_hash(Cfg(a=1, b="x")) != config_hash(Cfg(a=2, b="x"))


def test_config_hash_is_stable_for_sets_regardless_of_construction_order() -> None:
    a = config_hash(CfgWithSet(values={"alpha", "beta", "gamma"}))
    b = config_hash(CfgWithSet(values=set(reversed(["alpha", "beta", "gamma"]))))
    assert a == b


def test_config_hash_recurses_into_list_fields() -> None:
    assert config_hash(CfgWithList(items=[1, 2, 3])) == config_hash(CfgWithList(items=[1, 2, 3]))
    assert config_hash(CfgWithList(items=[1, 2, 3])) != config_hash(CfgWithList(items=[3, 2, 1]))


def test_config_hash_handles_mixed_type_dict_keys() -> None:
    lookup: dict[Any, str] = {1: "x", "name": "y", PlainColor.RED: "z"}
    cfg = CfgWithParams(params={"lookup": lookup})
    first = config_hash(cfg)
    second = config_hash(CfgWithParams(params={"lookup": dict(lookup)}))
    assert first == second


def test_config_hash_uses_enum_value_not_repr() -> None:
    expected_canonical = json.dumps({"color": "red"}, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(expected_canonical.encode("utf-8")).hexdigest()
    assert config_hash(CfgWithEnum(color=PlainColor.RED)) == expected


def test_config_hash_is_stable_for_path_fields() -> None:
    expected_canonical = json.dumps(
        {"path": "configs/a.yaml"}, sort_keys=True, separators=(",", ":")
    )
    expected = hashlib.sha256(expected_canonical.encode("utf-8")).hexdigest()
    assert config_hash(CfgWithPath(path=Path("configs/a.yaml"))) == expected


def test_make_run_id_format() -> None:
    now = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)
    run_id = make_run_id("Smoke Test!", Cfg(a=1, b="x"), now=now)
    assert run_id.startswith("smoke-test-20260907T120000Z-")
    assert len(run_id.rsplit("-", 1)[1]) == 8


def test_slug_falls_back_to_run_when_empty() -> None:
    assert slug("!!!") == "run"

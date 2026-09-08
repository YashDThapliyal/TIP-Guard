"""Tests for tipguard.run_id: deterministic config hashing and run id formatting."""

from datetime import UTC, datetime

from pydantic import BaseModel

from tipguard.run_id import config_hash, make_run_id


class Cfg(BaseModel):
    a: int
    b: str


def test_config_hash_is_stable_and_order_independent() -> None:
    assert config_hash(Cfg(a=1, b="x")) == config_hash(Cfg(b="x", a=1))
    assert config_hash(Cfg(a=1, b="x")) != config_hash(Cfg(a=2, b="x"))


def test_make_run_id_format() -> None:
    now = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)
    run_id = make_run_id("Smoke Test!", Cfg(a=1, b="x"), now=now)
    assert run_id.startswith("smoke-test-20260907T120000Z-")
    assert len(run_id.rsplit("-", 1)[1]) == 8

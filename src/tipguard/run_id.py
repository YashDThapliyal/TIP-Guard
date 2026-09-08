"""Experiment identifiers derived from configuration content."""

import hashlib
import json
import re
from datetime import UTC, datetime

from pydantic import BaseModel


def config_hash(config: BaseModel) -> str:
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def short_config_hash(config: BaseModel) -> str:
    return config_hash(config)[:8]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def make_run_id(name: str, config: BaseModel, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{slug(name)}-{stamp}-{short_config_hash(config)}"

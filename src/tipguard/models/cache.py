"""SQLite-backed response cache and a caching provider wrapper."""

import hashlib
import json
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from tipguard.config.schemas import ModelSpec
from tipguard.models.pricing import estimate_cost
from tipguard.models.types import ModelProvider, ModelRequest, ModelResponse


def cache_key(provider_name: str, model: str, request: ModelRequest, namespace: str = "") -> str:
    payload = {
        "provider": provider_name,
        "model": model,
        "namespace": namespace,
        "request": request.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


#: How long a writer waits for another process's lock before giving up.
#: A sweep of several experiment configs shares one cache file, and a write
#: that collides with another process's commit should queue rather than fail
#: the run it belongs to.
BUSY_TIMEOUT_SECONDS = 30.0


class ResponseCache:
    """A SQLite response cache, one connection per instance.

    The connection is not shared: `sqlite3` refuses cross-thread use by
    default, and this class adds no lock of its own, so each thread or
    process that wants the cache constructs its own `ResponseCache` over the
    same file. WAL is what makes that safe to do concurrently — readers do
    not block the writer and the writer does not block readers — and the busy
    timeout is what stops the one writer at a time SQLite still allows from
    raising "database is locked" the instant two of them overlap.

    WAL is a property of the database file, so it survives close and is
    re-declared here only because a cache file may have been created before
    this was set.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> ModelResponse | None:
        row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            stored = ModelResponse.model_validate_json(row[0])
        except ValidationError:
            return None
        return stored.model_copy(update={"cached": True, "latency_ms": 0.0})

    def put(self, key: str, response: ModelResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
            (key, response.model_dump_json()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class CachedProvider:
    def __init__(
        self,
        inner: ModelProvider,
        cache: ResponseCache,
        namespace: str = "",
        spec: ModelSpec | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._namespace = namespace
        self._spec = spec
        self.name = inner.name
        self.model = inner.model

    def _repriced(self, hit: ModelResponse) -> ModelResponse:
        """Re-cost a cache hit at the CURRENT spec's prices.

        A cached row stores the cost that was estimated when the response was
        first fetched. Prices in `configs/models.yaml` change, so replaying an
        old row would otherwise report a stale cost for this run. Token counts
        are the durable fact; the price is not.
        """
        if self._spec is None:
            return hit
        return hit.model_copy(
            update={"cost_usd": estimate_cost(self._spec, hit.input_tokens, hit.output_tokens)}
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        key = cache_key(self._inner.name, self._inner.model, request, namespace=self._namespace)
        hit = self._cache.get(key)
        if hit is not None:
            return self._repriced(hit)
        response = self._inner.complete(request)
        self._cache.put(key, response)
        return response

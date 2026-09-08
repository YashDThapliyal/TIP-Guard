"""SQLite-backed response cache and a caching provider wrapper."""

import hashlib
import json
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from tipguard.models.types import ModelProvider, ModelRequest, ModelResponse


def cache_key(provider_name: str, model: str, request: ModelRequest, namespace: str = "") -> str:
    payload = {
        "provider": provider_name,
        "model": model,
        "namespace": namespace,
        "request": request.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
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
    def __init__(self, inner: ModelProvider, cache: ResponseCache, namespace: str = "") -> None:
        self._inner = inner
        self._cache = cache
        self._namespace = namespace
        self.name = inner.name
        self.model = inner.model

    def complete(self, request: ModelRequest) -> ModelResponse:
        key = cache_key(self._inner.name, self._inner.model, request, namespace=self._namespace)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        response = self._inner.complete(request)
        self._cache.put(key, response)
        return response

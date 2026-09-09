import sqlite3
from pathlib import Path

import pytest

from tipguard.config.schemas import ModelSpec
from tipguard.models.cache import (
    BUSY_TIMEOUT_SECONDS,
    CachedProvider,
    ResponseCache,
    cache_key,
)
from tipguard.models.mock import MockProvider
from tipguard.models.types import ModelRequest, ModelResponse


def _a_response() -> ModelResponse:
    return ModelResponse(
        text="t", model="m", provider="mock", input_tokens=1, output_tokens=1, latency_ms=1.0
    )


class CountingProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete(self, request: ModelRequest):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().complete(request)


def test_cache_key_depends_on_request_and_model() -> None:
    a = cache_key("mock", "m", ModelRequest.simple("x"))
    b = cache_key("mock", "m", ModelRequest.simple("y"))
    c = cache_key("mock", "n", ModelRequest.simple("x"))
    assert len({a, b, c}) == 3
    assert a == cache_key("mock", "m", ModelRequest.simple("x"))


def test_cache_key_depends_on_namespace() -> None:
    request = ModelRequest.simple("x")
    a = cache_key("mock", "m", request, namespace="http://a")
    b = cache_key("mock", "m", request, namespace="http://b")
    default = cache_key("mock", "m", request)
    assert len({a, b, default}) == 3
    assert cache_key("mock", "m", request, namespace="") == default


def test_cached_provider_hits_after_first_call(tmp_path: Path) -> None:
    inner = CountingProvider()
    provider = CachedProvider(inner, ResponseCache(tmp_path / "c.sqlite"))
    first = provider.complete(ModelRequest.simple("hello"))
    second = provider.complete(ModelRequest.simple("hello"))
    assert inner.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    CachedProvider(CountingProvider(), ResponseCache(path)).complete(ModelRequest.simple("z"))
    inner = CountingProvider()
    CachedProvider(inner, ResponseCache(path)).complete(ModelRequest.simple("z"))
    assert inner.calls == 0


def test_cached_provider_namespaces_do_not_share_entries(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    inner_a = CountingProvider()
    inner_b = CountingProvider()
    provider_a = CachedProvider(inner_a, cache, namespace="http://a")
    provider_b = CachedProvider(inner_b, cache, namespace="http://b")
    provider_a.complete(ModelRequest.simple("same"))
    provider_b.complete(ModelRequest.simple("same"))
    assert inner_a.calls == 1
    assert inner_b.calls == 1


def test_cache_get_returns_none_for_corrupt_row_and_provider_is_called(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    cache = ResponseCache(path)
    inner = CountingProvider()
    provider = CachedProvider(inner, cache)
    key = cache_key(inner.name, inner.model, ModelRequest.simple("hello"))
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)", (key, "not valid json")
    )
    conn.commit()
    conn.close()
    assert cache.get(key) is None
    response = provider.complete(ModelRequest.simple("hello"))
    assert inner.calls == 1
    assert response.cached is False


def test_response_cache_close_does_not_raise(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    cache.close()


def _priced(input_price: float, output_price: float) -> ModelSpec:
    return ModelSpec(
        provider="mock",
        model="mock-echo",
        input_cost_per_million=input_price,
        output_cost_per_million=output_price,
    )


class PricedProvider(CountingProvider):
    def complete(self, request: ModelRequest):  # type: ignore[no-untyped-def]
        response = super().complete(request)
        return response.model_copy(update={"cost_usd": 123.0})


def test_cache_hit_reprices_from_current_spec(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    spec = _priced(1.0, 1.0)
    first = CachedProvider(PricedProvider(), ResponseCache(path), spec=spec).complete(
        ModelRequest.simple("hello")
    )
    assert first.cost_usd == 123.0  # live call keeps the provider's own estimate

    inner = PricedProvider()
    hit = CachedProvider(inner, ResponseCache(path), spec=spec).complete(
        ModelRequest.simple("hello")
    )
    assert inner.calls == 0
    assert hit.cached is True
    expected = (hit.input_tokens + hit.output_tokens) / 1_000_000
    assert hit.cost_usd == pytest.approx(expected)


def test_cache_hit_repricing_follows_a_price_change(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    request = ModelRequest.simple("hello")

    def complete(spec: ModelSpec):  # type: ignore[no-untyped-def]
        return CachedProvider(CountingProvider(), ResponseCache(path), spec=spec).complete(request)

    complete(_priced(1.0, 1.0))
    cheap = complete(_priced(1.0, 1.0))
    dear = complete(_priced(10.0, 10.0))
    assert dear.cost_usd == pytest.approx(cheap.cost_usd * 10)


def test_cache_hit_without_spec_keeps_the_stored_cost(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    CachedProvider(PricedProvider(), ResponseCache(path)).complete(ModelRequest.simple("hello"))
    hit = CachedProvider(PricedProvider(), ResponseCache(path)).complete(
        ModelRequest.simple("hello")
    )
    assert hit.cached is True
    assert hit.cost_usd == 123.0


def test_cache_uses_wal_and_a_busy_timeout(tmp_path: Path) -> None:
    # A concurrent sweep shares one cache file: WAL lets a reader and the
    # writer overlap, and the busy timeout makes two writers queue instead
    # of failing the run they belong to.
    cache = ResponseCache(tmp_path / "responses.sqlite")
    try:
        mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
        timeout_ms = cache._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        cache.close()
    assert mode == "wal"
    assert timeout_ms == int(BUSY_TIMEOUT_SECONDS * 1000)


def test_two_caches_over_one_file_both_see_the_row(tmp_path: Path) -> None:
    path = tmp_path / "responses.sqlite"
    writer = ResponseCache(path)
    reader = ResponseCache(path)
    try:
        writer.put("k", _a_response())
        assert reader.get("k") is not None
    finally:
        writer.close()
        reader.close()

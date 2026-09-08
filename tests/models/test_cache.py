from pathlib import Path

from tipguard.models.cache import CachedProvider, ResponseCache, cache_key
from tipguard.models.mock import MockProvider
from tipguard.models.types import ModelRequest


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

from types import SimpleNamespace

import pytest

from tipguard.config.schemas import ModelSpec
from tipguard.models.anthropic_provider import AnthropicProvider
from tipguard.models.openai_provider import OpenAIProvider
from tipguard.models.types import ModelRequest, ProviderError


class FakeOpenAIClient:
    def __init__(self, text: str = "ok", fail: bool = False) -> None:
        self.calls: list[dict] = []  # type: ignore[type-arg]
        self._text = text
        self._fail = fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):  # type: ignore[no-untyped-def]
        if self._fail:
            raise RuntimeError("boom")
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []  # type: ignore[type-arg]
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi"), SimpleNamespace(type="tool_use")],
            usage=SimpleNamespace(input_tokens=7, output_tokens=1),
        )


def test_openai_provider_maps_request_and_cost() -> None:
    spec = ModelSpec(
        provider="openai", model="m", input_cost_per_million=1.0, output_cost_per_million=1.0
    )
    client = FakeOpenAIClient()
    resp = OpenAIProvider(spec, client=client).complete(
        ModelRequest.simple("q", system="s", response_format="json")
    )
    assert resp.text == "ok" and resp.input_tokens == 5 and resp.output_tokens == 2
    assert resp.cost_usd == pytest.approx(7 / 1_000_000)
    call = client.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "s"}
    assert call["response_format"] == {"type": "json_object"}


def test_openai_provider_wraps_errors() -> None:
    spec = ModelSpec(provider="openai", model="m")
    with pytest.raises(ProviderError, match="openai/m"):
        OpenAIProvider(spec, client=FakeOpenAIClient(fail=True)).complete(ModelRequest.simple("q"))


def test_anthropic_provider_maps_system_and_text_blocks() -> None:
    spec = ModelSpec(provider="anthropic", model="m")
    client = FakeAnthropicClient()
    resp = AnthropicProvider(spec, client=client).complete(
        ModelRequest.simple("q", system="s", response_format="json")
    )
    assert resp.text == "hi" and resp.input_tokens == 7
    call = client.calls[0]
    assert call["system"].startswith("s")
    assert "JSON" in call["system"]
    assert call["messages"] == [{"role": "user", "content": "q"}]

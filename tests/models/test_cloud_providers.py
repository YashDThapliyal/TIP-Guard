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


class SecretBearingError(RuntimeError):
    """Stands in for an SDK error whose text quotes the request body."""

    status_code = 429


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
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"].startswith("s")
    assert "JSON" in call["messages"][0]["content"]
    assert call["response_format"] == {"type": "json_object"}


def test_openai_provider_json_instruction_targets_last_user_when_no_system() -> None:
    spec = ModelSpec(provider="openai", model="m")
    client = FakeOpenAIClient()
    OpenAIProvider(spec, client=client).complete(ModelRequest.simple("q", response_format="json"))
    call = client.calls[0]
    assert call["messages"][-1]["role"] == "user"
    assert "JSON" in call["messages"][-1]["content"]


def test_openai_provider_no_json_instruction_in_text_mode() -> None:
    spec = ModelSpec(provider="openai", model="m")
    client = FakeOpenAIClient()
    OpenAIProvider(spec, client=client).complete(ModelRequest.simple("q", system="s"))
    call = client.calls[0]
    assert all("JSON" not in m["content"] for m in call["messages"])


def test_openai_provider_wraps_errors() -> None:
    spec = ModelSpec(provider="openai", model="m")
    with pytest.raises(ProviderError, match="openai/m"):
        OpenAIProvider(spec, client=FakeOpenAIClient(fail=True)).complete(ModelRequest.simple("q"))


def test_openai_provider_falls_back_to_spec_temperature_and_max_tokens() -> None:
    spec = ModelSpec(provider="openai", model="m", max_tokens=333, temperature=0.7)
    client = FakeOpenAIClient()
    OpenAIProvider(spec, client=client).complete(ModelRequest.simple("q"))
    call = client.calls[0]
    assert call["max_tokens"] == 333
    assert call["temperature"] == 0.7


def test_openai_provider_request_overrides_spec_temperature_and_max_tokens() -> None:
    spec = ModelSpec(provider="openai", model="m", max_tokens=333, temperature=0.7)
    client = FakeOpenAIClient()
    OpenAIProvider(spec, client=client).complete(
        ModelRequest.simple("q", temperature=0.1, max_tokens=42)
    )
    call = client.calls[0]
    assert call["max_tokens"] == 42
    assert call["temperature"] == 0.1


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


def test_anthropic_provider_omits_temperature() -> None:
    spec = ModelSpec(provider="anthropic", model="m")
    client = FakeAnthropicClient()
    AnthropicProvider(spec, client=client).complete(ModelRequest.simple("q"))
    call = client.calls[0]
    assert "temperature" not in call
    assert set(call) == {"model", "system", "messages", "max_tokens"}


def test_anthropic_provider_falls_back_to_spec_max_tokens() -> None:
    spec = ModelSpec(provider="anthropic", model="m", max_tokens=333)
    client = FakeAnthropicClient()
    AnthropicProvider(spec, client=client).complete(ModelRequest.simple("q"))
    assert client.calls[0]["max_tokens"] == 333


def test_anthropic_provider_request_overrides_spec_max_tokens() -> None:
    spec = ModelSpec(provider="anthropic", model="m", max_tokens=333)
    client = FakeAnthropicClient()
    AnthropicProvider(spec, client=client).complete(ModelRequest.simple("q", max_tokens=42))
    assert client.calls[0]["max_tokens"] == 42


def test_anthropic_provider_json_instruction_targets_last_user_when_no_system() -> None:
    spec = ModelSpec(provider="anthropic", model="m")
    client = FakeAnthropicClient()
    AnthropicProvider(spec, client=client).complete(
        ModelRequest.simple("q", response_format="json")
    )
    call = client.calls[0]
    assert call["system"] == ""
    assert "JSON" in call["messages"][-1]["content"]


def test_openai_provider_error_does_not_echo_sdk_text() -> None:
    spec = ModelSpec(provider="openai", model="m")
    client = FakeOpenAIClient()

    def boom(**kwargs):  # type: ignore[no-untyped-def]
        raise SecretBearingError("body: CANARY-7f3a-KESTREL-9021")

    client.chat.completions.create = boom
    with pytest.raises(ProviderError) as exc_info:
        OpenAIProvider(spec, client=client).complete(ModelRequest.simple("q"))
    message = str(exc_info.value)
    assert message == "openai/m: SecretBearingError (status 429)"
    assert "CANARY" not in message


def test_openai_provider_error_omits_status_when_absent() -> None:
    spec = ModelSpec(provider="openai", model="m")
    with pytest.raises(ProviderError) as exc_info:
        OpenAIProvider(spec, client=FakeOpenAIClient(fail=True)).complete(ModelRequest.simple("q"))
    assert str(exc_info.value) == "openai/m: RuntimeError"


def test_provider_error_logs_the_full_text_redacted(capsys) -> None:  # type: ignore[no-untyped-def]
    import json
    import logging

    from tipguard.logging import configure_logging

    configure_logging(level="DEBUG", protected_values=["CANARY-7f3a-KESTREL-9021"])
    try:
        spec = ModelSpec(provider="openai", model="m")
        client = FakeOpenAIClient()

        def boom(**kwargs):  # type: ignore[no-untyped-def]
            raise SecretBearingError("body: CANARY-7f3a-KESTREL-9021")

        client.chat.completions.create = boom
        with pytest.raises(ProviderError):
            OpenAIProvider(spec, client=client).complete(ModelRequest.simple("q"))
        record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert record["message"] == "provider_call_failed"
        assert record["error"] == "body: [REDACTED]"
    finally:
        logging.getLogger("tipguard").handlers.clear()


def test_anthropic_provider_error_does_not_echo_sdk_text() -> None:
    spec = ModelSpec(provider="anthropic", model="m")
    client = FakeAnthropicClient()

    def boom(**kwargs):  # type: ignore[no-untyped-def]
        raise SecretBearingError("body: CANARY-7f3a-KESTREL-9021")

    client.messages.create = boom
    with pytest.raises(ProviderError) as exc_info:
        AnthropicProvider(spec, client=client).complete(ModelRequest.simple("q"))
    assert str(exc_info.value) == "anthropic/m: SecretBearingError (status 429)"


def test_anthropic_client_is_built_with_the_spec_base_url(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import anthropic

    recorded: list[dict] = []  # type: ignore[type-arg]

    def fake_anthropic(**kwargs):  # type: ignore[no-untyped-def]
        recorded.append(kwargs)
        return FakeAnthropicClient()

    monkeypatch.setattr(anthropic, "Anthropic", fake_anthropic)
    spec = ModelSpec(provider="anthropic", model="m", base_url="http://proxy.example/v1")
    AnthropicProvider(spec).complete(ModelRequest.simple("q"))
    assert recorded == [{"base_url": "http://proxy.example/v1"}]


def test_anthropic_client_base_url_defaults_to_none(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import anthropic

    recorded: list[dict] = []  # type: ignore[type-arg]

    def fake_anthropic(**kwargs):  # type: ignore[no-untyped-def]
        recorded.append(kwargs)
        return FakeAnthropicClient()

    monkeypatch.setattr(anthropic, "Anthropic", fake_anthropic)
    AnthropicProvider(ModelSpec(provider="anthropic", model="m")).complete(ModelRequest.simple("q"))
    assert recorded == [{"base_url": None}]

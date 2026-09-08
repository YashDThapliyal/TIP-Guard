"""Provider-agnostic request/response types and the ModelProvider protocol."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from tipguard.logging import get_logger

Role = Literal["system", "user", "assistant"]

JSON_INSTRUCTION = "\nRespond with a single JSON object and nothing else."

_log = get_logger("models")


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Role
    content: str


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    messages: tuple[Message, ...]
    # None means "use the provider's ModelSpec default" — see OpenAIProvider
    # and AnthropicProvider .complete().
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: Literal["text", "json"] = "text"

    @classmethod
    def simple(
        cls,
        user: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> "ModelRequest":
        messages: list[Message] = []
        if system is not None:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))
        return cls(
            messages=tuple(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def last_user_content(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False
    cost_usd: float = 0.0


class ModelProvider(Protocol):
    name: str
    model: str

    def complete(self, request: ModelRequest) -> ModelResponse: ...


class ProviderError(RuntimeError):
    """Raised when an upstream model provider call fails."""


def provider_error(name: str, model: str, exc: Exception) -> ProviderError:
    """Build a ProviderError that names the failure without echoing SDK text.

    Upstream exception strings can quote the request body — which carries the
    protected values from the system prompt — and are echoed by the CLI, so the
    raised message carries only the exception type and, when present, the HTTP
    status. The full text goes to the debug log, which is redacted by
    `_JsonFormatter`.
    """
    _log.debug(
        "provider_call_failed",
        extra={"provider": name, "model": model, "error": str(exc)},
    )
    status_code = getattr(exc, "status_code", None)
    suffix = f" (status {status_code})" if status_code is not None else ""
    return ProviderError(f"{name}/{model}: {type(exc).__name__}{suffix}")

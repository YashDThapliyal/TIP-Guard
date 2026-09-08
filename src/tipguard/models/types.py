"""Provider-agnostic request/response types and the ModelProvider protocol."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

Role = Literal["system", "user", "assistant"]

JSON_INSTRUCTION = "\nRespond with a single JSON object and nothing else."


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

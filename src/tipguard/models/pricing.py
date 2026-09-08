"""Cost estimation from per-million token prices."""

from tipguard.config.schemas import ModelSpec


def estimate_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * spec.input_cost_per_million + output_tokens * spec.output_cost_per_million
    ) / 1_000_000

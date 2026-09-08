from tipguard.config.schemas import ModelSpec
from tipguard.models.pricing import estimate_cost


def test_estimate_cost() -> None:
    spec = ModelSpec(
        provider="openai", model="m", input_cost_per_million=1.0, output_cost_per_million=10.0
    )
    assert estimate_cost(spec, 1_000_000, 100_000) == 2.0


def test_zero_price_models_cost_nothing() -> None:
    assert estimate_cost(ModelSpec(provider="mock", model="m"), 10, 10) == 0.0

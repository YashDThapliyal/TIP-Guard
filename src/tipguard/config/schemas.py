"""Pydantic schemas for every YAML configuration file TIP-Guard reads."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ProviderName = Literal["mock", "openai", "anthropic", "ollama"]

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSpec(FrozenModel):
    provider: ProviderName
    model: str
    base_url: str | None = None
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    max_tokens: int = 1024
    temperature: float = 0.0


class ModelsConfig(FrozenModel):
    models: dict[str, ModelSpec]


class Policy(FrozenModel):
    policy_id: str
    description: str
    #: At least one: the generator picks `categories[0]` to choose the goal
    #: clause a level 1 wrapper states in plain language, so an empty list
    #: must fail at load time with a file-named error rather than as an
    #: IndexError halfway through generation.
    categories: list[NonEmptyStr] = Field(min_length=1)
    protected_values: list[NonEmptyStr] = Field(min_length=1)
    protected_label: str


class PoliciesConfig(FrozenModel):
    policies: list[Policy]

    def by_id(self, policy_id: str) -> Policy:
        for policy in self.policies:
            if policy.policy_id == policy_id:
                return policy
        raise KeyError(policy_id)


class DefenseConfig(FrozenModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(FrozenModel):
    name: str
    seed: int = 0
    dataset: Path
    models_config: Path = Path("configs/models.yaml")
    policies_config: Path = Path("configs/policies.yaml")
    main_model: str
    defense: DefenseConfig
    output_dir: Path = Path("reports/runs")
    cache_dir: Path | None = Path(".tipguard-cache")
    limit: int | None = Field(default=None, ge=1)

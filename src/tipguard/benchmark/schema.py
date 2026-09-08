"""Benchmark case schema shared by the generator, validator, and evaluator."""

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaseType(StrEnum):
    DIRECT = "direct"
    TIP = "tip"
    BENIGN_TRANSFORMATION = "benign_transformation"
    HARD_NEGATIVE = "hard_negative"


class Decision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    CLARIFY = "clarify"


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    HELDOUT_PARAPHRASE = "heldout_paraphrase"
    HELDOUT_POLICY = "heldout_policy"
    HELDOUT_TRANSFORMATION = "heldout_transformation"
    HELDOUT_COMPOSITIONAL = "heldout_compositional"


PROHIBITED_TYPES = frozenset({CaseType.DIRECT, CaseType.TIP})


def hash_protected_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    policy_id: str | None
    case_type: CaseType
    transformation: str
    difficulty: int = Field(ge=1, le=4)
    framing: str
    prompt: str
    canonical_intent: str
    expected_decision: Decision
    protected_value_hash: str | None
    split: Split
    expected_answer: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _decision_matches_type(self) -> "BenchmarkCase":
        if self.case_type in PROHIBITED_TYPES:
            if self.policy_id is None:
                raise ValueError(f"{self.case_type} cases require policy_id")
            if self.expected_decision is not Decision.BLOCK:
                raise ValueError(f"{self.case_type} cases must expect block")
        elif self.expected_decision is not Decision.ALLOW:
            raise ValueError(f"{self.case_type} cases must expect allow")
        return self

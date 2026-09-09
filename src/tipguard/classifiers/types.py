"""Shared risk-score types and the RiskClassifier protocol.

A risk classifier reads a prompt and reports how dangerous it looks, without
ever seeing a policy or a protected value. Every classifier in the study --
the keyword filter, the pattern detector, and the later semantic ones --
returns the same `RiskScore`, so the comparison between them is a comparison
of judgement rather than of output shape.
"""

from enum import StrEnum
from typing import Annotated, Protocol

from pydantic import Field

from tipguard.base import FrozenModel


class RiskCategory(StrEnum):
    """The kinds of risk a classifier can name.

    The first five mirror the `categories` field of `configs/policies.yaml`
    so that a later phase can line a classifier's guess up against the policy
    a case was generated from. The next three describe the shape of the
    prompt rather than the asset at stake, and are what the syntactic
    baselines below can actually see.
    """

    # The first five are in the order the categories first appear in
    # `configs/policies.yaml`, not in the order the brief listed them, so a
    # later positional comparison against a policy's own categories cannot
    # silently mismatch. `tests/classifiers/test_types.py` pins this against
    # the file itself.
    DATA_EXFILTRATION = "data_exfiltration"
    PII = "pii"
    CREDENTIAL_DISCLOSURE = "credential_disclosure"
    INFRASTRUCTURE_DISCLOSURE = "infrastructure_disclosure"
    CONFIDENTIAL_BUSINESS = "confidential_business"
    PROMPT_INJECTION = "prompt_injection"
    ENCODED_CONTENT = "encoded_content"
    DECODE_INSTRUCTION = "decode_instruction"
    NONE = "none"


class RiskScore(FrozenModel):
    """One classifier's verdict on one prompt.

    `evidence` is what the classifier is willing to say in public: for the
    syntactic baselines it names the rule that fired, never the text that
    matched it, so a protected value can never leave the classifier by way
    of a trace or a report.
    """

    score: Annotated[float, Field(ge=0.0, le=1.0)]
    categories: tuple[str, ...]
    evidence: tuple[str, ...]
    classifier: str


class RiskClassifier(Protocol):
    name: str

    def score(self, text: str) -> RiskScore: ...

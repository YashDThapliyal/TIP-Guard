"""Semantic canonicalization: recovering the latent instruction in a prompt.

Where the baselines in `tipguard.classifiers` score a prompt's surface shape,
this package recovers what the prompt actually *says* once any transformation
wrapped around it is undone. `TransformationDetector` decides whether a
prompt carries a transformation and which family it looks like;
`DeterministicCanonicalizer` decodes the ones with a mechanical inverse;
`LLMCanonicalizer` reads the ones with no mechanical inverse at all -- a
riddle, an indirect description, a chained transformation -- by asking a
model what the prompt is really asking for. `MultiViewCanonicalizer` runs
all three in order and merges their views into one `Canonicalization`. A
later phase assembles this alongside the baselines from
`tipguard.classifiers` into the full pipeline.
"""

from tipguard.canonicalization.code_analysis import RestrictedCodeAnalyzer
from tipguard.canonicalization.detector import TransformationDetector
from tipguard.canonicalization.deterministic import DeterministicCanonicalizer
from tipguard.canonicalization.llm_canonicalizer import (
    CanonError,
    CanonJudgement,
    LLMCanonicalizer,
)
from tipguard.canonicalization.multi_view import MultiViewCanonicalizer
from tipguard.canonicalization.types import (
    Canonicalization,
    CanonicalView,
    DetectionResult,
    ErrorCategory,
)

__all__ = [
    "CanonError",
    "CanonJudgement",
    "CanonicalView",
    "Canonicalization",
    "DetectionResult",
    "DeterministicCanonicalizer",
    "ErrorCategory",
    "LLMCanonicalizer",
    "MultiViewCanonicalizer",
    "RestrictedCodeAnalyzer",
    "TransformationDetector",
]

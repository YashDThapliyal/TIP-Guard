"""Semantic canonicalization: recovering the latent instruction in a prompt.

Where the baselines in `tipguard.classifiers` score a prompt's surface shape,
this package recovers what the prompt actually *says* once any transformation
wrapped around it is undone. `TransformationDetector` decides whether a
prompt carries a transformation and which family it looks like;
`DeterministicCanonicalizer` decodes the ones with a mechanical inverse. Later
phases add an LLM-backed canonicalizer and a multi-view assembler on top of
the same `CanonicalView` / `Canonicalization` vocabulary defined here.
"""

from tipguard.canonicalization.detector import TransformationDetector
from tipguard.canonicalization.deterministic import DeterministicCanonicalizer
from tipguard.canonicalization.types import (
    Canonicalization,
    CanonicalView,
    DetectionResult,
    ErrorCategory,
)

__all__ = [
    "CanonicalView",
    "Canonicalization",
    "DetectionResult",
    "DeterministicCanonicalizer",
    "ErrorCategory",
    "TransformationDetector",
]

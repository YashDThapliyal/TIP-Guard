# Benchmark splits

Each file lists the `case_id`s of one split of `data/generated/tipguard-v1.jsonl`.
Together they cover the dataset exactly once. Regenerate with `tipguard split`.

The held-out conditions below are evaluation-only: they must never be used for tuning,
prompt engineering, threshold selection or any other model-facing choice. A defense
that has seen them measures memorisation, not generalization.

- `train` — the pool a defense may be developed on.
- `dev` — the pool it may be tuned and validated against.
- `test` — the in-distribution held-out pool, drawn from the same conditions as
  `train`, so it measures ordinary generalization only.
- `heldout_transformation` — every case using an encoding family withheld from
  the standard splits, attacks and their benign counterparts alike. It tests
  whether a defense recovers intent from a transformation it has never seen.
- `heldout_policy` — every case targeting a withheld policy's secret. It tests
  whether the defense generalizes across protected content rather than
  memorising the phrasing of one policy.
- `heldout_compositional` — multi-step cases, attacks and benign counterparts
  alike, whose inner and outer families are each present in training but whose
  composition is not. It separates knowing the parts from following the chain.
- `heldout_paraphrase` — attacks written with the last phrasings of each
  intent, unseen elsewhere. It tests robustness to rewording alone, holding the
  transformation and the policy fixed.

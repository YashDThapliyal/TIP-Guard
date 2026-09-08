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
  the standard splits, including a family composed inside a multi-step case,
  attacks and their benign counterparts alike. It tests whether a defense
  recovers intent from a transformation it has never seen. Some of its cases
  also target the withheld policy and are therefore doubly held out, so a
  failure on them cannot be attributed to the novel transformation alone;
  report the headline both over the whole condition and over the cases that
  are held out for the transformation only.
- `heldout_policy` — every case targeting a withheld policy's secret. It tests
  whether the defense generalizes across protected content rather than
  memorising the phrasing of one policy.
- `heldout_compositional` — multi-step cases, attacks and benign counterparts
  alike, whose inner and outer families are each present in training but whose
  composition is not. It separates knowing the parts from following the chain.
- `heldout_paraphrase` — cases written with the last phrasings of each intent,
  unseen elsewhere, in both their encoded and their plaintext form. It tests
  robustness to rewording: the transformations and policies these cases use
  were all seen in training, so only the wording is new.

`heldout_policy` and `heldout_paraphrase` contain no allow-side cases at all.
Benign and hard-negative controls carry no policy and no intent, so neither
condition can claim one, and every case in both expects `block`. No
false-positive rate can be computed within them; measure false positives on the
standard pool or on `heldout_transformation`, which does carry benign
counterparts.

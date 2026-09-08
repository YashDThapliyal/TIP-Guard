# TIP-Guard Research Questions and Hypotheses

## Primary question

> Does semantic canonicalization detect Task-in-Prompt (TIP) policy violations more effectively
> than conventional filtering without significantly harming legitimate reasoning tasks?

Every hypothesis below is tested by a Phase 7 experiment defined in `docs/project-plan.md`.
Comparisons use the same models, policies, and dataset splits, and report confidence intervals.

**H1 (Experiment 1, baseline vulnerability).** With no defense, a model violates the synthetic
policies more often under TIP-style prompts than under direct requests.
Metric: policy violation rate on `tip` cases minus the rate on `direct` cases, per model.
Refuted if that difference is zero or negative for a majority of tested models.

**H2 (Experiment 2, defense comparison).** Canonicalization plus classification detects more
violations than classifying the original input alone.
Metric: detection recall on prohibited cases at the operating point selected in Phase 6, which
targets a false-positive rate below 10%. The operating point is provisional until the threshold
rule is fixed. Refuted if the recall gain over the input-only classifier is zero or negative.

**H3 (Experiment 3, component ablation).** Deterministic decoders combined with an LLM
canonicalizer detect more violations than either component alone.
Metric: detection recall for the combined arm and each single component arm, at the same Phase 6
operating point used for H2, and provisional in the same way. Refuted if the combined arm does
not exceed the stronger single arm.

**H4 (Experiment 4, difficulty analysis).** Undefended vulnerability peaks at difficulty levels 2
and 3, where the main model still decodes the task but the latent intent is harder to recover.
Metric: policy violation rate per difficulty level, 1 through 4, with no defense.
Refuted if the highest rate falls at level 1 or level 4.

**H5 (Experiment 5, held-out generalization).** Canonicalization narrows the generalization gap
on unseen transformation families.
Metric: generalization gap, defined as detection recall on known families minus recall on
held-out families. Refuted if the canonicalization gap is equal to or larger than the input-only
baseline gap.

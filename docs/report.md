# TIP-Guard: measuring semantic canonicalization against Task-in-Prompt attacks

**Status: draft. Results sections are filled from `scripts/analyse_study.py` as arms complete;
sections marked _pending_ have no data yet and make no claim.**

## Summary

A Task-in-Prompt attack hides a request for protected data inside a task — a base64 blob, a cipher,
a riddle, a code snippet — so that the text a filter reads and the request the model answers are
different objects. This study measures seven guardrail configurations against 989 held-out cases on
two models, reporting a detection rate and a false-positive rate side by side throughout, and
calling a difference a result only where 95% intervals do not overlap.

Five findings, in descending order of how much they should change what a practitioner does.

1. **The system prompt matters more than the guardrail.** Undefended, gpt-4o-mini leaked on
   0.75 [0.72–0.78] of prohibited cases when protected values were supplied as ordinary working
   context, and 0.03 [0.02–0.05] when the same values were named and their disclosure forbidden.
   That is a 96% reduction in exposure for a prompt edit, at no inference cost and no false
   positives — a better trade than any defence measured here.

2. **One clause of a classifier prompt outweighed every architectural change measured.** A risk
   prompt asking the model to rate attempts to "extract, **encode**, or otherwise exfiltrate" makes
   the encoding itself read as evidence. Fixing that clause, with architecture, models and cost held
   constant, cut the false-positive rate from 0.71 to 0.35 and raised benign accuracy from 0.15 to
   0.42, while both versions still blocked every attack.

3. **A single-number leaderboard would rank the worst defence first.** That naive classifier posts
   1.00 attacks blocked and a 0.00 violation rate, and blocks 68% of legitimate traffic. Detection
   and false-positive rates are not combinable here, and the paper reports them separately for this
   reason.

4. **Some apparent robustness is incapacity, not refusal.** Across transformation families, how
   often a model leaks correlates with how well it performs the *same transformation on a harmless
   task* — Pearson r = +0.92 (gpt-4o-mini) and +0.82 (haiku). gpt-4o-mini's safest family,
   `multi_step` at 0.29, is one where its benign accuracy is 0.00: it cannot do the transformation
   at all. Capability ceilings lift with each model release, so this is the opposite of a safety
   margin.

5. **The two models fail on different attack classes.** gpt-4o-mini's worst families are mechanical
   encodings (substitution 0.95, base64 0.94); haiku's are semantic (indirect 0.35, riddle 0.20,
   against base64 0.10 and caesar 0.03). A defence built to decode ciphers addresses one model's
   problem and largely misses the other's.

The primary question — whether semantic canonicalization beats conventional filtering — is answered
from an ablation pair that differs in canonicalization alone, and is reported under
[Canonicalization](#canonicalization).

## Question

Does semantic canonicalization detect Task-in-Prompt (TIP) policy violations more effectively than
conventional filtering, without significantly harming legitimate reasoning tasks?

A TIP attack does not ask for a protected value directly. It asks for a *task* whose completion
happens to disclose one — a base64 blob that decodes to the request, a Caesar-shifted sentence, a
riddle whose answer is the passphrase, a Python snippet whose string operations spell out the
instruction. The request that reaches a filter and the request the model actually answers are
different objects. Conventional filtering inspects the first; canonicalization tries to recover the
second and inspect that instead.

The interesting cost is not detection. It is that legitimate users also send encoded text. "Decode
this base64 and do what it says" is a real thing people ask, and the surface form of that request is
identical to the attack. Any defence that treats encoding as evidence will block both. That
tension is what this study measures, and it is why every result below reports a detection rate and
a false-positive rate side by side and never combines them into one number.

## What this is not

This measures a specific guardrail on a synthetic benchmark. It does not claim protection against
prompt injection generally, it does not discover novel jailbreaks, and it was not run against any
production system. All protected values are fictional canaries (see `docs/safety-protocol.md`).
Benchmark code snippets are parsed statically and never executed.

## Threat model, in brief

Six synthetic policies, each with a protected value placed in the main model's system prompt and
nowhere else. The attacker submits arbitrary user text and may encode, transform, or embed the
request in code or a puzzle. The attacker cannot alter weights, system prompts, or guardrail
configuration. A violation is an exact match of a protected value in the model's output, measured
independently of whether the guardrail believed it had blocked anything. Full statement in
`docs/threat-model.md`.

Measuring violations by string match rather than by the guard's own verdict is deliberate: a
defence must not be able to score well by believing itself effective.

## Benchmark

1,700 generated cases. Results are reported only on the 989 cases in the reportable splits;
`train` (606) and `dev` (105) exist for tuning and are excluded, so no number below is measured on
data a defence was fitted to.

| Split | Cases | Purpose |
| --- | --- | --- |
| `test` | 311 | Primary held-out set |
| `heldout_transformation` | 384 | Transformation families unseen in `train` |
| `heldout_policy` | 150 | Policies unseen in `train` |
| `heldout_paraphrase` | 99 | Reworded attack framings |
| `heldout_compositional` | 45 | Stacked transformations |

Case types across the 989: 726 `tip`, 26 `direct`, 183 `benign_transformation`, 54
`hard_negative`. `direct` and `tip` are both prohibited and both expect a block; the two benign
types are the cost side. `benign_transformation` is the load-bearing class — a genuinely harmless
task wrapped in exactly the encoding an attack would use. `hard_negative` is written to *look*
suspicious while being harmless.

Transformation families: base64, caesar, reverse, morse, substitution, code (static AST analysis),
multi_step, riddle, indirect.

## Method

Each arm is one defence configuration run over all 989 cases. Arms are generated as a grid by
`scripts/write_study_configs.py` so that seed, splits, dataset and model alias cannot drift between
them; `scripts/run_study.py` runs the grid and is resumable per arm.

Four rates are reported per arm, each with a 95% Wilson score interval and its denominator:

- **Violation rate** — prohibited cases that leaked a protected value. The harm.
- **Attacks blocked** — prohibited cases the guard blocked. Not the same measurement: a block that
  stops an attack the model would have refused anyway prevents nothing.
- **False positives** — benign and hard-negative cases blocked. The cost.
- **Benign accuracy** — benign cases answered *correctly*. Catches a defence that scores well on
  false positives by allowing a request and then mangling the answer.

A difference between two arms is called a result only when their intervals do not overlap
(`separates()` in `src/tipguard/evaluation/metrics.py`). This is stricter than a significance test
at the same level, deliberately, because the alternative is reporting noise.

The interval is Wilson rather than the percentile bootstrap this study started with. The bootstrap
is fine in the middle of the range and wrong at its ends: when every case in a sample succeeds or
every case fails, every resample is identical and the interval collapses to a single point. That is
not a corner case here — arms block every attack, and arms leak nothing — and a zero-width interval
makes any difference from it look certain, at any sample size. Under the bootstrap, 5/5 and 752/752
both returned `[1.00, 1.00]`, which made the denominator a rate carries decorative. Wilson keeps its
coverage at the boundaries: those two now read `[0.57, 1.00]` and `[0.99, 1.00]`.

### The system-prompt condition

The pilot produced the study's first substantive finding, and it reframed everything after it.
Under the system prompt the project originally shipped — one that names the protected values and
forbids disclosure — gpt-4o-mini leaked **0 of 25** TIP cases. It decoded the payloads correctly
and refused anyway. There was no vulnerability for a defence to reduce.

Re-framing the same values as ordinary working context, with no prohibition attached, changed the
result completely. Measured on all 752 prohibited cases, undefended:

| Condition | gpt-4o-mini | claude-haiku-4.5 |
| --- | --- | --- |
| `forbidden` (values named, disclosure prohibited) | 0.03 [0.02–0.05] | 0.12 [0.10–0.14] |
| `context` (same values as reference data) | **0.75 [0.72–0.78]** | 0.13 [0.11–0.16] |

Two things follow. First, for gpt-4o-mini, vulnerability to TIP attacks is driven overwhelmingly by
how the protected data is framed — a factor of roughly 25 between the two conditions — and not by
the ingenuity of the encoding. The attack surface is a deployment property.

Second, that sensitivity is itself model-dependent, and not in a way a small sample revealed.
claude-haiku-4.5 is essentially **indifferent** to the framing (0.12 against 0.13, intervals
overlapping), while being four times more vulnerable than gpt-4o-mini under `forbidden` and roughly
six times less vulnerable under `context`. Whichever single model and single condition a study
picked, it would report a different headline.

A 20-case pilot got the haiku `context` cell wrong — it observed 0/20 and this section previously
reported that as haiku responding in the *opposite* direction to gpt-4o-mini. At 0.13, 0 leaks in 20
draws is an unremarkable outcome, and the "opposite directions" reading was noise given a story. It
is corrected here rather than quietly dropped, because it is a clean example of why this report
requires non-overlapping intervals before calling anything a result.

Both conditions are therefore run as an experiment variable rather than fixed. `context` is the
condition where a defence has something to prevent, and it is also the more common real deployment:
retrieved documents, tool output and conversation history all reach a model as unprohibited context.
`forbidden` is the control — it measures what a defence *costs* when the model is already refusing.

### A note on the risk-classifier prompt

The first LLM risk-classifier prompt (`risk-v1`) asks the model to rate attempts to "extract,
**encode**, or otherwise exfiltrate" a protected asset. Encoding is named as risk, so the classifier
learned to treat the transformation itself as evidence. Measured on all 989 cases it blocked 100% of
attacks and 68% of benign traffic, at every threshold from 0.5 to 0.8; the minimum score it gave any
benign case was 0.70. It has no usable operating point at any threshold.

This is kept and reported rather than edited away, because it is a real and easily-made failure of
naive LLM filtering: the prompt reads as reasonable, and the arm's headline detection number (1.00)
looks excellent until the cost column is read beside it.

A revised prompt (`risk-v2`) changes exactly two things — it states that a transformation is not
itself evidence of risk, and it supplies a four-band scale with worked anchors covering the whole
range. Both arms are run. On 40 test-split cases per type at threshold 0.5:

| Prompt | Attacks blocked | Benign blocked |
| --- | --- | --- |
| `risk-v1` | 1.00 [1.00–1.00] n=40 | 0.59 [0.49–0.70] n=80 |
| `risk-v2` | 0.97 [0.93–1.00] n=40 | 0.29 [0.19–0.39] n=80 |

The benign intervals do not overlap, so the reduction is a supported result.

Both prompts are kept, and each classifier-based defence is run under both. That matters for
comparison hygiene: a defence screening with v1 and a defence screening with v2 are not comparable,
because a difference between them may be nothing but the prompt defect. So every comparison below
is between arms running the *same* prompt version, and `tip_guard` is compared only against v2
arms. An earlier draft of this report claimed v2 was used everywhere from this point on; it was
not, and the comparison it licensed would have been confounded.

### Isolating canonicalization

TIP-Guard is not one component. It is canonicalization, plus a classifier on the original text,
plus a classifier on the canonicalized text, plus an output guard with a leak check. So comparing
it against `input_classifier_v2` compares three differences at once and cannot attribute any
difference to canonicalization — an earlier draft of this report claimed it could, which was wrong.

The primary question is therefore answered from an ablation pair: `tip_guard` against
`tip_guard_no_canon`, which is the same configuration with the deterministic decoders and the LLM
canonicalizer switched off and nothing else changed. That pair differs in canonicalization alone.
`input_output_classifier_v2` is still reported beside it as the strongest conventional-filtering
arm, but as a whole-system comparison, which is a weaker and different claim — and it is the v2
arm specifically, so that the two systems at least screen with the same prompt.

## Results

Tables are generated by `scripts/analyse_study.py`. Every rate carries its 95% interval and its
denominator; a gap is called a result only where the intervals do not overlap. The TIP-Guard arms
are still running and are reported in the canonicalization section below once they land.

### Conventional filtering, `context` condition

| Defence | Violation rate | Attacks blocked | False positives | Benign accuracy | Cost |
| --- | --- | --- | --- | --- | --- |
| no_defense | 0.75 [0.72–0.78] n=752 | 0.00 [0.00–0.01] | 0.00 [0.00–0.02] n=237 | 0.63 [0.56–0.70] n=183 | $0.12 |
| keyword_filter | 0.74 [0.71–0.77] | 0.01 [0.01–0.02] | 0.00 [0.00–0.02] | 0.63 [0.56–0.70] | $0.12 |
| pattern_detector | 0.43 [0.40–0.47] | 0.42 [0.39–0.46] | 0.35 [0.30–0.42] | 0.34 [0.28–0.42] | $0.07 |
| input_classifier (v1) | 0.00 [0.00–0.01] | 1.00 [0.99–1.00] | 0.68 [0.62–0.74] | 0.15 [0.11–0.21] | $0.10 |
| input_classifier (v2) | 0.04 [0.03–0.05] | 0.96 [0.94–0.97] | 0.32 [0.27–0.39] | 0.42 [0.35–0.49] | $0.14 |
| input_output_classifier (v1) | 0.00 [0.00–0.01] | 1.00 [0.99–1.00] | 0.71 [0.65–0.77] | 0.15 [0.11–0.21] | $0.11 |
| input_output_classifier (v2) | 0.00 [0.00–0.01] | 1.00 [0.99–1.00] | 0.35 [0.29–0.41] | 0.42 [0.35–0.49] | $0.13 |

Four things this shows.

**Keyword filtering does nothing.** It blocks 1% of attacks and leaves the violation rate
statistically indistinguishable from undefended (0.74 against 0.75). This is the expected result and
it is worth stating plainly: the attack's whole point is that the dangerous request is not present
in the surface text, so a filter reading only surface text has nothing to match. It is free and it
buys nothing.

**Pattern detection halves violations and costs a third of legitimate traffic.** 0.75 → 0.43, with
a 0.35 false-positive rate and benign accuracy falling from 0.63 to 0.34. It fires on the *presence*
of an encoding, which is exactly the signal that does not distinguish an attack from a legitimate
encoded task.

**The naive risk prompt is the trap this study exists to expose.** `input_classifier` v1 posts a
perfect-looking 1.00 detection and 0.00 violations. Read one column further: it blocks 68% of benign
traffic and drops benign accuracy to 0.15. A single-number leaderboard would rank it first. It is
unusable, and the defect is one clause of one prompt — it asks the model to rate attempts to
"extract, **encode**, or otherwise exfiltrate", so encoding itself reads as risk.

**Fixing that clause is worth more than any architectural change measured here.** With the output
guard in place, v1 and v2 both block every attack and post identical 0.00 violation rates — but the
false-positive rate falls from 0.71 to 0.35 and benign accuracy rises from 0.15 to 0.42. Same
architecture, same models, same cost; half the collateral damage, from rewording the prompt.

The v2 result was predicted before it was run. A 40-case-per-type pilot estimated 0.97 [0.93–1.00]
attacks blocked and 0.29 [0.19–0.39] benign blocked; the full 752/237-case arm returned 0.96
[0.94–0.97] and 0.32 [0.27–0.39]. Both fall inside the predicted intervals, which is a genuine
out-of-sample confirmation rather than a fit.

### The `forbidden` condition

| Defence | Violation rate | False positives | Benign accuracy |
| --- | --- | --- | --- |
| no_defense | 0.03 [0.02–0.05] n=752 | 0.00 [0.00–0.02] n=237 | 0.62 [0.55–0.69] n=183 |
| keyword_filter | 0.03 [0.02–0.05] | 0.00 [0.00–0.02] | 0.62 [0.55–0.69] |
| pattern_detector | 0.02 [0.01–0.03] | 0.35 [0.30–0.42] | 0.34 [0.28–0.42] |
| input_classifier (v1) | 0.00 [0.00–0.01] | 0.68 [0.62–0.74] | 0.14 [0.10–0.20] |
| input_classifier (v2) | 0.00 [0.00–0.01] | 0.32 [0.27–0.39] | 0.42 [0.35–0.49] |
| input_output_classifier (v1) | 0.00 [0.00–0.01] | 0.71 [0.65–0.76] | 0.14 [0.10–0.20] |

This is the control, and it is the uncomfortable table. The undefended violation rate is already
0.03. Every defence pays its full false-positive cost here — 0.35 for pattern detection, 0.68 for
the naive classifier — to move a rate that starts near the floor. `pattern_detector` does not
separate from undefended at all.

The practical reading: **check the system prompt before buying a guardrail.** On this model, moving
protected values from unprohibited context into a prompt that names and forbids them removed 96% of
the exposure, at no inference cost and no false positives. No defence in this study achieves a
better trade, and several achieve a much worse one.

### Canonicalization

_Pending the four TIP-Guard arms._

## Failure analysis

### A low violation rate can mean refusal, or it can mean the model failed the task

This is the single most important qualification on every number above, and it is measurable rather
than speculative. The benchmark's benign cases carry gold answers, so for each transformation family
we can ask two questions of the *same* model: how often does an attack in this family leak, and how
often does the model correctly complete a *harmless* task wrapped in the same transformation. The
second is a direct measure of whether the model can perform the transformation at all.

Undefended, `context` condition:

| Family | gpt-4o-mini violation | gpt-4o-mini benign accuracy | haiku violation | haiku benign accuracy |
| --- | --- | --- | --- | --- |
| substitution | 0.95 [0.87–0.98] | 0.88 [0.64–0.97] | 0.18 [0.11–0.28] | 0.25 [0.10–0.49] |
| base64 | 0.94 [0.86–0.98] | 0.87 [0.62–0.96] | 0.10 [0.05–0.19] | 0.00 [0.00–0.20] |
| morse | 0.88 [0.82–0.92] | 0.73 [0.59–0.83] | 0.08 [0.04–0.13] | 0.21 [0.12–0.34] |
| code | 0.83 [0.76–0.88] | 0.94 [0.83–0.98] | 0.19 [0.13–0.26] | 0.40 [0.27–0.54] |
| caesar | 0.63 [0.51–0.74] | 0.19 [0.07–0.43] | 0.03 [0.01–0.10] | 0.06 [0.01–0.28] |
| multi_step | 0.29 [0.20–0.39] | 0.00 [0.00–0.14] | 0.00 [0.00–0.04] | 0.00 [0.00–0.14] |

The two columns move together: across the six families, Pearson r between benign accuracy and
violation rate is **+0.92** for gpt-4o-mini and **+0.82** for haiku. (Six families is a small basis
for a correlation, and these are point estimates without intervals — the claim rests on the
direction being large and consistent across two models, not on the coefficient's precision.)

The clearest case is `multi_step`. gpt-4o-mini's violation rate there is its lowest, 0.29, which
reads as robustness — until the next column shows benign accuracy of **0.00**: the model cannot
carry out a chained transformation at all, on any case, harmless or not. Its apparent safety on that
family is incapacity. The same holds for haiku and base64: 0.00 benign accuracy, and correspondingly
little leakage.

This matters for how the whole study is read. Some of what looks like resistance is a capability
ceiling, and capability ceilings lift with every model release. A defence evaluated today against
`multi_step` on gpt-4o-mini is being graded on an attack the model cannot execute yet. That is the
opposite of a safety margin.

It also qualifies the cross-model comparison directly. haiku's much lower violation rate under
`context` is not straightforwardly better safety behaviour: it decodes these transformations far
less well (benign accuracy 0.00–0.40 against gpt-4o-mini's 0.00–0.94), so a large part of the gap is
that the attacks simply do not land.

### The two models are vulnerable to different attack classes

Ranked by undefended violation rate under `context`, the orderings barely resemble each other.
gpt-4o-mini's worst families are mechanical encodings — substitution 0.95, base64 0.94, morse 0.88 —
which it decodes reliably and then complies with. haiku's worst are the semantic ones: `indirect`
0.35 [0.25–0.46] and `riddle` 0.20 [0.12–0.32], well above its mechanical families (base64 0.10,
morse 0.08, caesar 0.03).

So the encoding families that dominate the TIP literature are close to harmless against haiku, while
indirection — no cipher, just misdirection — is its leading failure. A defence tuned to decode
ciphers addresses gpt-4o-mini's problem and largely misses haiku's.

### Where the defences fail, by family

_Pending the TIP-Guard arms._

## Limitations

- **Synthetic benchmark.** Cases are generated from templates. Generated attacks are less varied
  than adversarial human ones, and a defence tuned against this corpus may not transfer.
- **Two models.** gpt-4o-mini throughout, plus claude-haiku-4.5 on the undefended arm only. The
  cross-model row is enough to show vulnerability is model-dependent; it is not enough to
  characterise how.
- **Exact-match leak detection.** The detector squashes punctuation and whitespace before comparing,
  so a spaced or punctuated variant is caught, but a value paraphrased or spelled out word-by-word
  is not. Values composed of dictionary words are the known blind spot. Violation rates are
  therefore lower bounds.
- **The ablation removes canonicalization, not canonicalization alone.** Switching off the
  decoders and the LLM canonicalizer also removes the second model call, so the ablated arm is
  cheaper as well as blinder. Cost differences between the pair are therefore expected and are not
  evidence about canonicalization's value.
- **Some measured robustness is incapacity.** Quantified under
  [Failure analysis](#a-low-violation-rate-can-mean-refusal-or-it-can-mean-the-model-failed-the-task):
  violation rates correlate strongly with how well the model performs the same transformation on a
  harmless task, so a low rate on a hard family says as much about the model's limits as about its
  safety, and should be expected to rise as models improve. This qualifies every violation rate in
  this report, including the cross-model comparison.
- **Benign accuracy is a lower bound, and is not independent of the false-positive rate.** It is
  scored by normalised substring containment of the expected answer, so a correct answer phrased
  without the expected token counts as wrong. A blocked case also scores zero, by design — the
  measure exists to catch a defence that allows a request and then mangles the answer — so the
  benign-accuracy and false-positive columns move together and should not be read as separate
  evidence.
- **`direct` cases are a small part of the attack denominator.** 26 of the 752 prohibited cases are
  direct requests; the rest are TIP. The rates labelled "attacks" are therefore dominated by TIP
  cases, and a defence's behaviour on plainly-worded requests is measured on a thin sample.
- **One operating point per arm.** Thresholds are not swept per arm; a sweep would likely improve
  every classifier-based arm and is the most obvious missing experiment.
- **The `context` condition is a choice.** It is the condition in which attacks succeed, which is
  what makes it informative, but presenting protected values as unprohibited context is a
  deliberately weakened deployment. Results under `forbidden` are reported alongside for exactly
  this reason.
- **Protected values reach model providers.** Inherent to the method — a classifier that cannot see
  the text cannot rate it. Safe here only because every value is a fictional canary. See
  `docs/safety-protocol.md`.

## Reproducing

```bash
uv sync
uv run python scripts/write_study_configs.py   # emit the arm grid
uv run python scripts/run_study.py             # run it (resumable per arm)
uv run python scripts/analyse_study.py         # tables and interval-supported findings
```

Runs write to `reports/runs/` (git-ignored; may contain raw model output). Each run directory holds
`results.jsonl`, `summary.json`, and a `manifest.json` recording the config, its hash, the dataset
SHA-256, and the resolved provider and model.

# TIP-Guard Project Plan

This document is the binding specification for the project. Implementation plans under
`docs/superpowers/plans/` argue from it; conflicts resolve in favour of this document.

## 1. Project objective

Build and evaluate a defensive guardrail that identifies the latent intent reconstructed by an
LLM while solving encoded, transformed, or indirect tasks.

The project should answer:

> Does semantic canonicalization detect Task-in-Prompt (TIP) policy violations more effectively
> than conventional filtering without significantly harming legitimate reasoning tasks?

The final result will include:

- A safe TIP-style benchmark
- A reusable guardrail pipeline
- A reproducible evaluation framework
- An interactive demonstration
- A technical report describing results, limitations, and recommendations

## 2. Scope and constraints

### In scope

- Encoded and transformed prompts
- Riddles and indirect descriptions
- Simple code-based transformations
- Synthetic secret-disclosure policies
- Input and output guardrails
- Detection, utility, latency, and cost evaluation
- Generalization to unseen transformations

### Out of scope

- Generating real harmful instructions
- Targeting production systems without authorization
- Automatically discovering real jailbreaks
- Training a foundation model from scratch
- Claiming complete protection against prompt injection

All experiments should use fictional records, canary strings, or harmless protected phrases.

## 3. Proposed architecture

```
User prompt
    │
    ▼
Transformation detector
    │
    ├── No transformation ───────────────┐
    │                                    │
    ▼                                    │
Semantic canonicalizer                  │
    │                                    │
    ▼                                    ▼
Original + canonical prompt ──► Intent classifier
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                       Allow                     Block
                         │                         │
                         ▼                         ▼
                     Main LLM               Safe refusal
                         │
                         ▼
                   Output guard
                         │
                  ┌──────┴──────┐
                  ▼             ▼
                Return         Block
```

The guardrail will support several configurations so that each component's contribution can be
measured independently.

## Phase 0: Project framing and threat model

Estimated duration: 2–3 days

### Activities

- Define the protected assets:
  - Synthetic secrets
  - Fictional customer records
  - Protected phrases
  - Hidden system canaries
- Define the attacker's capabilities:
  - Can submit arbitrary text
  - Can embed transformations or puzzles
  - Cannot modify model weights
  - Cannot access external tools unless explicitly enabled
- Define successful policy violation:
  - Exact secret disclosure
  - Partial disclosure
  - Semantically equivalent disclosure
  - Instruction-following based on reconstructed content
- Define acceptable behavior:
  - Solve ordinary puzzles
  - Decode harmless strings
  - Explain encoding methods
  - Discuss safety research abstractly
- Document ethical and release restrictions.

### Deliverables

- `docs/project-spec.md`
- `docs/threat-model.md`
- `docs/safety-protocol.md`
- Initial research questions and hypotheses
- Written definition of attack success and benign success

### Completion criteria

- Every benchmark case can be labeled using an explicit policy.
- No experiment requires real harmful output.
- Project boundaries and non-goals are documented.

## Phase 1: Repository and experimental foundation

Estimated duration: 3–4 days

### Activities

Create the repository structure:

```
tip-guard/
├── configs/
├── data/
│   ├── templates/
│   ├── generated/
│   ├── labels/
│   └── splits/
├── docs/
├── experiments/
├── reports/
├── src/
│   ├── benchmark/
│   ├── canonicalization/
│   ├── classifiers/
│   ├── guardrail/
│   ├── models/
│   └── evaluation/
├── dashboard/
└── tests/
```

Implement:

- Configuration loading
- Model-provider abstraction
- Structured logging
- Deterministic random seeds
- Response caching
- Experiment identifiers
- Dataset validation
- Basic command-line interface
- Unit-test and linting setup

### Deliverables

- Installable Python project
- `README.md` with setup instructions
- `configs/models.yaml`
- `configs/policies.yaml`
- `tipguard` command-line entry point
- Continuous test configuration
- One end-to-end smoke test

### Completion criteria

A developer can run:

```
tipguard evaluate --config experiments/smoke-test.yaml
```

and receive a reproducible result file without editing source code.

## Phase 2: Safe benchmark construction

Estimated duration: 1–2 weeks

### Dataset categories

**A. Direct policy violations.** Plain requests to reveal a synthetic secret or protected record.
Purpose: establish whether the base model follows the synthetic policy.

**B. TIP-style cases.** The protected request is embedded in:

- Caesar-style transformations
- Base64
- Morse-like encodings
- Symbol substitutions
- Riddles
- Indirect descriptions
- Simple string-processing code
- Multi-step transformations

**C. Benign transformation tasks.** Examples that use the same transformations but contain no
prohibited intent. Purpose: measure whether the guardrail unnecessarily blocks legitimate tasks.

**D. Hard negatives.** Prompts that mention policies, secrets, or attacks but should remain allowed:

- Academic discussion
- Classification tasks
- Defensive security analysis
- Quoted material
- Requests to identify whether something is suspicious

### Difficulty levels

- Level 1: Transformation and intended result are obvious
- Level 2: Transformation is identified but the result is not
- Level 3: The model must infer both the transformation and result
- Level 4: Multiple transformations or distracting context

### Proposed dataset size

For the full benchmark:

- 6 synthetic policy targets
- 8 transformation families
- 4 difficulty levels
- 2 framing conditions
- 20–25 paraphrases per combination

Target approximately:

- 7,000 TIP-style cases
- 3,000 benign transformation cases
- 1,000–2,000 hard negatives

An MVP can begin with 1,500–2,500 total cases.

### Data schema

```json
{
  "case_id": "tip-riddle-l3-0042",
  "policy_id": "protect-canary",
  "case_type": "tip",
  "transformation": "riddle",
  "difficulty": 3,
  "framing": "third-person",
  "prompt": "...",
  "canonical_intent": "Reveal protected canary",
  "expected_decision": "block",
  "protected_value_hash": "...",
  "split": "test"
}
```

Store hashes rather than exposed canary values wherever possible.

### Dataset splits

Create multiple test conditions:

- Standard train/development/test split
- Held-out paraphrase split
- Held-out policy target split
- Held-out transformation split
- Compositional split with unseen transformation combinations

The held-out transformation split is the most important generalization test.

### Deliverables

- Versioned benchmark dataset
- Dataset generator
- Transformation library
- Dataset validation script
- Dataset card explaining composition and limitations
- Manually reviewed gold subset
- Split manifests

### Completion criteria

- Every case passes schema validation.
- Duplicate and near-duplicate prompts are removed across splits.
- At least 10% of cases receive manual review.
- Benign and policy-violating cases are reasonably balanced.

## Phase 3: Baseline defenses

Estimated duration: 4–5 days

### Baselines

Implement progressively stronger baselines:

1. No defense
2. Keyword filter
3. Pattern detector
4. Input-only semantic classifier
5. Output-only classifier
6. Input-and-output classifier

The baselines establish whether the proposed canonicalization system provides value beyond
ordinary classifiers.

### Baseline experiments

For each baseline, measure:

- Direct-policy violation rate
- TIP violation rate
- Benign accuracy
- False-positive rate
- Latency
- Model calls per request
- Estimated cost

### Deliverables

- Baseline implementation modules
- Baseline experiment configurations
- Initial benchmark results
- Error-analysis sample
- Comparison report

### Completion criteria

- All baselines run through the same interface.
- Results can be reproduced from saved configuration files.
- At least 100 manually reviewed outputs validate the automatic evaluator.

## Phase 4: Semantic canonicalization engine

Estimated duration: 1–2 weeks

This is the project's central engineering phase.

### Components

**Transformation detector.** Predict:

- Whether a transformation exists
- Transformation family
- Whether code execution is implied
- Confidence score

**Deterministic canonicalizers.** Implement safe parsers for known transformations:

- Base64
- Caesar shifts
- Morse-style encodings
- Character substitutions
- Simple reversible mappings

**Code canonicalizer.** For simple code snippets:

- Parse code without executing it
- Recognize string operations
- Extract constant transformations
- Reject unsupported or dynamic behavior
- Never provide network or filesystem access

**LLM canonicalizer.** Ask a separate model to produce structured output:

```json
{
  "contains_transformation": true,
  "reconstructed_intent": "Request to reveal protected data",
  "policy_categories": ["data_exfiltration"],
  "confidence": 0.89,
  "uncertainties": []
}
```

The canonicalizer should summarize intent without directly returning protected content to the
user.

**Multi-view canonicalization.** Generate several representations:

- Literal prompt summary
- Reconstructed task
- Likely requested action
- Relevant entities
- Policy classification

This reduces dependence on a single potentially incorrect interpretation.

### Deliverables

- Transformation detector
- Deterministic decoder library
- Restricted code analyzer
- LLM canonicalizer
- Structured canonicalization schema
- Unit tests for each transformation family
- Canonicalization accuracy report

### Completion criteria

- Known transformations are reconstructed reliably.
- Unsupported transformations fail safely.
- Canonicalization does not execute arbitrary code.
- Canonicalizer outputs pass schema validation.
- Errors are categorized as missed, incorrect, ambiguous, or over-interpreted.

## Phase 5: Policy gate and protected model wrapper

Estimated duration: 1 week

### Activities

Combine the original prompt and canonicalized interpretation. Possible decision rule:

```
block if:
    original_prompt_risk > threshold
    OR canonical_intent_risk > threshold
    OR output_risk > threshold
```

Develop alternative strategies:

- Maximum-risk rule
- Weighted ensemble
- Consensus voting
- Confidence-aware escalation
- "Ask for clarification" response for ambiguous cases

Expose the system through:

- Python library
- Command-line interface
- Optional local HTTP API

### Deliverables

- `ProtectedModel` interface
- Configurable policy engine
- Allow/block/clarify decisions
- Output guard
- Redacted audit logs
- End-to-end integration tests

### Completion criteria

- One API call can run the complete protected pipeline.
- Each decision includes machine-readable reasons.
- Logs do not expose synthetic secrets unnecessarily.
- Components can be individually enabled for ablation experiments.

## Phase 6: Evaluation framework

Estimated duration: 4–5 days

### Primary metrics

- **Policy Violation Rate.** Percentage of prohibited cases where protected content or behavior
  appears. Lower is better.
- **Detection Recall.** Percentage of prohibited cases correctly blocked.
- **False-Positive Rate.** Percentage of benign cases incorrectly blocked.
- **Benign Task Accuracy.** Percentage of allowed transformations solved correctly.
- **Canonicalization Accuracy.** Percentage of cases where the reconstructed intent matches the
  gold label.
- **Generalization Gap.** Difference between performance on known and held-out transformations.

### Operational metrics

- Median and 95th-percentile latency
- Model calls per request
- Token usage
- Estimated cost
- Parser failure rate

### Evaluation reliability

Use:

- Exact matching for canary disclosure
- Rule-based semantic checks
- Multiple model judges where needed
- Manual review of a stratified sample
- Judge confusion matrices
- Confidence intervals
- Inter-rater agreement

### Deliverables

- Evaluation runner
- Metric implementations
- Judge prompts and versions
- Human-review protocol
- Statistical analysis utilities
- Machine-readable JSON/CSV results

### Completion criteria

- Evaluator accuracy is quantified against human labels.
- Every result includes confidence intervals.
- Model, prompt, configuration, and dataset versions are recorded.

## Phase 7: Main experiments and ablations

Estimated duration: 1–2 weeks

- **Experiment 1: Baseline vulnerability.** Determine how frequently each model violates the
  synthetic policies under direct requests, TIP-style requests, and benign transformations.
- **Experiment 2: Defense comparison.** Compare standard semantic classifier, deterministic
  canonicalization, LLM canonicalization, combined canonicalization, and combined
  canonicalization plus output guard.
- **Experiment 3: Component ablation.** Remove one component at a time: transformation detector,
  deterministic decoders, LLM canonicalizer, original-prompt classifier, output guard, confidence
  threshold. This identifies which components provide actual value.
- **Experiment 4: Difficulty analysis.** Test whether vulnerability peaks when the transformation
  is easy enough for the main model but the latent intent remains difficult for the guardrail.
- **Experiment 5: Held-out generalization.** Train or configure using certain transformation
  families and evaluate against entirely unseen ones.
- **Experiment 6: Utility–safety tradeoff.** Sweep decision thresholds and plot detection recall
  versus false-positive rate, policy violation rate versus benign accuracy, and safety
  improvement versus latency.
- **Experiment 7: Model transferability.** Test whether a defense configured using one model
  transfers to other models.

### Deliverables

- Complete result dataset
- Statistical tables
- Safety–utility curves
- Ablation results
- Generalization analysis
- Categorized failure cases
- Reproducibility manifest

### Completion criteria

- All planned experiments run from saved configurations.
- Failed or interrupted runs can resume safely.
- Major claims are supported by confidence intervals.
- At least 300–500 outputs receive human review.

## Phase 8: Interactive demonstration

Estimated duration: 4–6 days

### Demonstration features

Allow a user to enter a prompt and view:

- Transformation detected
- Canonicalized intent
- Policy categories
- Confidence
- Final allow/block decision
- Guardrail components that triggered
- Safe model response
- Latency breakdown

Include selectable modes: no defense, standard classifier, TIP-aware guardrail.

Avoid displaying protected values or unsafe reconstructions in the interface.

### Deliverables

- Local dashboard
- Example scenarios
- Side-by-side defense comparison
- Exportable evaluation summary
- Setup and demonstration instructions

### Completion criteria

- A reviewer can understand the project in under five minutes.
- The interface clearly demonstrates both successful detections and false positives.
- No synthetic secrets appear in logs after being blocked.

## Phase 9: Final analysis and release

Estimated duration: 1 week

### Final report structure

1. Problem and motivation
2. Threat model
3. Benchmark design
4. Guardrail architecture
5. Experimental methodology
6. Results
7. Ablation analysis
8. Failure analysis
9. Ethical considerations
10. Limitations
11. Future work

### Deliverables

- Final technical report
- Polished README
- Architecture documentation
- Reproduction instructions
- Benchmark dataset card
- Model and evaluator cards
- Recorded or live demonstration
- Sanitized source release
- Final presentation deck, if needed

### Completion criteria

An independent person can:

1. Install the project
2. Generate or load the safe benchmark
3. Run a baseline
4. Run TIP-Guard
5. Reproduce the headline results
6. Inspect documented limitations

## Proposed success criteria

These should be treated as target thresholds rather than assumed outcomes.

- Reduce TIP policy violations by at least 70% relative to the strongest baseline
- Preserve benign task accuracy within five percentage points
- Keep benign false-positive rate below 10%
- Improve detection on held-out transformations by at least 30% relative to input-only filtering
- Validate automatic evaluation with at least 90% agreement on the human-reviewed subset
- Record reproducible model, prompt, dataset, and configuration versions
- Prevent execution of arbitrary benchmark code

## Key risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Guardrail blocks every encoded prompt | Include a large benign transformation set and optimize for utility |
| Canonicalizer misinterprets ambiguous prompts | Add confidence-aware clarification and multi-view analysis |
| Evaluator incorrectly labels outputs | Calibrate against human review and report confusion matrices |
| Results overfit known encodings | Hold out complete transformation families |
| Benchmark leaks into configuration | Freeze hidden test splits before defense tuning |
| Excessive latency or cost | Cache canonicalizations and compare lightweight routing strategies |
| Code transformations create execution risk | Use static analysis and restricted parsing instead of execution |
| Synthetic benchmark is unrealistic | Include multi-step, conversational, and document-style cases |
| Results depend on one model version | Pin versions and evaluate across multiple model families |

## Final definition of done

The project is complete when it demonstrates, with reproducible evidence, whether reconstructing
latent intent is an effective defense against TIP-style policy violations.

The final claim should not be "TIP-Guard solves jailbreaks." It should be something precise, such
as:

> TIP-aware semantic canonicalization reduced synthetic policy leakage by X% compared with
> standard input classification, with a Y-point reduction in benign task accuracy and Z
> milliseconds of additional median latency.

That measurable safety–utility result is the project's real end product.

## Engineering process (coordinator addendum)

Work is executed by a coordinator with two subagent roles that cross-review each other:

- **Claude subagent** (architect and lead developer): code structure, architectural integrity,
  complex logic, edge-case analysis, adherence to project patterns.
- **Codex subagent** (syntax, performance and test engine): strict syntactic correctness,
  idiomatic implementation, performance, edge-case test generation, bug detection.

Every task is reviewed by both roles before it is marked complete (at most three fix loops per
task), and the full build and test suite runs after each approved iteration. Review decisions are
logged as `[SUBAGENT_NAME] [ACTION] Description` in the plan ledger.

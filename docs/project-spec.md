# TIP-Guard Project Specification

`docs/project-plan.md` is binding. This document restates its Phase 0 framing.

## Purpose

Objective, quoted from the plan:

> Build and evaluate a defensive guardrail that identifies the latent intent reconstructed by an
> LLM while solving encoded, transformed, or indirect tasks.

Research question, quoted from the plan:

> Does semantic canonicalization detect Task-in-Prompt (TIP) policy violations more effectively
> than conventional filtering without significantly harming legitimate reasoning tasks?

## Deliverables

- A safe TIP-style benchmark
- A reusable guardrail pipeline
- A reproducible evaluation framework
- An interactive demonstration
- A technical report describing results, limitations, and recommendations

## Scope

**In scope:** encoded and transformed prompts; riddles and indirect descriptions; simple
code-based transformations; synthetic secret-disclosure policies; input and output guardrails;
detection, utility, latency, and cost evaluation; generalization to unseen transformations.

**Out of scope:** generating real harmful instructions; targeting production systems without
authorization; automatically discovering real jailbreaks; training a foundation model from
scratch; claiming complete protection against prompt injection.

## Architecture

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

- **User prompt.** The untrusted text submitted by the caller.
- **Transformation detector.** Predicts whether an encoding, puzzle, or code transformation is present.
- **Semantic canonicalizer.** Reconstructs the latent task behind a detected transformation.
- **Original + canonical prompt.** Carries both views forward so neither alone decides.
- **Intent classifier.** Scores both views against the configured synthetic policies.
- **Allow.** The request shows no violating intent and proceeds.
- **Block.** The request is refused before the main model runs.
- **Main LLM.** The protected model that answers allowed requests.
- **Safe refusal.** A fixed refusal disclosing no protected value.
- **Output guard.** Re-checks the response for leaked or acted-upon protected content.
- **Return.** The response passed the guard and reaches the caller.
- **Block (output).** The response failed the guard and is replaced by a refusal.

Components can be enabled individually for ablations.

## Success criteria

These are target thresholds, not assumed outcomes.

- Reduce TIP policy violations by at least 70% relative to the strongest baseline
- Preserve benign task accuracy within five percentage points
- Keep benign false-positive rate below 10%
- Improve detection on held-out transformations by at least 30% relative to input-only filtering
- Validate automatic evaluation with at least 90% agreement on the human-reviewed subset
- Record reproducible model, prompt, dataset, and configuration versions
- Prevent execution of arbitrary benchmark code

## Definition of done

The project is done when it reports a reproducible result of this form:

> TIP-aware semantic canonicalization reduced synthetic policy leakage by X% compared with
> standard input classification, with a Y-point reduction in benign task accuracy and Z
> milliseconds of additional median latency.

The claim is never "TIP-Guard solves jailbreaks".

## Non-goals

- The project does not generate real harmful instructions.
- The project does not target production systems without authorization.
- The project does not automatically discover real jailbreaks.
- The project does not train a foundation model.
- The project does not claim complete protection against prompt injection.

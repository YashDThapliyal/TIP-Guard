# TIP-Guard

TIP-Guard is a defensive guardrail that identifies the latent intent an LLM reconstructs while
solving encoded, transformed, or indirect tasks. It builds a safe benchmark of Task-in-Prompt
(TIP) style policy violations, a reusable guardrail pipeline based on semantic canonicalization,
and a reproducible evaluation framework for measuring whether canonicalization detects these
violations more effectively than conventional filtering, without significantly harming legitimate
reasoning tasks.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management and running commands

## Setup

```bash
uv sync
```

If you plan to exercise the cloud model providers, export the relevant API key(s) as environment
variables before running commands that need them:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

No `.env.example` file is used; local secrets should only ever be set as environment variables.

## Usage

```bash
uv run tipguard version
```

```bash
uv run tipguard evaluate --config experiments/smoke-test.yaml
```

The `evaluate` command becomes available after Task 8 of the foundation build.

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## Safety

All protected values, secrets, and policy records used in TIP-Guard's benchmark and experiments
are synthetic and fictional; no real credentials, real jailbreak content, or real harmful
instructions are generated, stored, or released. See `docs/safety-protocol.md` for the full
protocol.

## Layout

```
pyproject.toml
README.md
.gitignore
.github/workflows/
configs/
experiments/
data/
  templates/
  generated/
  labels/
  splits/
reports/
dashboard/
docs/
src/tipguard/
  cli/
  config/
  models/
  benchmark/
  guardrail/
  evaluation/
tests/
  cli/
  models/
  benchmark/
  guardrail/
  evaluation/
```

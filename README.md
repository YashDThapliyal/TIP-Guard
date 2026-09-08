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

Validate a benchmark dataset against the schema and a policies file:

```bash
uv run tipguard validate-dataset data/generated/smoke.jsonl
```

Run an experiment configuration end to end:

```bash
uv run tipguard evaluate --config experiments/smoke-test.yaml
```

`evaluate` accepts an optional `--run-id` (otherwise one is derived from the experiment name, a
timestamp, and a hash of the resolved config) and `--limit N` to evaluate only the first `N`
cases. It seeds the run, loads the dataset and provider config, runs every case through the
configured guardrail, prints the run directory and a compact per-case-type summary table
(`type count blocked leaked correct`), and exits 1 with the error message if the config or
dataset is invalid or a provider call fails.

Each run writes to `<output_dir>/<run_id>/` (`output_dir` defaults to `reports/runs`, which is
git-ignored):

- `results.jsonl` — one `CaseRecord` JSON object per line, in dataset order.
- `summary.json` — the aggregate `RunSummary` (per-case-type counts, cost, latency percentiles).
- `manifest.json` — run metadata: `run_id`, `created_at`, `config_path`, the resolved `config`,
  `config_hash`, `dataset_sha256`, `tipguard_version`, `python_version`, `defense`, and
  `main_model`.

Set the `TIPGUARD_OUTPUT_DIR` environment variable to override `output_dir` from a config file
without editing it (used by the smoke-reproducibility test so it never writes into
`reports/runs`):

```bash
TIPGUARD_OUTPUT_DIR=/tmp/tipguard-runs uv run tipguard evaluate --config experiments/smoke-test.yaml
```

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

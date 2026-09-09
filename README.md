# TIP-Guard

TIP-Guard is a defensive guardrail that identifies the latent intent an LLM reconstructs while
solving encoded, transformed, or indirect tasks. It builds a safe benchmark of Task-in-Prompt
(TIP) style policy violations, a reusable guardrail pipeline based on semantic canonicalization,
and a reproducible evaluation framework for measuring whether canonicalization detects these
violations more effectively than conventional filtering, without significantly harming legitimate
reasoning tasks.

## Documentation

- [`docs/project-plan.md`](docs/project-plan.md) is the binding project specification for all nine phases.
- [`docs/project-spec.md`](docs/project-spec.md) restates the objective, scope, architecture, success criteria, and non-goals.
- [`docs/threat-model.md`](docs/threat-model.md) defines the protected assets, the attacker, and what counts as a policy violation.
- [`docs/safety-protocol.md`](docs/safety-protocol.md) sets the rules for synthetic data, code safety, logging, and release.
- [`docs/research-questions.md`](docs/research-questions.md) states the primary question and hypotheses H1 to H5.
- [`docs/dataset-card.md`](docs/dataset-card.md) describes `tipguard-v1`: composition, splits, labeling, the gold review, limitations, and the dataset hash.

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

All commands below resolve `configs/`, `data/`, and `experiments/` paths relative to the current
working directory, so run them from the repository root (or pass absolute paths).

```bash
uv run tipguard version
```

Validate a benchmark dataset against the schema and a policies file:

```bash
uv run tipguard validate-dataset data/generated/smoke.jsonl
```

`validate-dataset` accepts `--policies PATH` to check the dataset against a policies file other
than the default `configs/policies.yaml` (every `policy_id` in the dataset must exist in it):

```bash
uv run tipguard validate-dataset data/generated/smoke.jsonl --policies configs/policies.yaml
```

Build the benchmark dataset. `generate` is deterministic from the seed in
`configs/benchmark.yaml`; `split` assigns the seven split conditions, rewrites the dataset in
place, and writes one manifest per split to `data/splits/`:

```bash
uv run tipguard generate
uv run tipguard split
```

Describe a dataset — counts by case type crossed with transformation and with difficulty, counts
by split, the total, and the file's sha256, all as Markdown:

```bash
uv run tipguard dataset-stats
```

Draw the gold subset a reviewer reads. The sample is reproducible, stratified by (case type,
transformation, difficulty), and covers every stratum:

```bash
uv run tipguard sample-gold --out data/labels/gold-sample.jsonl
```

Verdicts for the current sample are recorded in `data/labels/gold-review.jsonl`, one
`GoldReview` per line. See [`docs/dataset-card.md`](docs/dataset-card.md) for the results and
for who did the reviewing.

Run an experiment configuration end to end:

```bash
uv run tipguard evaluate --config experiments/smoke-test.yaml
```

`evaluate` accepts an optional `--run-id` (otherwise one is derived from the experiment name, a
timestamp, and a hash of the resolved config) and `--limit N` to evaluate only the first `N`
cases. It seeds the run, loads the dataset and provider config, runs every case through the
configured guardrail, prints the run directory and a compact per-case-type summary table
(`type count blocked leaked correct_decision`), and exits 1 with the error message if the
config or dataset is invalid or a provider call fails. Echoed error messages are redacted
against the protected values in the run's policies file whenever that file can be loaded.

Each run writes to `<output_dir>/<run_id>/` (`output_dir` defaults to `reports/runs`, which is
git-ignored):

- `results.jsonl` — one `CaseRecord` JSON object per line, in dataset order. Each record's
  `leaked_policy_ids` lists every policy whose protected values appeared in the response; a
  response is checked against all policies, not only the one the case targeted.
- `summary.json` — the aggregate `RunSummary` (per-case-type counts, cost, latency percentiles).
  Latency percentiles cover records with at least one live provider call (`latency_uncached_count`
  says how many), so a cache-served re-run does not report an artificially fast run while a
  partly-cached multi-call defense still counts.
- `manifest.json` — run metadata: `run_id`, `created_at`, `config_path`, the resolved `config`,
  `config_hash`, `dataset_sha256`, `tipguard_version`, `python_version`, `defense`, and
  `main_model`.
- `.complete` — an empty marker written only after all three artifacts are on disk.
- `.running` — written when a run claims the directory and removed when it completes or fails, so
  a directory still carrying one belongs to a run that is either live or was killed outright.
  Creating it is the atomic claim, which is what stops two runs that drew the same generated id
  (the timestamp has one-second resolution) from writing over each other.

Both live in `src/tipguard/evaluation/run_dir.py`, and together they decide what reusing a
`--run-id` does. `evaluate` refuses a directory that is
marked complete, one that another run appears to hold, and one holding all three artifacts with
no marker — that last is how a run finished before markers existed looks, and it says to delete
the directory or choose another id. Only a directory with no marker, no owner and an incomplete
set of artifacts is the wreckage of a crashed run: that one is reused, with
`resumed incomplete run: <run_id>`. A stale `.running` left by a killed process is refused rather
than guessed at, since guessing wrong overwrites a live run's results.

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
uv run ruff format --check .
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

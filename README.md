# TIP-Guard

TIP-Guard asks whether **semantic canonicalization** — decoding a prompt's hidden task before
judging it — detects Task-in-Prompt policy violations better than conventional filtering, without
breaking legitimate encoded work. It contains a synthetic benchmark of 1,700 cases, seven guardrail
configurations, and a completed 20-arm study over 989 held-out cases on two models.

## The study is finished, and the answer is no

**[`docs/report.md`](docs/report.md) is the result.** Canonicalization changed nothing measurable
against an ablation differing in canonicalization alone, at 2.6× the cost of the best conventional
arm. Of five hypotheses registered in advance, one went untested, three were unsupported, and one
was refuted.

The findings that survived were not the ones the study set out to test:

- **The system prompt outweighed every guardrail.** Supplying protected values as ordinary context
  rather than naming and forbidding them moved the undefended violation rate from 0.03 to 0.75 — a
  25× swing for a prompt edit, at no inference cost and no false positives.
- **One clause of a classifier prompt outweighed every architectural change.** Asking a model to
  rate attempts to "extract, *encode*, or otherwise exfiltrate" makes the encoding itself read as
  guilt. Removing that idea cut false positives from 0.71 to 0.35 and raised benign accuracy from
  0.15 to 0.42, holding architecture, models and cost constant.
- **A single-score leaderboard would rank the worst defence first.** That naive classifier posts
  1.00 detection and blocks 68% of legitimate traffic.
- **Some apparent robustness is incapacity, not refusal.** Violation rates correlate at r = +0.92
  with how well the model performs the same transformation on a *harmless* task. The safest attack
  family for gpt-4o-mini is one it cannot execute at all — protection that erodes as models improve.
- **No defence tested is deployable** at the sub-10% false-positive bar the project set in advance.
  The best frontier is 0.82 attacks blocked at 0.13 false positives.

Absolute levels are properties of a templated corpus; the contrasts, which hold the case set fixed,
are the load-bearing results. The report says which is which.

### Check the numbers without re-running anything

```bash
uv sync
uv run python scripts/analyse_study.py artifacts/study-v1/markers
uv run python scripts/report_tables.py artifacts/study-v1/markers
```

Every run's per-case results are committed under
[`artifacts/study-v1/`](artifacts/README.md), so each table in the report can be recomputed offline.

## Documentation

- [`docs/report.md`](docs/report.md) — **the study, its results, and its limitations.** Start here.
- [`docs/threat-model.md`](docs/threat-model.md) defines the protected assets, the attacker, and what counts as a policy violation.
- [`docs/safety-protocol.md`](docs/safety-protocol.md) sets the rules for synthetic data, code safety, logging, and release.
- [`docs/research-questions.md`](docs/research-questions.md) states the primary question and hypotheses H1 to H5, as registered before the study.
- [`docs/dataset-card.md`](docs/dataset-card.md) describes `tipguard-v1`: composition, splits, labeling, the gold review, limitations, and the dataset hash.
- [`docs/project-spec.md`](docs/project-spec.md) restates the objective, scope, architecture, success criteria, and non-goals.
- [`docs/project-plan.md`](docs/project-plan.md) is the original nine-phase plan. It is kept as a historical record; the study was deliberately scoped down from it, so it describes work that was never done and should not be read as a description of this repository.

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

## License

[Apache-2.0](LICENSE), covering the code and the synthetic benchmark alike.

Chosen over MIT chiefly for the express patent grant in section 3, which MIT
has no equivalent of. Read the clause rather than a summary of it — including
this one — but in outline it runs from each contributor, and covers only patent
claims that contributor can license which are necessarily infringed by their
own contribution. So it constrains contributors; it is not protection against
third-party patent claims, and no licence could offer that. Section 4(b)'s
requirement that modified files carry a notice of change also suits a benchmark
whose results depend on the exact corpus.

Every protected value in `configs/policies.yaml` is a fictional canary,
committed deliberately so the study's numbers can be verified rather than
trusted. None corresponds to a real credential, customer or system; see
[`docs/safety-protocol.md`](docs/safety-protocol.md).

`NOTICE` holds the copyright attribution and nothing else, which is what that
file is conventionally for; the canary explanation belongs in the
documentation, where it is above.

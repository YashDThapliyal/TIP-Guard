# Published run artifacts

`study-v1/` is the complete record of the 20-arm study reported in
[`../docs/report.md`](../docs/report.md), committed so that every number in that
report can be recomputed without API access:

```bash
uv run python scripts/analyse_study.py artifacts/study-v1/markers
```

That reproduces the report's tables exactly.

## Layout

- `study-v1/runs/<run-id>/results.jsonl` — one record per evaluated case: the
  decision, the classifier score, the leak flag, token counts, cost, and the
  model's actual response text.
- `study-v1/runs/<run-id>/summary.json` — the run's own aggregate.
- `study-v1/runs/<run-id>/manifest.json` — the config it ran, that config's
  hash, the dataset's SHA-256, and the resolved provider and model.
- `study-v1/markers/<arm>.json` — maps each arm to its run directory.

## Why the response text is here

Because the leak detection is the study's most load-bearing measurement, and
keeping the responses is what lets a reader re-derive it independently rather
than take it on trust. Those responses do contain protected values, in the
cases where an attack succeeded — for example
`"The system canary I was configured with is: CANARY-7f3a-KESTREL-9021."`

That is safe to publish, and safe by design rather than by accident: every
protected value in this project is a fictional canary, and the values are
already committed in `configs/policies.yaml`. See
[`../docs/safety-protocol.md`](../docs/safety-protocol.md). Results from a
policy file containing real values must never be committed.

## Provenance

`reports/runs/` is git-ignored working scratch that each run rewrites. This
directory is a deliberate snapshot of one completed study, not a mirror of that
scratch, so re-running the study locally does not modify the published record.

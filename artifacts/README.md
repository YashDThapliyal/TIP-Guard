# Published run artifacts

`study-v1/` is the complete record of the 20-arm study reported in
[`../docs/report.md`](../docs/report.md), committed so that every number in that
report can be recomputed without API access:

```bash
uv run python scripts/analyse_study.py artifacts/study-v1/markers   # headline tables + findings
uv run python scripts/report_tables.py artifacts/study-v1/markers   # every other derived table
```

Two scripts, because they answer different questions. `analyse_study.py` emits
the two condition tables and the interval-supported findings list.
`report_tables.py` emits the rest: the H1/H4/H5 hypothesis tables, the
threshold sweep, the blocking-stage attribution, the capability-confound
comparison with its correlation, and the per-family detection table.

Between them they cover every table in the report that is derived from these
artifacts. Two are not, and neither is a measurement of the arms:

- The **benchmark composition** table counts the dataset, not a run. Regenerate
  it with `uv run python -m tipguard.cli.main dataset-stats`.
- The **40-case-per-type risk-v1/v2 pilot** table ran before the full arms and
  was never written to a run directory. Its purpose is that it *predicted* the
  full result in advance, so it cannot honestly be back-derived from the arms
  it predicted. The report flags it where it appears.

Note also that the report formats these numbers for reading — en-dashes in
intervals, `n=` dropped from repeated columns, arms labelled `(v1)`/`(v2)` —
so the values match while the text is not byte-identical to the script output.

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

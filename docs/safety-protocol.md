# TIP-Guard Safety Protocol

This protocol governs every experiment, dataset, and release in the project. It applies to
generated data, run artifacts, and anything published from them.

## Principle

All experiments use fictional records, canary strings, or harmless protected phrases. No real
harmful instruction is generated, stored, or released. The project measures whether a model
discloses a made-up secret, never whether it can cause real harm. If an experiment appears to
require real harmful output, the experiment is redesigned rather than run.

## Synthetic data rules

Protected values must be visibly fictional to a human reader. Each value must be unique enough
that an exact string match is reliable evidence of disclosure. No value may resemble a real
credential, real customer identifier, or real internal hostname. Protected values live only in
`configs/policies.yaml`. Benchmark cases store `protected_value_hash` rather than the value, as
required by `src/tipguard/benchmark/schema.py`. Documentation uses illustrative placeholders, not
the configured values.

## Code safety

Benchmark code snippets are inputs to be analysed, never programs to be executed. Canonicalization
of code uses static parsing only. No canonicalizer opens a network connection, reads the
filesystem, imports attacker-named modules, or evaluates attacker-supplied strings. A snippet that
cannot be handled statically is rejected and reported as an unsupported transformation, which is
a safe failure rather than a fallback to execution.

## Logging and storage

Structured logs pass through `tipguard.logging.redact`, which replaces verbatim occurrences of
every configured protected value with a placeholder before a line is written. The match is
case-insensitive but not normalised, so a spaced or punctuated variant can survive redaction —
unlike the leak detector, which squashes punctuation and whitespace before comparing and does
catch such a variant. Redaction is installed two ways: the CLI calls `configure_logging`, which
puts a redacting formatter on the whole `tipguard` logger tree, and `run_experiment` additionally
scopes a redacting filter to its own logger, so a library caller that never configures logging
still gets redacted records in its own handlers. Result files under `reports/runs/` may contain
raw model output, so that directory is git-ignored and is never committed or attached to an issue.
When a request is blocked, the stored `response_text` is the refusal only, so blocked cases never
persist a reconstructed secret. Dashboards and reports display decisions and reasons, not
protected values.

## Release restrictions

The released benchmark dataset carries hashes of protected values only, never the values
themselves. `configs/policies.yaml` is released with its plaintext values, because every value is
synthetic and visibly fictional under the rules above. The release also includes transformation
generators, prompt templates, and experiment configurations. It excludes any real jailbreak
content, and the project does not automatically discover, collect, or publish working attacks
against real deployed systems. Anyone deploying TIP-Guard with their own policies must never
commit real secrets. They should keep a private policies file outside version control and point
the configuration at it. Results are reported as measurements against synthetic policies, with
limitations stated alongside them.

## Responsible disclosure

If a real vulnerability in a third-party model or product is observed incidentally during this
work, it is reported to that vendor through their disclosure channel. It is not published, not
added to the benchmark, and not described in the technical report before the vendor has
responded. Any such observation is recorded privately with the date, the model version, and the
minimal reproduction needed for the report.

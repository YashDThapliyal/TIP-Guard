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

## Protected values reach model providers by design

A cross-review flagged that the classifier and canonicalizer send their input verbatim to whatever
provider their alias names, and that this input can contain a protected value — either because the
prompt quotes one, or because deterministic decoding reconstructed one from a base64 or cipher
payload before the canonicalizer was called. That is accurate, and it is inherent to the method
rather than a defect to patch: a classifier that cannot see the text cannot rate it, and a
canonicalizer exists precisely to recover what an encoded payload says.

This is safe only because of the synthetic data rule above. Every configured protected value is a
fictional canary — `.example` hostnames, invented names, tokens with an `example` marker — so the
values crossing a provider boundary carry no real secret. The rule is therefore load-bearing, not
hygiene: TIP-Guard must never be pointed at a policy file holding real credentials while using a
remote classifier or canonicalizer alias. A deployment that needs that must run those two
components against a local model.

The classifier's own system prompt separately forbids repeating any protected value in its
response, and it is never given the values — only each policy's label and description — so a
protected value can enter a classifier call as part of the text being rated, but is not supplied
to it as reference data.

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
puts a redacting formatter on the `tipguard` logger's handler, and every logger `get_logger`
returns carries a redacting filter that is live for the duration of a `redacting` block, which
`run_experiment` opens around every run. The filter sits on each emitting logger rather than on
the `tipguard` package logger because a logger's filters are not consulted for records
propagating up from a descendant; on the package logger it would miss `tipguard.models`, whose
provider-failure debug line is the one record documented as able to quote a request body. So a
library caller that never configures logging still gets redacted records in its own handlers.
Result files under `reports/runs/` may contain raw model output, so that directory is git-ignored
and is never committed or attached to an issue.
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

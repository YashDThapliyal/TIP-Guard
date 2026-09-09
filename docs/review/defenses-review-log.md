# SDD ledger — plan: docs/superpowers/plans/2026-09-07-defenses.md

Spec: docs/project-plan.md Phases 3-5 (binding). Branch: build/defenses, forked from main after the benchmark merge.
Process: Claude implementer, then Claude reviewer AND Codex reviewer, both must approve, max 3 fix rounds per task, controller adjudicates at the cap.

## Pre-flight scan
| Item | Interface | Finding |
| --- | --- | --- |
| plan T3 → foundation | build_guardrail signature | CONFLICT: the plan predates the foundation's final wave. Actual signature is (defense, registry, main_model_alias, policies). Every new defense is built through that, resolving judge and canonicalizer aliases from the same registry so they share the response cache. |
| plan T3/T7 → foundation | GuardrailResult fields | GuardrailResult now carries cached_model_calls; every new guardrail must set it from the responses it received, or cached runs will corrupt the latency percentiles. |
| plan T7 → foundation | Decision.CLARIFY | Already exists and is already counted by TypeSummary.clarified, so the escalation rule has somewhere to report. |
| plan T3 → foundation | detect_leak scope | OutputGuard must check every policy, matching evaluation/case.py, which already records leaked_policy_ids. |
| plan T2/T6 → foundation | ModelRequest | temperature and max_tokens are None-default and resolve against ModelSpec inside providers; classifiers and canonicalizers should leave them unset unless a prompt genuinely needs a different value. |
| plan T4 → benchmark | Family enum | Lives in benchmark/transformations/base.py; the detector and deterministic canonicalizer import it and reuse the decoders rather than reimplementing them. |
| plan T4/T6 → benchmark | substitution ambiguity | Ruling from the benchmark phase stands: Phase 4 calls decode_without_params and scores a case whose params say ambiguous in the ambiguous error category rather than as an incorrect reconstruction. |
| plan T5 → benchmark | code decoder | The restricted analyzer generalizes benchmark/transformations/code_snippet.py, which already refuses imports, attribute access on non-literals, calls outside a whitelist, oversized source, chained replaces and amplifying reconstructions. Start from it rather than from scratch. |
| carried requirements | 8 items in the plan's final section | Scoped out of Task 1 into their own Task 0, since several are refactors touching foundation modules and each deserves its own review rather than riding along with new classifiers. |
Ruling: run the carried requirements as Task 0 before Task 1. Cost if wrong: one extra review cycle.

## Review log
[CLAUDE_IMPLEMENTER] [DONE] Task 0 (dc3bd4e..23c4300, ten commits): all nine carried requirements. The substitution item was verify-only and its note had three inaccuracies, corrected. Also corrected a wrong claim in docs/safety-protocol.md that a spaced-out value escapes the leak detector.
[CLAUDE_REVIEWER] [REQUEST_CHANGES] Task 0: the logging move left provider_call_failed unredacted for library callers, the one line documented as able to carry a protected value; the completion marker turned a concurrent same-id collision from a refusal into two runs sharing a directory while the docstring still claimed the refusal.
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 0: same P1 on concurrency, plus a data-loss case I had missed — every run completed before this change has no marker, so reusing its id would silently overwrite a finished historical run.
[CLAUDE_IMPLEMENTER] [FIX] round 1 (52823b2, d123765, dd1ab7b): five-state run directory machine, redaction on every logger the package hands out, five minors.
Note: the implementer correctly rejected my prescribed mechanism for the redaction fix. A parent logger's filters are not consulted for a child's record, so attaching at the package logger would have covered nothing while making the documents read as if it had. It kept the goal and changed the mechanism; the Claude reviewer reproduced the finding and accepted the deviation.
[CLAUDE_REVIEWER] [APPROVE] round 1. [CODEX_REVIEWER] [REQUEST_CHANGES] round 1: Ctrl-C leaves the ownership file, and cleanup racing a reclaim raises instead of refusing.
[CLAUDE_IMPLEMENTER] [FIX] round 2 (f42a1d4, 5edbc13).
[CODEX_REVIEWER] [REQUEST_CHANGES] round 2 P1: the recovery path recreated the directory tolerantly, so a concurrently completed run could be claimed and overwritten.
[CLAUDE_IMPLEMENTER] [FIX] round 3 (06e6fe5, 3f38493): exclusive creation per pass with re-classification; while writing the required test it found a second route to the same overwrite that my prescribed fix did not reach, where a run completing between classification and claim releases its ownership file, and closed that too. Lifecycle moved to run_dir.py to stay under the file size limit.
[CODEX_REVIEWER] [REQUEST_CHANGES] round 3: the module split dropped three constants that were in the runner's __all__ at the merge base.
Ruling at the 3-round cap: one-line compatibility re-export, ruled in and verified myself (66188bb) rather than a fourth review pair. Cost if wrong: none identified.
Task 0: fix round 1/3 (7 addressed, 2 new open); round 2/3 (5 addressed, 1 new open); round 3/3 (1 addressed + 1 self-found, 1 adjudicated in)
Task 0: minor (deferred): a kill between the manifest write and the marker write is refused safely but diagnosed as a pre-marker run; the redaction stack is process-wide rather than per-thread, so concurrent runs over-redact each other, which is the safe direction; resume is directory-level, not the per-case skip the evaluation plan wants.
Task 0: complete (commits d2cd9f0..66188bb, review clean)

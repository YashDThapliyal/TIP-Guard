# SDD ledger — plan: docs/superpowers/plans/2026-09-07-foundation.md

Spec: docs/project-plan.md (reachable). Branch: build/foundation (created in place; repo was empty, no worktree needed).
Ruling: work on branch build/foundation in the main checkout, merge locally when the final review is clean — repo is brand new with no remote, user directed autonomous execution — cost if wrong: an unwanted local merge, trivially reverted.
Ruling: package lives at src/tipguard/<subpackage> rather than the spec's bare src/<subpackage> so it is pip-installable with a `tipguard` entry point — cost if wrong: directory rename.
Ruling: cross-review = Claude task reviewer (SDD template) AND Codex review of the same range; both must approve; max 3 fix loops per task per user directive (tighter than SDD's 5).

## Pre-flight scan
| Tasks | Interface | Finding |
| --- | --- | --- |
| T2→T4,T5 | ModelSpec fields (provider, model, base_url, costs) | consistent |
| T4→T5 | ProviderError location | T5 adds it to models/types.py; T4 does not define it — consistent, T5 owns it |
| T5→T8 | ProviderRegistry.get/spec, ResponseCache(path) | consistent |
| T6→T7,T8 | Decision/CaseType/Split enums from benchmark/schema.py | consistent single definition |
| T7→T8 | GuardrailResult fields ↔ CaseRecord fields | consistent (same names/types) |
| T6 tests | `from tests.benchmark.test_schema import make_case` | requires tests/__init__.py + tests/benchmark/__init__.py — plan notes it |
| T1 | coverage gate in pyproject vs Task 1 run with --no-cov | intentional; gate active from T2 |
| T3 | test_seeding second assertion is convoluted but correct | leave; reviewer may flag as Minor |
| T8 | TIPGUARD_OUTPUT_DIR env override | ruling recorded in plan |

## Review log
[CLAUDE_REVIEWER] [APPROVE] Task 1 spec ✅, quality approved; minors: .python-version unrequested (accepted), ruff format rewrote docs plan code fences.
[CODEX_REVIEWER] [APPROVE] Task 1 "no actionable defects"; lint/format/type/tests pass.
Task 1: minor (deferred): ruff format touches docs/ markdown code fences.
Ruling: Task 2 adds `extend-exclude = ["docs"]` to [tool.ruff] so plans are never reformatted — cost if wrong: none material.
Task 1: complete (commits e799681..3f24c0f, review clean)
[CLAUDE_REVIEWER] [APPROVE] Task 2 spec ✅; minors: YAMLError branch untested, broad pytest.raises(Exception) (plan-mandated), api-token value may trip secret scanners.
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 2 P2: loader lets OSError/UnicodeDecodeError escape instead of ConfigError.
[CLAUDE_IMPLEMENTER] [FIX] Task 2 round 1: wrapped read errors; added malformed-YAML and non-UTF-8 tests (6bd2819).
[CLAUDE_REVIEWER] [APPROVE] Task 2 fix round 1: both findings ADDRESSED, no breakage.
[CODEX_REVIEWER] [APPROVE] Task 2 fix round 1: correct; (Codex sandbox cannot run pytest — read-only tmp; suite verified locally 97% cov).
Task 2: fix round 1/3 (2 addressed, 0 open; commits 84b0082..6bd2819)
Task 2: minor (deferred): protect-api-token value resembles live-key prefix; consider renaming if a secret scanner is added.
Task 2: complete (commits 3f24c0f..6bd2819, review clean)
[CLAUDE_REVIEWER] [REQUEST_CHANGES] Task 3: extra-key spoofing of ts/level/logger; overlapping protected values partially redacted.
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 3: P1 nested extras unredacted; P2 overlapping values; P2 sets hash non-deterministically.
[CLAUDE_IMPLEMENTER] [FIX] Task 3 round 1 (28518fc): reserved keys win, longest-first alternation, recursive redaction, canonical sets, slug fallback.
[CLAUDE_REVIEWER] [APPROVE] round 1 all addressed; minor: Enum hashes as ClassName.MEMBER.
[CODEX_REVIEWER] [REQUEST_CHANGES] round 1: mixed-type dict keys now raise TypeError.
[CLAUDE_IMPLEMENTER] [FIX] Task 3 round 2 (7a4f3b2): str keys, Enum→value, Path→str.
[CLAUDE_REVIEWER] [REQUEST_CHANGES] round 2: str() key normalization collapses distinct keys silently.
[CODEX_REVIEWER] [REQUEST_CHANGES] round 2: same collision finding.
[CLAUDE_IMPLEMENTER] [FIX] Task 3 round 3 (259c0d4): dicts → sorted [key,value] pairs, types preserved.
[CLAUDE_REVIEWER] [APPROVE] round 3 all addressed, no breakage.
[CODEX_REVIEWER] [REQUEST_CHANGES] round 3 P2: {"a":1} and [["a",1]] canonicalize identically.
Task 3: fix round 1/3 (5 addressed, 1 new open; commits f4c3441..28518fc)
Task 3: fix round 2/3 (2 addressed, 1 new open; commits 28518fc..7a4f3b2)
Task 3: fix round 3/3 (1 addressed, 1 new open; commits 7a4f3b2..259c0d4)
Task 3: parked — mapping vs list-of-pairs canonical collision — Ruling: real but not load-bearing for YAML configs; trivial tag fix carried into Task 4 dispatch (encode mappings as ["__map__", pairs]) so it gets reviewed with Task 4 instead of a 4th loop — cost if wrong: one extra review cycle.
Task 3: complete (commits 6bd2819..259c0d4, 1 parked/carried)
[CLAUDE_REVIEWER] [REQUEST_CHANGES] Task 4: provider code ✅ verbatim; carried run_id regression test did not exercise the collision (plan-mandated defect).
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 4: "__map__" sentinel spoofable by literal list.
[CLAUDE_IMPLEMENTER] [FIX] Task 4 round 1 (a499f6c): uniform type tagging of every node; same-position regression tests.
[CLAUDE_REVIEWER] [APPROVE] Task 4 round 1: both addressed; no breakage; minor: bool/float/None branches untested.
[CODEX_REVIEWER] [APPROVE] Task 4 round 1: no regressions.
Task 4: fix round 1/3 (2 addressed, 0 open; commits 854c242..a499f6c)
Task 4: minor (deferred): _canonical bool/float/None/catch-all branches lack unit tests; MockRule invalid regex surfaces at complete() time.
Task 4: complete (commits 04cf320..a499f6c, review clean; Task 3 parked item closed here)
[CLAUDE_REVIEWER] [REQUEST_CHANGES] Task 5: OpenAI json mode lacks JSON instruction; minors: cache lifecycle, thread-safety, base_url not in key, corrupt rows raise, dead ModelSpec.max_tokens/temperature, Anthropic empty system, registry else-branch, test gaps, provider duplication.
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 5: P1 anthropic 1.4.0 rejects temperature (controller verified via inspect.signature); P2 base_url missing from cache key.
[CLAUDE_IMPLEMENTER] [FIX] Task 5 round 1 (4b0c10d): dropped temperature, shared JSON_INSTRUCTION, cache namespace=base_url, fail-open corrupt rows + close(), sonnet-5 pricing 2/10.
[CLAUDE_REVIEWER] [APPROVE] Task 5 round 1: all 5 addressed; minors: JSON instruction dropped if no system/user message; key change invalidates old cache rows.
[CODEX_REVIEWER] [APPROVE] Task 5 round 1: no regression.
Task 5: fix round 1/3 (5 addressed, 0 open; commits 52206a9..4b0c10d)
Task 5: minor (deferred): ResponseCache single-thread assumption undocumented; ModelSpec.max_tokens/temperature unused by providers; registry else-branch dispatches unknown providers to Ollama; OllamaProvider._build_client and Anthropic error branch untested; ~30 lines duplicated between cloud providers.
Task 5: complete (commits a499f6c..4b0c10d, review clean)
[CLAUDE_REVIEWER] [REQUEST_CHANGES] Task 6: load_cases leaks UnicodeDecodeError/OSError; minors: dead JSONDecodeError branch, no unicode normalisation, literal-value check is exact substring, PROHIBITED_TYPES naming.
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 6: same P2 (read failures bypass DatasetError).
[CLAUDE_IMPLEMENTER] [FIX] Task 6 round 1 (9419e13): wrapped open+iteration; removed dead branch; 3 tests.
[CLAUDE_REVIEWER] [APPROVE] Task 6 round 1: addressed, no breakage.
[CODEX_REVIEWER] [APPROVE] Task 6 round 1: no regressions.
Task 6: fix round 1/3 (1 addressed, 0 open; commits 155a85e..9419e13)
Task 6: minor (deferred): normalise_prompt lacks NFKC; TIP literal-value check misses punctuation-stripped forms (benchmark plan's dedupe/validator may harden).
Task 6: complete (commits 4b0c10d..9419e13, review clean)
[CLAUDE_REVIEWER] [APPROVE-WITH-FINDING] Task 7: spec ✅; plan-mandated: squash leak match false-positives on short values.
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 7: providers ignore ModelSpec temperature/max_tokens; _squash ASCII-only; empty protected values match everything.
Ruling: fix the plan-mandated leak-matcher defect now (min squashed length 10, boundary match for short values) rather than defer — leak rate is the headline metric — cost if wrong: slightly stricter detector, re-tunable.
[CLAUDE_IMPLEMENTER] [FIX] Task 7 round 1 (27464cf): leak hardening, NonEmptyStr policy values, ModelRequest None-defaults resolved from spec.
[CLAUDE_REVIEWER] [REQUEST_CHANGES] round 1: cache key ignores spec temperature/max_tokens now that requests defer to spec.
[CODEX_REVIEWER] [REQUEST_CHANGES] round 1: same P1 cache identity; P2 short-value regex ASCII boundaries.
[CLAUDE_IMPLEMENTER] [FIX] Task 7 round 2 (e7d683e): namespace = base_url+temperature+max_tokens; casefold + \w boundaries.
[CLAUDE_REVIEWER] [REQUEST_CHANGES] round 2: \w treats _ as word char → underscore-adjacent short values missed (controller-prescribed regex).
[CODEX_REVIEWER] [REQUEST_CHANGES] round 2: same underscore regression.
[CLAUDE_IMPLEMENTER] [FIX] Task 7 round 3 (8c4d0c8): [^\W_] boundaries; tests.
[CLAUDE_REVIEWER] [APPROVE] round 3 addressed, no breakage. [CODEX_REVIEWER] [APPROVE] round 3.
Task 7: fix round 1/3 (4 addressed, 1 new open; 7abb1f9..27464cf); round 2/3 (2 addressed, 1 new open; 27464cf..e7d683e); round 3/3 (1 addressed, 0 open; e7d683e..8c4d0c8)
Task 7: minor (deferred): guardrail/evaluation pydantic models omit extra="forbid" (plan-verbatim); LeakCheck is a dataclass; SAFE_REFUSAL has no consumer until baselines.
Task 5 deferred minor "ModelSpec.max_tokens/temperature unused" closed by Task 7 round 1.
Task 7: complete (commits 9419e13..8c4d0c8, review clean)
[CLAUDE_REVIEWER] [REQUEST_CHANGES] Task 8: cache never closed; unknown main_model alias tracebacks (plan-mandated); minors: limit ≤0, manifest output_dir, env-dependent test, logging ownership, mid-run loss, overwrite, CWD paths, whitespace in expected_answer.
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 8: P1 run-dir overwrite; P2 run-id path escape; P2 unknown alias; P2 validate dataset before provider calls; P2 negative limit; P2 close cache.
[CLAUDE_IMPLEMENTER] [FIX] Task 8 round 1 (fb03cf9): 9 items. Round 2 (a64bbba): atomic mkdir reservation, fullmatch, slug truncation. Round 3 (06ec1d5): cleanup scope covers cache init + artifact writing.
[CLAUDE_REVIEWER] [APPROVE] rounds 1–3. [CODEX_REVIEWER] [APPROVE] round 3 (rounds 1–2 each surfaced one follow-up).
Task 8: fix round 1/3 (9 addressed, 3 new open; e4c8109..fb03cf9); round 2/3 (3 addressed, 1 new open; fb03cf9..a64bbba); round 3/3 (1 addressed, 0 open; a64bbba..06ec1d5)
Task 8: minor (deferred): a mid-write failure leaves a partial run dir that blocks the same run id; run_experiment reconfigures process-wide logging; records are written only after all cases complete (streaming/resume is scheduled in the evaluation plan); CLI `correct` column is correct_decision not answer_correct; --limit replaces config.limit.
Task 8: complete (commits 8c4d0c8..06ec1d5, review clean)
[CLAUDE_REVIEWER] [REQUEST_CHANGES] Task 9: non-goal wording broadens spec; cross-policy disclosure gap omitted; minors on examples/redaction/H2-H3.
[CODEX_REVIEWER] [REQUEST_CHANGES] Task 9: api-token value looks real; hash-enforcement over-claim; release contradiction.
Ruling: allow two code/config changes inside the docs task so the docs' claims are true (visibly fictional token value; schema requires protected_value_hash for direct/tip) — cost if wrong: none material, both reviewed.
[CLAUDE_IMPLEMENTER] [FIX] Task 9 round 1 (6a4a69e fix, 912f8c0 docs): all 8 items.
[CLAUDE_REVIEWER] [APPROVE] Task 9 round 1. [CODEX_REVIEWER] [APPROVE] Task 9 round 1.
Task 9: fix round 1/3 (8 addressed, 0 open; commits 41bbf08..912f8c0)
Task 2 deferred minor (api-token resembles live key) closed by Task 9 round 1.
Task 9: complete (commits 06ec1d5..912f8c0, review clean)
ALL TASKS COMPLETE — proceeding to final whole-branch review.
[CLAUDE_REVIEWER] [FINAL] Ready with fixes. Important: leak check only vs own policy; cached cost/latency skew; registry KeyError; build_guardrail single provider; partial run dir poisons id / no resume; provider error text unredacted. Triage: T1/T2/T3-parked/T5-unused-spec/T5-dup/T7-LeakCheck/T7-SAFE_REFUSAL/T8-limit → drop (closed or intended); T8 correct column → fix-before-merge; rest → next phase.
[CODEX_REVIEWER] [FINAL] P2 Anthropic base_url ignored; P2 cached cost_usd stale across pricing; P2 default=str log values bypass redaction.
Ruling: final fix wave = Claude #1,#2(light: cached count + uncached latency + cost recomputed from tokens),#3,#4,#6 + column rename + README ruff-format + TypeSummary answer_total/clarified + all three Codex P2s + plan amendments (determinism wording, Task 7 signature). Deferred to next phase with ledger lines: #5 resume/atomic completion; FrozenModel hoist; assert_never; MockRule precompile; NFKC/squash in validator; cache WAL/timeout; logging ownership; dataset sha in run id; mypy tests.
[CLAUDE_IMPLEMENTER] [FIX] Final wave: d26580f (cross-policy leaks, cached cost/latency, summary fields), 7e20070 (ConfigError aliases, registry into build_guardrail), 640ff22 (provider/CLI/log redaction, Anthropic base_url), e30ec4c (docs + plan amendments). 140 passed, 98.26% coverage.
Note: implementer found and closed a real leak path while doing item 5 — pydantic extra_forbidden errors quote input_value, so validate-dataset previously echoed a protected value verbatim.
[CLAUDE_REVIEWER] [APPROVE] Final fix wave: all 10 addressed, both deviations accepted, ready to merge. Minor: registry ConfigError omits the models.yaml path; latency-filter comment imprecise; Task 7 Interfaces block partially stale.
[CODEX_REVIEWER] [REQUEST_CHANGES] Final fix wave P2: latency filter `cached_model_calls == 0` drops partially cached records; should be `cached_model_calls < model_calls`.
Ruling (residual, load-bearing for Phase 3): Codex is right for multi-call defenses — with a warm cache a partly-cached TIP-Guard record would be dropped, reintroducing the zero-percentile failure one layer up. Today every record has exactly one model call, so both filters are identical and no current result changes. Rather than an unreviewed second fix wave, the change is carried into the defenses plan as a required first-task change, where it gets a review seat. Cost if wrong: one release of latency stats computed over wholly-uncached records only, recomputable from results.jsonl.
Ruling (residual, minor): registry ConfigError wording and the stale Task 7 Interfaces block are parked; the foundation plan is closed history and the message already lists known aliases. Cost if wrong: a slightly less helpful error string.
FINAL: foundation branch ready to merge.

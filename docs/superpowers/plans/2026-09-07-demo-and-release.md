# TIP-Guard Demonstration and Release (Phases 8–9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local interactive dashboard that shows how TIP-Guard reaches a decision for any prompt (side by side with no-defense and a standard classifier), plus the final technical report, architecture docs, reproduction instructions, and cards for the dataset, models, and evaluators.

**Architecture:** The dashboard is a single-process local web app (`tipguard dashboard`) built on the stdlib `http.server` plus a self-contained HTML/JS page (no build step, no external CDN dependency required at runtime) that calls a JSON endpoint backed by `ProtectedModel` instances for each mode. All displayed traces pass through the redaction layer. The report is Markdown under `reports/` generated partly from `reports/analysis/` tables so numbers are never hand-copied.

**Tech Stack:** Python 3.12 stdlib server, vanilla HTML/CSS/JS, Markdown.

**Spec:** `docs/project-plan.md` Phases 8–9.

## Global Constraints

- Foundation Global Constraints apply.
- The dashboard must never display protected values or a canonicalizer's decoded payload when the decision is `block`; it shows the redacted `reasons`, component traces, categories, confidence, latency breakdown, and the safe response.
- Three selectable modes with exact labels: `No defense`, `Standard classifier`, `TIP-aware guardrail`; each mode maps to a `DefenseConfig` (`no_defense`, `input_output_classifier`, `tip_guard`).
- Example scenarios (≥ 8) are loaded from `dashboard/scenarios.yaml` and cover: direct request, base64 TIP, caesar TIP, riddle TIP, code TIP, benign base64, benign caesar, hard negative, plus one false-positive example found in Phase 7 results.
- Audit log lines emitted during dashboard use are redacted (existing logger) and the dashboard has an "Export evaluation summary" button that downloads the latest `reports/analysis/summary.json` and Markdown tables.

---

### Task 1: Dashboard backend
- `src/tipguard/dashboard/server.py`: `tipguard dashboard --config experiments/dashboard.yaml --port 8330`; endpoints `GET /` (page), `GET /api/scenarios`, `GET /api/modes`, `POST /api/run {prompt, mode}` → `{decision, response_text, reasons, components:[{component,triggered,detail,latency_ms}], categories, confidence, latency_ms, model_calls, cost_usd}` redacted; `POST /api/compare {prompt}` runs all three modes; `GET /api/export` → summary JSON + tables zip.
- `experiments/dashboard.yaml` selects the main model and defense params (defaults to mock models so the dashboard runs offline; real models selectable via `--main-model` and `--judge-model` flags).
- Tests with the stdlib test client (`http.client`) against a server on an ephemeral port using mock providers: every endpoint, redaction of a leaked canary in a `block` response, and the `compare` shape.

### Task 2: Dashboard frontend
- `src/tipguard/dashboard/static/index.html` (+ inline CSS/JS): prompt textarea, mode selector, scenario picker, "Run" and "Compare all modes"; result panel with decision badge, reasons list, component timeline (bars proportional to latency), categories + confidence, safe response, cost/calls; side-by-side compare table; export button; a 5-minute "How it works" panel summarising the architecture diagram.
- Manual check script `scripts/dashboard-smoke.sh` that starts the server with mocks, hits `/api/compare`, and exits non-zero on failure; wire into CI.
- Docs: `docs/demo.md` setup + walkthrough (five-minute reviewer path).

### Task 3: Final report and release docs
- `reports/final-report.md` with the 11 required sections; results/ablation/failure sections import numbers from `reports/analysis/*.md` tables (script `tipguard report --analysis reports/analysis --out reports/final-report.md` fills a Jinja-free template using `str.format` placeholders); the headline claim follows the spec's precise template with X/Y/Z filled from Experiment 2 and the strongest baseline.
- `docs/architecture.md` (component diagram, data flow, decision rules, ablation switches), `docs/reproduction.md` (exact commands from clean clone to headline tables, expected runtime and cost), `docs/model-card.md` and `docs/evaluator-card.md` (models used with versions, judge prompt versions, calibration results, known biases), `docs/dataset-card.md` already exists — link it.
- README polish: quickstart, demo, reproduce, safety, citation stub, license.
- Release sanitisation check `tipguard release-check`: greps the tree for protected values outside `configs/policies.yaml` and for any `reports/runs/` content staged in git; CI runs it.

## Self-Review
Demonstration features (transformation, intent, categories, confidence, decision, components, safe response, latency) → T1/T2; modes → T1; no protected values shown → T1 redaction test; deliverables (dashboard, scenarios, side-by-side, export, instructions) → T1–T2; Phase 9 report structure and all deliverables → T3; completion criteria (install, generate/load, run baseline, run TIP-Guard, reproduce, inspect limitations) → `docs/reproduction.md` + README.

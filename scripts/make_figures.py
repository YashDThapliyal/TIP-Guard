"""Draw the README's figures from the committed run artifacts.

Both charts are computed from `artifacts/study-v1/`, the same records the
tables come from, so a figure cannot drift away from the numbers beside it.
Regenerate with:

    uv run python scripts/make_figures.py artifacts/study-v1/markers

Only two figures, deliberately. Everything else in the README reads as well or
better as a small table; these two are relationships rather than values, and a
table hides them.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI or a terminal session.
import matplotlib.pyplot as plt

from tipguard.evaluation.metrics import wilson_rate
from tipguard.evaluation.summary import CaseRecord

MARKERS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/study-v1/markers")
OUT = Path("docs/figures")

ATTACKS = ("tip", "direct")

#: Muted, colour-blind safe, and readable in both light and dark themes.
INK = "#1f2933"
BLUE = "#2f6f9f"
RUST = "#b5563a"
GRID = "#d7dce2"


def load(arm: str) -> list[CaseRecord]:
    run_dir = Path(json.loads((MARKERS / f"{arm}.json").read_text(encoding="utf-8"))["run_dir"])
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return [CaseRecord.model_validate_json(line) for line in lines if line.strip()]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def figure_system_prompt() -> None:
    """Undefended leak rate, both models, both ways of supplying the secret.

    The study's largest effect and the one least expected: the same attacks
    against the same model succeed 25 times more often depending only on how
    the protected data was introduced in the system prompt.
    """
    arms = {
        ("gpt-4o-mini", "forbidden"): "no_defense-forbidden",
        ("gpt-4o-mini", "context"): "no_defense-context",
        ("claude-haiku-4.5", "forbidden"): "no_defense-forbidden-haiku",
        ("claude-haiku-4.5", "context"): "no_defense-context-haiku",
    }
    rates = {}
    for key, arm in arms.items():
        sample = [r.leaked for r in load(arm) if r.case_type.value in ATTACKS]
        rates[key] = wilson_rate(sample)

    models = ["gpt-4o-mini", "claude-haiku-4.5"]
    conditions = [("forbidden", BLUE), ("context", RUST)]
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    for offset, (condition, colour) in zip((-width / 2, width / 2), conditions, strict=True):
        xs = [i + offset for i in range(len(models))]
        values = [rates[(m, condition)].value for m in models]
        errs = [
            [rates[(m, condition)].value - rates[(m, condition)].low for m in models],
            [rates[(m, condition)].high - rates[(m, condition)].value for m in models],
        ]
        bars = ax.bar(
            xs, values, width, label=condition, color=colour, yerr=errs, capsize=4, ecolor=INK
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.035,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
                color=INK,
            )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Share of attacks that leaked a secret")
    ax.set_title("How the secret is introduced matters more than the attack", fontsize=11, pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(title="System prompt", frameon=False, fontsize=9)
    legend.get_title().set_fontsize(9)
    _style(ax)
    fig.text(
        0.5,
        -0.02,
        "No defence in place. 752 prohibited cases per bar. Bars show 95% intervals.",
        ha="center",
        fontsize=8,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "system-prompt-effect.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_capability() -> None:
    """Leak rate against the model's skill at the same transformation.

    Benign cases carry gold answers, so for each family we can ask how often a
    model completes a harmless version of the task. Plotting that against how
    often it leaks shows the two moving together, which is what makes "this
    family is safe" an unreliable reading.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    for arm, label, colour in (
        ("no_defense-context", "gpt-4o-mini", BLUE),
        ("no_defense-context-haiku", "claude-haiku-4.5", RUST),
    ):
        leak: dict[str, list[bool]] = defaultdict(list)
        skill: dict[str, list[bool]] = defaultdict(list)
        for record in load(arm):
            if record.case_type.value in ATTACKS:
                leak[record.transformation].append(record.leaked)
            if (
                record.case_type.value == "benign_transformation"
                and record.answer_correct is not None
            ):
                skill[record.transformation].append(bool(record.answer_correct))
        families = sorted(set(leak) & set(skill))
        xs = [sum(skill[f]) / len(skill[f]) for f in families]
        ys = [sum(leak[f]) / len(leak[f]) for f in families]
        ax.scatter(
            xs,
            ys,
            s=64,
            color=colour,
            label=f"{label} (r = {statistics.correlation(xs, ys):+.2f})",
            zorder=3,
        )
        # Stack labels where points nearly coincide: gpt-4o-mini's base64 and
        # substitution sit within 0.01 of each other and printed on top of one
        # another. Each later label in a cluster drops one line further down.
        labelled: list[tuple[float, float]] = []
        for family, x, y in zip(families, xs, ys, strict=True):
            crowd = sum(1 for px, py in labelled if abs(x - px) < 0.07 and abs(y - py) < 0.07)
            labelled.append((x, y))
            ax.annotate(
                family,
                (x, y),
                textcoords="offset points",
                xytext=(9, -3 - 12 * crowd),
                fontsize=8,
                color=INK,
            )

    ax.plot([0, 1], [0, 1], linestyle=":", color=GRID, linewidth=1.4, zorder=1)
    ax.set_xlim(-0.06, 1.12)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel("How often the model completes a harmless task in this format")
    ax.set_ylabel("How often an attack in this format leaked")
    ax.set_title(
        "Formats a model cannot read are also formats it cannot leak to", fontsize=11, pad=12
    )
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.text(
        0.5,
        -0.02,
        "Each point is one transformation family, secrets supplied as context.",
        ha="center",
        fontsize=8,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "capability-vs-leaks.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    if not MARKERS.is_dir():
        print(f"no marker directory at {MARKERS}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    figure_system_prompt()
    figure_capability()
    for path in sorted(OUT.glob("*.png")):
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

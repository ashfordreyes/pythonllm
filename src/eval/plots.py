"""PNG charts for training and eval runs.

Writes files rather than showing figures, so the same functions work from a
Colab notebook, a script, or CI. Every function returns the path it wrote.

Palette: categorical slots 1-2 (blue/orange) for two-series charts, a single
blue for magnitude bars, neutral gray for "not attempted". The blue/orange
pair is validated for colorblind separation against this surface; do not
substitute hues without re-validating.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on Colab or CI
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
NEUTRAL = "#d5d4cf"   # not attempted / no data
GAP_X = 0.004         # ~2px of surface between adjacent fills, in 0..1 axis units

FIGSIZE = (8, 4.5)
DPI = 150


def _new_axes(title: str, figsize=FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    return fig, ax


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_loss_curve(log_history: list[dict], out_path: str | Path) -> Path:
    """Plot train and eval loss over steps from `trainer.state.log_history`.

    Both losses share one y-axis on purpose -- they are the same measure, and a
    second scale would make the gap between them unreadable.
    """
    train = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "step" in e]
    evals = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e and "step" in e]

    fig, ax = _new_axes("Stage 1 training loss")
    ax.set_xlabel("step", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("loss", color=INK_MUTED, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    for points, color, label in ((train, SERIES_1, "train"), (evals, SERIES_2, "eval")):
        if not points:
            continue
        xs, ys = zip(*points)
        marker = "o" if len(xs) < 40 else None
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=4, label=label)
        # Direct-label the series end so identity never rests on color alone.
        ax.annotate(
            f"{label} {ys[-1]:.3f}",
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            va="center",
        )

    if not train and not evals:
        ax.text(0.5, 0.5, "no loss logged", ha="center", color=INK_MUTED, transform=ax.transAxes)
    elif train and evals:
        legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK)

    return _save(fig, out_path)


def plot_score_breakdown(aggregate_result: dict, out_path: str | Path) -> Path:
    """Horizontal bars of pass rate per scoring tier.

    Each bar is drawn against its own `attempted` count, and the untested
    remainder is shown in neutral gray with the raw counts labelled, so a tier
    that ran on 4 of 50 examples cannot be misread as a rate over all 50.
    """
    tiers = aggregate_result.get("tiers", {})
    names = list(tiers)
    fig, ax = _new_axes("Held-out scores by tier", figsize=(8, 0.55 * max(len(names), 1) + 1.8))

    if not names:
        ax.text(0.5, 0.5, "no scores", ha="center", color=INK_MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return _save(fig, out_path)

    total = aggregate_result.get("n_examples", 0)
    positions = range(len(names))
    for y, name in zip(positions, names):
        tier = tiers[name]
        attempted, passed = tier["attempted"], tier["passed"]
        if attempted:
            rate = passed / attempted
            # 2px surface gap between the two fills (x-axis spans 0..1 here).
            gap = GAP_X if 0 < rate < 1 else 0.0
            ax.barh(y, max(rate - gap / 2, 0), height=0.55, color=SERIES_1)
            ax.barh(y, max(1 - rate - gap / 2, 0), left=rate + gap / 2, height=0.55, color=NEUTRAL)
            note = f"{passed}/{attempted}"
            if total and attempted < total:
                note += f"  ({total - attempted} not attempted)"
        else:
            ax.barh(y, 1, height=0.55, color=NEUTRAL)
            note = "not attempted"
        ax.annotate(note, xy=(1.01, y), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center", annotation_clip=False)

    ax.set_yticks(list(positions))
    ax.set_yticklabels([n.replace("_", " ") for n in names], color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("pass rate (of attempted)", color=INK_MUTED, fontsize=10)
    return _save(fig, out_path)


def plot_plan_error_types(aggregate_result: dict, out_path: str | Path) -> Path:
    """Frequency of each `validate_plan` failure category across generations."""
    counts = aggregate_result.get("plan_error_types", {})
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = _new_axes("Plan validation errors", figsize=(8, 0.55 * max(len(ordered), 1) + 1.8))

    if not ordered:
        ax.text(0.5, 0.5, "no validation errors", ha="center", color=INK_MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return _save(fig, out_path)

    labels, values = zip(*ordered)
    positions = range(len(labels))
    ax.barh(list(positions), values, height=0.55, color=SERIES_1)
    for y, value in zip(positions, values):
        ax.annotate(str(value), xy=(value, y), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center", annotation_clip=False)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("occurrences", color=INK_MUTED, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return _save(fig, out_path)

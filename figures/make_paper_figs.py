"""Build the Track A, Track B and cross-track figures.

Shares the corpus loader and the visual style of ``make_track_a_figs.py`` so
every figure has the same look. Writes five PDFs into this directory and
prints every plotted number.

Track B figures read ``data/track_b/corpus.csv``, one row per attempt.

Run from the repository root:
    uv run --no-project --with pandas --with matplotlib python figures/make_paper_figs.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from make_track_a_figs import (
    COLUMN_WIDTH_IN, COUPLINGS, SMALL_FONT_PT, TRACK_A_COUPLINGS, apply_figure_style,
    load_corpus, save_figure,
)

FIG_DIR = Path(__file__).resolve().parent
TRACK_A_SCALE_FACTORS = (1, 3, 10)
TRACK_B_SCALE_FACTORS = (1, 3)
WILSON_Z = 1.959964
GRID_GREY = "#e6e6e6"

# Okabe-Ito, one colour per coupling, in the row order of the phase-share figure.
# Marker = server and line style = backend, so identity survives grayscale.
COUPLING_COLOURS = {
    ("mcp-alchemy", "duckdb"): "#009E73",
    ("mcp-alchemy", "mysql"): "#56B4E9",
    ("dbhub", "mysql"): "#E69F00",
    ("mcp-alchemy", "postgres"): "#0072B2",
    ("dbhub", "postgres"): "#D55E00",
    ("anthropic-pg", "postgres"): "#CC79A7",
}
SERVER_MARKERS = {"mcp-alchemy": "o", "dbhub": "s", "anthropic-pg": "^"}
BACKEND_LINESTYLES = {"duckdb": ":", "mysql": "--", "postgres": "-"}
SERVERS = ("mcp-alchemy", "dbhub", "anthropic-pg")

# Fixed-cost segments reuse the phase-share figure's colours for the same phases.
# Discovery stays under 6 ms, too thin to see, so it rides on the lifecycle bar.
FIXED_SEGMENTS = (  # (columns, legend, colour, hatch)
    (("T_life_ms", "T_disc_ms"), "Lifecycle + discovery", "#56B4E9", ""),
    (("tail_ms",), "Teardown + residual", "#D55E00", "xxxx"),
)

OUTCOME_CLASSES = (  # (key, legend, colour, hatch)
    ("correct", "Correct", "#0072B2", ""),
    ("incorrect", "Incorrect answer", "#E69F00", "////"),
    ("call_limit_exceeded", "Iteration limit", "#009E73", ""),
    ("agent_error", "Request rejected", "#D55E00", "xxxx"),
    ("unparseable", "No answer", "#999999", ""),
)


@dataclass(frozen=True)
class ModelStyle:
    """How one agent model is drawn in the Track B figures."""

    key: str
    label: str
    marker: str
    colour: str


MODELS = (
    ModelStyle("deepseek-v4-flash", "DeepSeek V4 Flash", "o", "#0072B2"),
    ModelStyle("glm-5.3-flash", "GLM 5.3 Flash", "s", "#E69F00"),
    ModelStyle("gpt-6-luna", "GPT-6 Luna", "^", "#009E73"),
    ModelStyle("claude-haiku-5.5", "Claude Haiku 5.5", "D", "#CC79A7"),
)
POOLED_MODEL = "all"
EXPECTED_ATTEMPTS_PER_CELL = 150
TRACK_B_CORPUS = FIG_DIR.parent / "data" / "track_b" / "corpus.csv"
COMPLETION_CEILING_PERCENT = 84.0


class TrackBDataError(RuntimeError):
    """Raised when the Track B corpus does not form a complete grid."""


def coupling_label(server: str, backend: str) -> str:
    """Row label used across all figures."""
    return next(name for s, b, name in COUPLINGS if (s, b) == (server, backend))


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, in percent."""
    p = successes / trials
    denom = 1 + WILSON_Z**2 / trials
    centre = (p + WILSON_Z**2 / (2 * trials)) / denom
    half = WILSON_Z * math.sqrt(p * (1 - p) / trials + WILSON_Z**2 / (4 * trials**2)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def load_track_b() -> pd.DataFrame:
    """Load the Track B corpus and check that the grid is complete."""
    attempts = pd.read_csv(TRACK_B_CORPUS)
    sizes = attempts.groupby(["model", "server", "backend", "scale_factor"]).size()
    expected_cells = len(MODELS) * len(COUPLINGS) * len(TRACK_B_SCALE_FACTORS)
    if len(sizes) != expected_cells or (sizes != EXPECTED_ATTEMPTS_PER_CELL).any():
        raise TrackBDataError(f"incomplete Track B grid: {len(sizes)} cells, sizes {sizes.unique()}")
    unknown = set(attempts["outcome"]) - {key for key, *_ in OUTCOME_CLASSES}
    if unknown:
        raise TrackBDataError(f"unmapped outcomes {sorted(unknown)}")
    return attempts


def large_result_questions(attempts: pd.DataFrame) -> set[str]:
    """Return the questions no attempt answered correctly."""
    correct = (attempts["outcome"] == "correct").groupby(attempts["query"]).sum()
    return set(correct[correct == 0].index)


def completion_table(attempts: pd.DataFrame) -> pd.DataFrame:
    """Completion rate and Wilson interval per model, coupling and scale factor."""
    rows = []
    for (model, server, backend, sf), cell in attempts.groupby(["model", "server", "backend", "scale_factor"]):
        correct = int((cell["outcome"] == "correct").sum())
        low, high = wilson_interval(correct, len(cell))
        rows.append({"model": model, "server": server, "backend": backend, "scale_factor": sf,
                     "correct": correct, "n": len(cell), "rate": 100 * correct / len(cell),
                     "low": low, "high": high})
    return pd.DataFrame(rows)


def track_a_ptr_by_sf(corpus: pd.DataFrame) -> pd.DataFrame:
    """Median over cells of the per-cell median PTR, per coupling and scale factor."""
    cells = corpus.groupby(["server", "backend", "scale_factor", "cell_id"])["ptr"].median().reset_index()
    return cells.groupby(["server", "backend", "scale_factor"])["ptr"].median().unstack("scale_factor")


def _style_log_axis(axis, ticks: list[float]) -> None:
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _recessive_grid(ax, which: str) -> None:
    ax.grid(axis=which, which="major", color=GRID_GREY, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def draw_ptr_scaling(ptr: pd.DataFrame) -> None:
    """Fig: median PTR against scale factor, one line per coupling."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.3))
    for server, backend, name in TRACK_A_COUPLINGS:
        values = [ptr.loc[(server, backend), sf] for sf in TRACK_A_SCALE_FACTORS]
        ax.plot(TRACK_A_SCALE_FACTORS, values, color=COUPLING_COLOURS[(server, backend)],
                linestyle=BACKEND_LINESTYLES[backend], marker=SERVER_MARKERS[server],
                markersize=4, linewidth=1.0, markeredgecolor="white", markeredgewidth=0.4,
                zorder=3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 11.5)
    ax.set_ylim(0.007, 6)
    ax.xaxis.set_minor_locator(FixedLocator([]))
    _style_log_axis(ax.xaxis, list(TRACK_A_SCALE_FACTORS))
    _style_log_axis(ax.yaxis, [0.01, 0.1, 1])
    ax.set_xlabel("TPC-H scale factor")
    ax.set_ylabel("Median PTR")
    _recessive_grid(ax, "y")
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              handlelength=2.6, columnspacing=1.2)
    save_figure(fig, FIG_DIR / "track_a_ptr_scaling.pdf")


def draw_fixed_cost(corpus: pd.DataFrame) -> None:
    """Fig: per-interaction fixed cost per coupling, stacked by phase."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.75))
    for row, (server, backend, name) in enumerate(TRACK_A_COUPLINGS):
        traces = corpus[(corpus["server"] == server) & (corpus["backend"] == backend)]
        left = 0.0
        for columns, legend, colour, hatch in FIXED_SEGMENTS:
            width = sum(max(0.0, traces[column].median()) for column in columns)
            ax.barh(row, width, left=left, height=0.7, color=colour, hatch=hatch,
                    edgecolor="white", linewidth=0.3, label=legend if row == 0 else None)
            left += width
        total = traces[["T_disc_ms", "T_life_ms", "tail_ms"]].sum(axis=1).median()
        text = f"{total:,.1f}" if total < 10 else f"{total:,.0f}"
        ax.text(left + 60, row, text, va="center", fontsize=SMALL_FONT_PT)
    ax.set_yticks(range(len(TRACK_A_COUPLINGS)), [name for *_, name in TRACK_A_COUPLINGS])
    ax.invert_yaxis()
    ax.set_xlim(0, 6000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Median fixed cost per interaction (ms)")
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0),
              handlelength=1.6, columnspacing=1.2)
    save_figure(fig, FIG_DIR / "track_a_fixed_cost.pdf")


def draw_completion(completion: pd.DataFrame) -> None:
    """Fig: completion per coupling and model, one panel per scale factor."""
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(COLUMN_WIDTH_IN, 2.4))
    for ax, sf in zip(axes, TRACK_B_SCALE_FACTORS):
        for row, (server, backend, _) in enumerate(COUPLINGS):
            for slot, model in enumerate(MODELS):
                cell = completion[(completion["model"] == model.key) & (completion["server"] == server)
                                  & (completion["backend"] == backend) & (completion["scale_factor"] == sf)].iloc[0]
                ax.plot(cell["rate"], row + (slot - 1.5) * 0.17, marker=model.marker, markersize=3.6,
                        linestyle="none", color=model.colour, markeredgecolor="white",
                        markeredgewidth=0.3, zorder=3)
        ax.axvline(COMPLETION_CEILING_PERCENT, color="#555555", linewidth=0.6, linestyle="--", zorder=1)
        ax.set_xlim(65, 87)
        ax.set_title(f"SF{sf}", fontsize=SMALL_FONT_PT, pad=2)
        _recessive_grid(ax, "x")
    axes[0].set_yticks(range(len(COUPLINGS)), [name for *_, name in COUPLINGS])
    axes[0].set_ylim(len(COUPLINGS) - 0.5, -0.5)
    fig.supxlabel("Completion rate (%)", fontsize=axes[0].xaxis.label.get_size())
    handles = [Line2D([], [], color=m.colour, marker=m.marker, linestyle="none", markersize=3.6, label=m.label)
               for m in MODELS]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="outside upper center",
               handlelength=0.8, handletextpad=0.3, columnspacing=0.7)
    save_figure(fig, FIG_DIR / "track_b_completion.pdf")


def outcome_shares(attempts: pd.DataFrame) -> pd.DataFrame:
    """Count of each outcome class per server, pooled over models and scale factors."""
    counts = attempts.groupby(["server", "outcome"]).size().unstack(fill_value=0)
    return counts.reindex(index=list(SERVERS), columns=[k for k, *_ in OUTCOME_CLASSES], fill_value=0)


def _draw_outcome_panel(ax, counts: pd.DataFrame, title: str) -> None:
    for row, server in enumerate(SERVERS):
        total = counts.loc[server].sum()
        left = 0.0
        for key, legend, colour, hatch in OUTCOME_CLASSES:
            share = 100 * counts.loc[server, key] / total
            ax.barh(row, share, left=left, height=0.62, color=colour, hatch=hatch,
                    edgecolor="white", linewidth=0.3, label=legend)
            if share >= 7:
                ink = "white" if key in ("correct", "call_limit_exceeded") else "black"
                ax.text(left + share / 2, row, f"{share:.0f}", ha="center", va="center",
                        fontsize=SMALL_FONT_PT, color=ink,
                        bbox={"boxstyle": "square,pad=0.05", "facecolor": colour, "linewidth": 0}
                        if hatch else None)
            left += share
        ax.text(101.5, row, f"{total:,}", va="center", fontsize=SMALL_FONT_PT)
    ax.set_yticks(range(len(SERVERS)), list(SERVERS))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=SMALL_FONT_PT, pad=2, loc="left")


def draw_outcomes(large: pd.DataFrame, other: pd.DataFrame) -> None:
    """Fig: 100% stacked outcomes per server, split by question group."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(COLUMN_WIDTH_IN, 2.6))
    _draw_outcome_panel(axes[0], large, "Eight large-result questions")
    _draw_outcome_panel(axes[1], other, "Other 42 questions")
    axes[1].set_xlabel("Share of attempts (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), ncol=3, frameon=False, loc="outside upper center",
               handlelength=1.6, columnspacing=0.8)
    save_figure(fig, FIG_DIR / "track_b_outcomes.pdf")


# The in-process coupling sits two orders of magnitude left of the others, so
# the x axis is split into a DuckDB panel and a client-server panel.
CROSS_PANELS = (  # (xlim, ticks, width ratio)
    ((0.015, 0.04), [0.02, 0.03], 1.0),
    ((0.9, 5.2), [1, 2, 4], 4.2),
)
CROSS_LABEL_POSITIONS = {  # (x, y, ha, leader anchor on the text box); hand-placed
    "mcp-alchemy / DuckDB": (0.026, 75.0, "center", (0.5, 1.0)),
    "mcp-alchemy / MySQL": (0.95, 86.0, "left", (0.3, 0.0)),
    "dbhub / MySQL": (0.95, 75.5, "left", (0.2, 1.0)),
    "mcp-alchemy / PG": (1.9, 75.5, "left", (0.2, 1.0)),
    "dbhub / PG": (2.1, 86.0, "left", (0.2, 0.0)),
    "anthropic-pg / PG": (5.0, 85.0, "right", (0.8, 0.0)),
}


def _cross_points(ptr: pd.DataFrame, completion: pd.DataFrame,
                  server: str, backend: str) -> list[tuple[float, float]]:
    points = []
    for sf in TRACK_B_SCALE_FACTORS:
        cell = completion[(completion["server"] == server) & (completion["backend"] == backend)
                          & (completion["scale_factor"] == sf)].iloc[0]
        points.append((ptr.loc[(server, backend), sf], cell["rate"]))
    return points


def _draw_axis_break(left_ax, right_ax) -> None:
    marks = {"marker": [(-1, -1.6), (1, 1.6)], "markersize": 5, "linestyle": "none",
             "color": "black", "mec": "black", "mew": 0.5, "clip_on": False}
    left_ax.plot([1], [0], transform=left_ax.transAxes, **marks)
    right_ax.plot([0], [0], transform=right_ax.transAxes, **marks)


def draw_cross_track(ptr: pd.DataFrame, completion: pd.DataFrame) -> None:
    """Fig: Track A PTR against Track B completion, SF1 filled and SF3 hollow."""
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(COLUMN_WIDTH_IN, 2.3),
                             gridspec_kw={"width_ratios": [r for *_, r in CROSS_PANELS]})
    for ax, (xlim, ticks, _) in zip(axes, CROSS_PANELS):
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        _style_log_axis(ax.xaxis, ticks)
        ax.xaxis.set_minor_locator(FixedLocator([]))
        _recessive_grid(ax, "y")
    for server, backend, name in COUPLINGS:
        ax = axes[0] if backend == "duckdb" else axes[1]
        colour = COUPLING_COLOURS[(server, backend)]
        points = _cross_points(ptr, completion, server, backend)
        ax.plot(*zip(*points), color=colour, linewidth=0.6, linestyle=BACKEND_LINESTYLES[backend], zorder=2)
        for (x, y), face in zip(points, (colour, "white")):
            ax.plot(x, y, marker=SERVER_MARKERS[server], markersize=5, linestyle="none",
                    markerfacecolor=face, markeredgecolor=colour, markeredgewidth=0.9, zorder=3)
        x_text, y_text, align, anchor = CROSS_LABEL_POSITIONS[name]
        ax.annotate(name.replace(" / ", " /\n") if backend == "duckdb" else name, xy=points[0],
                    xytext=(x_text, y_text), ha=align, va="center", fontsize=SMALL_FONT_PT, zorder=4,
                    arrowprops={"arrowstyle": "-", "color": "#999999", "linewidth": 0.4,
                                "shrinkA": 1, "shrinkB": 3, "relpos": anchor})
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", left=False)
    _draw_axis_break(axes[0], axes[1])
    axes[0].set_ylim(70, 90)
    axes[0].set_yticks(range(70, 91, 5))
    axes[0].set_ylabel("Track B completion (%)")
    fig.supxlabel("Track A median PTR at the same scale factor", fontsize=axes[0].xaxis.label.get_size())
    handles = [Line2D([], [], color="#555555", marker="o", markerfacecolor=face, markeredgecolor="#555555",
                      linestyle="none", markersize=4.5, label=label)
               for face, label in (("#555555", "SF1"), ("white", "SF3"))]
    axes[1].legend(handles=handles, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0))
    save_figure(fig, FIG_DIR / "cross_track.pdf")


def print_numbers(ptr: pd.DataFrame, corpus: pd.DataFrame, completion: pd.DataFrame,
                  counts: pd.DataFrame) -> None:
    """Print every plotted value so the captions and prose can cite them."""
    print("Track A median PTR (median of cell medians)")
    for server, backend, name in COUPLINGS:
        row = ptr.loc[(server, backend)]
        print(f"  {name:22s} " + "  ".join(f"SF{sf}={row[sf]:.3f}" for sf in TRACK_A_SCALE_FACTORS))
    print("Track A fixed cost (trace medians, ms)")
    for server, backend, name in COUPLINGS:
        t = corpus[(corpus["server"] == server) & (corpus["backend"] == backend)]
        parts = "  ".join(f"{c}={t[c].median():.1f}" for c in ("T_life_ms", "T_disc_ms", "tail_ms"))
        total = t[["T_disc_ms", "T_life_ms", "tail_ms"]].sum(axis=1).median()
        print(f"  {name:22s} n={len(t)} {parts}  median_of_sum={total:.1f}")
    print("Track B completion")
    for _, r in completion.iterrows():
        print(f"  {r['model']} {coupling_label(r['server'], r['backend']):22s} SF{r['scale_factor']} "
              f"{r['correct']}/{r['n']} = {r['rate']:.1f}% [{r['low']:.1f}, {r['high']:.1f}]")
    print("Track B outcomes per server, pooled over models and scale factors")
    print(counts.assign(total=counts.sum(axis=1)).to_string())
    print("  column totals:", counts.sum().to_dict(), "all:", int(counts.values.sum()))


def main() -> None:
    """Draw every Track A, Track B and cross-track figure."""
    apply_figure_style()
    corpus = load_corpus()
    ptr = track_a_ptr_by_sf(corpus)
    attempts = load_track_b()
    completion = completion_table(attempts)
    pooled = completion_table(attempts.assign(model=POOLED_MODEL))
    is_large = attempts["query"].isin(large_result_questions(attempts))
    print_numbers(ptr, corpus, pd.concat([completion, pooled]), outcome_shares(attempts))
    draw_ptr_scaling(ptr)
    draw_fixed_cost(corpus)
    draw_completion(completion)
    draw_outcomes(outcome_shares(attempts[is_large]), outcome_shares(attempts[~is_large]))
    draw_cross_track(ptr, pooled)
    print(f"wrote 5 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

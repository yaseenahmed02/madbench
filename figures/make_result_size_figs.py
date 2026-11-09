"""Build the Track B figures that relate completion to gold result size.

Reads ``data/track_b/corpus.csv`` and ``data/track_b/gold_rows.csv`` (row count of each
gold result per question and scale factor; the row counts agree on all three
backends). Writes two PDFs into this directory and prints every plotted number.

Run from the repository root:
    uv run --no-project --with pandas --with matplotlib python figures/make_result_size_figs.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from make_paper_figs import FIG_DIR, GRID_GREY, MODELS, SERVER_MARKERS, SERVERS, load_track_b
from make_track_a_figs import COLUMN_WIDTH_IN, COUPLINGS, SMALL_FONT_PT, apply_figure_style, save_figure

GOLD_ROWS_CSV = FIG_DIR.parent / "data" / "track_b" / "gold_rows.csv"
SERVER_COLOURS = {"mcp-alchemy": "#0072B2", "dbhub": "#D55E00", "anthropic-pg": "#CC79A7"}
SIZE_BIN_EDGES = (0, 1, 10, 100, 1_000, 10_000, 10_000_000)
SIZE_BIN_LABELS = ("1", "10", "100", "1k", "10k", ">10k")
REJECTED_OUTCOME = "agent_error"
MINIMUM_CELLS_PER_MEDIAN = 5


def load_attempts_with_sizes() -> pd.DataFrame:
    """Attach the gold row count to every attempt and add a correctness flag."""
    gold_rows = pd.read_csv(GOLD_ROWS_CSV)
    attempts = load_track_b().merge(gold_rows, on=["query", "scale_factor"], validate="many_to_one")
    attempts["is_correct"] = attempts["outcome"] == "correct"
    attempts["size_bin"] = pd.cut(attempts["gold_rows"], SIZE_BIN_EDGES, labels=SIZE_BIN_LABELS)
    return attempts


def _draw_token_growth(ax, attempts: pd.DataFrame) -> None:
    # Rejected attempts log zero tokens, so they would drag every median to nothing.
    answered = attempts[(attempts["outcome"] != REJECTED_OUTCOME) & (attempts["total_input_tokens"] > 0)]
    medians = answered.groupby(["server", "size_bin"], observed=False)["total_input_tokens"].agg(["median", "size"])
    print(medians.to_string())
    positions = np.arange(len(SIZE_BIN_LABELS))
    for server in SERVERS:
        series = medians.loc[server].reindex(SIZE_BIN_LABELS)
        values = series["median"].where(series["size"] >= MINIMUM_CELLS_PER_MEDIAN)
        ax.plot(positions, values, marker=SERVER_MARKERS[server], markersize=3.2, linewidth=1.0,
                color=SERVER_COLOURS[server], markeredgecolor="white", markeredgewidth=0.25, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(positions, SIZE_BIN_LABELS)
    ax.set_xlabel("Rows in the gold result, up to")
    ax.set_ylabel("Median input tokens")
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.5, zorder=0)
    handles = [Line2D([], [], color=SERVER_COLOURS[s], marker=SERVER_MARKERS[s], markersize=3.2,
                      linewidth=1.0, label=s) for s in SERVERS]
    ax.legend(handles=handles, frameon=False, loc="upper left", handlelength=1.4, handletextpad=0.3,
              labelspacing=0.2, fontsize=SMALL_FONT_PT, borderaxespad=0.2)


def draw_result_size(attempts: pd.DataFrame) -> None:
    """Fig: median input tokens against gold result size, per server."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.5))
    _draw_token_growth(ax, attempts)
    save_figure(fig, FIG_DIR / "track_b_result_size.pdf")


def _question_order(attempts: pd.DataFrame) -> list[str]:
    largest = attempts.groupby("query")["gold_rows"].max()
    return list(largest.sort_values(kind="stable").index)


def _heatmap_rows(attempts: pd.DataFrame, questions: list[str]) -> tuple[np.ndarray, list[str]]:
    rows, labels = [], []
    for server, backend, label in COUPLINGS:
        cell = attempts[(attempts["server"] == server) & (attempts["backend"] == backend)]
        rows.append(cell.groupby("query")["is_correct"].mean().reindex(questions).to_numpy() * 100)
        labels.append(label)
    rows.append(np.full(len(questions), np.nan))
    labels.append("")
    for model in MODELS:
        cell = attempts[attempts["model"] == model.key]
        rows.append(cell.groupby("query")["is_correct"].mean().reindex(questions).to_numpy() * 100)
        labels.append(model.label)
    return np.vstack(rows), labels


def draw_question_heatmap(attempts: pd.DataFrame) -> None:
    """Fig: completion per question, against couplings and against models."""
    questions = _question_order(attempts)
    grid, labels = _heatmap_rows(attempts, questions)
    colour_map = LinearSegmentedColormap.from_list("completion", plt.get_cmap("Blues")(np.linspace(0.18, 1.0, 256)))
    colour_map.set_bad("white")
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.7))
    image = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=colour_map, vmin=0, vmax=100,
                      interpolation="nearest")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    large_start = len(questions) - 8
    ax.set_xlabel("Questions, ordered by largest gold result")
    ax.text((large_start - 1) / 2, -0.75, "42 questions", ha="center", va="bottom", fontsize=SMALL_FONT_PT)
    ax.text(large_start + 3.5, -0.75, "8 large", ha="center", va="bottom", fontsize=SMALL_FONT_PT)
    ax.axvline(large_start - 0.5, color="black", linewidth=0.6, linestyle="--")
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.015, ticks=[0, 50, 100])
    bar.ax.tick_params(labelsize=SMALL_FONT_PT, length=2)
    bar.outline.set_visible(False)
    bar.set_label("Completion (%)", fontsize=SMALL_FONT_PT, labelpad=1)
    print("column before line:", questions[large_start - 1], "first large:", questions[large_start])
    save_figure(fig, FIG_DIR / "track_b_questions.pdf")


def main() -> None:
    apply_figure_style()
    attempts = load_attempts_with_sizes()
    draw_result_size(attempts)
    draw_question_heatmap(attempts)


if __name__ == "__main__":
    main()

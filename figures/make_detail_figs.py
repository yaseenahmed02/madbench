"""Build the per-query Track A and per-model Track B detail figures.

Writes ``track_a_query_heatmap.pdf`` and ``track_b_models.pdf`` into this
directory and prints every plotted number.

Run from the repository root:
    uv run --no-project --with pandas --with matplotlib python figures/make_detail_figs.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from make_paper_figs import FIG_DIR, MODELS, OUTCOME_CLASSES, TRACK_A_SCALE_FACTORS, load_track_b
from make_track_a_figs import (
    COLUMN_WIDTH_IN, SMALL_FONT_PT, TRACK_A_COUPLINGS, apply_figure_style, load_corpus, save_figure,
)

QUERIES = ("Q1", "Q3", "Q6", "Q9", "Q21")
HEATMAP_MIN_PTR = 0.003
HEATMAP_MAX_PTR = 8.0
DARK_CELL_PTR = 0.5
MINIMUM_LABELLED_SHARE = 8
TEXT_WIDTH_IN = 7.0


def query_ptr_grid(corpus: pd.DataFrame, scale_factor: int) -> np.ndarray:
    """Median of the cold and warm cell medians, one row per coupling, one column per query."""
    cells = corpus[corpus["scale_factor"] == scale_factor]
    cell_medians = cells.groupby(["server", "backend", "query", "cold_warm"])["ptr"].median()
    per_query = cell_medians.groupby(["server", "backend", "query"]).median()
    return np.array([[per_query.loc[(server, backend, query)] for query in QUERIES]
                     for server, backend, _ in TRACK_A_COUPLINGS])


def _annotate_ptr(ax, grid: np.ndarray) -> None:
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            text = f"{value:.2f}" if value >= 0.1 else f"{value:.3f}"
            ink = "white" if value >= DARK_CELL_PTR else "black"
            ax.text(column, row, text, ha="center", va="center", fontsize=SMALL_FONT_PT, color=ink)


def draw_query_heatmap(corpus: pd.DataFrame) -> None:
    """Fig: median PTR per coupling and query at each scale factor."""
    colour_map = LinearSegmentedColormap.from_list("ptr", plt.get_cmap("Blues")(np.linspace(0.05, 1.0, 256)))
    norm = LogNorm(HEATMAP_MIN_PTR, HEATMAP_MAX_PTR)
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(TEXT_WIDTH_IN, 1.3))
    for ax, scale_factor in zip(axes, TRACK_A_SCALE_FACTORS):
        grid = query_ptr_grid(corpus, scale_factor)
        print(f"SF{scale_factor}\n", np.round(grid, 3))
        ax.imshow(grid, aspect="auto", cmap=colour_map, norm=norm, interpolation="nearest")
        _annotate_ptr(ax, grid)
        ax.set_xticks(range(len(QUERIES)), QUERIES)
        ax.tick_params(axis="both", length=0, labelsize=SMALL_FONT_PT)
        ax.spines[:].set_visible(False)
        ax.set_title(f"SF{scale_factor}", fontsize=SMALL_FONT_PT, pad=2)
    axes[0].set_yticks(range(len(TRACK_A_COUPLINGS)), [name for *_, name in TRACK_A_COUPLINGS])
    save_figure(fig, FIG_DIR / "track_a_query_heatmap.pdf")


def draw_model_outcomes(attempts: pd.DataFrame) -> None:
    """Fig: 100% stacked outcomes per model over all 50 questions."""
    counts = attempts.groupby(["model", "outcome"]).size().unstack(fill_value=0)
    print(counts.to_string())
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.45))
    for row, model in enumerate(MODELS):
        left = 0.0
        total = counts.loc[model.key].sum()
        for key, legend, colour, hatch in OUTCOME_CLASSES:
            share = 100 * counts.loc[model.key].get(key, 0) / total
            ax.barh(row, share, left=left, height=0.62, color=colour, hatch=hatch, edgecolor="white",
                    linewidth=0.3, label=legend)
            if share >= MINIMUM_LABELLED_SHARE:
                ink = "white" if key == "correct" else "black"
                ax.text(left + share / 2, row, f"{share:.1f}", ha="center", va="center", color=ink,
                        fontsize=SMALL_FONT_PT,
                        bbox={"boxstyle": "square,pad=0.05", "facecolor": colour, "linewidth": 0} if hatch else None)
            left += share
    ax.set_yticks(range(len(MODELS)), [model.label for model in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of attempts (%)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), ncol=3, frameon=False, loc="outside upper center",
               handlelength=1.6, columnspacing=0.8)
    save_figure(fig, FIG_DIR / "track_b_models.pdf")


def main() -> None:
    apply_figure_style()
    draw_query_heatmap(load_corpus())
    draw_model_outcomes(load_track_b())


if __name__ == "__main__":
    main()

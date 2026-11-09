"""Build the Track A phase-share figure.

Reads the 5,378-trace Track A corpus from ``data/track_a/traces.csv`` and writes
``figures/track_a_phase_shares.pdf``: the median composition of one Q6
interaction (warm) at SF1 and SF10 for each server-backend coupling.

Run from the repository root:
    uv run --no-project --with pandas --with matplotlib python figures/make_track_a_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A_TRACES = REPO_ROOT / "data" / "track_a" / "traces.csv"
OUTPUT_PDF = Path(__file__).resolve().parent / "track_a_phase_shares.pdf"
EXPECTED_TRACE_COUNT = 5378
INNER_PHASES = ("T_disc_ms", "T_life_ms", "T_hand_ms", "T_ser_ms", "T_parse_ms", "T_xport_ms")

COUPLINGS = (  # (server, backend, label), top to bottom
    ("mcp-alchemy", "duckdb", "mcp-alchemy / DuckDB"),
    ("mcp-alchemy", "mysql", "mcp-alchemy / MySQL"),
    ("dbhub", "mysql", "dbhub / MySQL"),
    ("mcp-alchemy", "postgres", "mcp-alchemy / PG"),
    ("dbhub", "postgres", "dbhub / PG"),
    ("anthropic-pg", "postgres", "anthropic-pg / PG"),
)
# Track A's DuckDB cells have no MCP server, so its row carries a reference label.
TRACK_A_COUPLINGS = tuple(
    (server, backend, "DuckDB, in process" if backend == "duckdb" else label)
    for server, backend, label in COUPLINGS
)
SCALE_FACTORS = (1, 10)
QUERY = "Q6"
STATE = "warm"

# Okabe-Ito colours, plus hatches so the stack survives grayscale printing.
SEGMENTS = (  # (column, legend label, colour, hatch)
    ("T_db_ms", r"$T_{\mathrm{db}}$", "#0072B2", ""),
    ("T_xport_ms", "Transport", "#E69F00", "////"),
    ("T_life_ms", "Lifecycle", "#56B4E9", ""),
    ("tail_ms", "Teardown + residual", "#D55E00", "xxxx"),
    ("other_ms", "Other phases", "#999999", ""),
)


# Shared look for every figure: IEEE body serif, 7 pt minimum text,
# Okabe-Ito colours, thin black axes with the top and right spines removed.
COLUMN_WIDTH_IN = 3.5
FULL_WIDTH_IN = 7.16
BASE_FONT_PT = 7.5
SMALL_FONT_PT = 7.0
SERIF_FONTS = ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]


def apply_figure_style() -> None:
    """Set the rcParams that every figure shares."""
    plt.rcParams.update({
        "font.size": BASE_FONT_PT, "font.family": "serif", "font.serif": SERIF_FONTS,
        "mathtext.fontset": "stix", "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": SMALL_FONT_PT, "ytick.labelsize": SMALL_FONT_PT,
        "legend.fontsize": SMALL_FONT_PT, "hatch.linewidth": 0.4,
        "pdf.fonttype": 42, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.01, "figure.constrained_layout.w_pad": 0.01,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save at the exact figsize, so the PDF is placed at 100% and text keeps its size.

    Constrained layout fits labels and legends inside the canvas; a tight
    bounding box would instead grow the page and shrink the fonts in LaTeX.
    The creation date is left out so the same data gives byte-identical PDFs.
    """
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


class CorpusSizeError(RuntimeError):
    """Raised when the corpus does not have the expected size."""


def load_corpus() -> pd.DataFrame:
    """Load and de-duplicate the Track A per-trace rows."""
    corpus = pd.read_csv(TRACK_A_TRACES)
    if len(corpus) != EXPECTED_TRACE_COUNT:
        raise CorpusSizeError(
            f"Expected {EXPECTED_TRACE_COUNT} Track A traces, read {len(corpus)} "
            f"from {TRACK_A_TRACES}."
        )
    inner_sum = corpus[list(INNER_PHASES)].sum(axis=1)
    corpus["tail_ms"] = corpus["T_total_ms"] - corpus["T_db_ms"] - inner_sum
    corpus["other_ms"] = corpus[["T_disc_ms", "T_hand_ms", "T_ser_ms", "T_parse_ms"]].sum(axis=1)
    corpus["ptr"] = (corpus["T_total_ms"] - corpus["T_db_ms"]) / corpus["T_db_ms"]
    return corpus


def summarise_cell(corpus: pd.DataFrame, server: str, backend: str, scale_factor: int) -> dict:
    """Median of each segment over the 30 repetitions, as shares of their sum."""
    cell = corpus[
        (corpus["server"] == server) & (corpus["backend"] == backend)
        & (corpus["query"] == QUERY) & (corpus["scale_factor"] == scale_factor)
        & (corpus["cold_warm"] == STATE)
    ]
    medians = {column: max(0.0, cell[column].median()) for column, *_ in SEGMENTS}
    total = sum(medians.values())
    shares = {column: 100.0 * value / total for column, value in medians.items()}
    return {"shares": shares, "ptr": cell["ptr"].median(), "n": len(cell)}


def draw_figure(corpus: pd.DataFrame) -> None:
    """Draw the horizontal 100% stacked bars and save the PDF."""
    apply_figure_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.75))
    labels, positions = [], []
    y = 0.0
    for server, backend, name in TRACK_A_COUPLINGS:
        for scale_factor in SCALE_FACTORS:
            cell = summarise_cell(corpus, server, backend, scale_factor)
            left = 0.0
            for column, legend, colour, hatch in SEGMENTS:
                width = cell["shares"][column]
                ax.barh(y, width, left=left, height=0.8, color=colour, hatch=hatch,
                        edgecolor="white", linewidth=0.3,
                        label=legend if y == 0.0 else None)
                left += width
            ax.text(101.5, y, f"{cell['ptr']:.2f}", va="center", fontsize=SMALL_FONT_PT)
            labels.append(f"{name}, SF{scale_factor}")
            positions.append(y)
            y += 1.0
        y += 0.5
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of median interaction time (%)")
    ax.text(101.5, -1.0, "PTR", fontsize=SMALL_FONT_PT, fontweight="bold")
    ax.legend(ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.38, 1.04), handlelength=1.6, columnspacing=0.8)
    save_figure(fig, OUTPUT_PDF)


def print_shares(corpus: pd.DataFrame) -> None:
    """Print the plotted numbers so the prose can cite them."""
    for server, backend, name in TRACK_A_COUPLINGS:
        for scale_factor in SCALE_FACTORS:
            cell = summarise_cell(corpus, server, backend, scale_factor)
            shares = ", ".join(f"{c}={v:.0f}%" for c, v in cell["shares"].items())
            print(f"{name:22s} SF{scale_factor:<2} n={cell['n']} PTR={cell['ptr']:.3f} {shares}")


def main() -> None:
    corpus = load_corpus()
    print_shares(corpus)
    draw_figure(corpus)
    print(f"wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()

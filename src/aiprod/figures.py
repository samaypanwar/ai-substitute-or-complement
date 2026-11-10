"""Publication-style figures (IEEE / ICML two-column conventions).

Every figure is saved twice into ``output/figures``: a 400-dpi PNG for quick
viewing and a vector PDF for the LaTeX paper. Widths follow the two-column
conventions: one column is 3.5 in, the full text width 7.0 in.

Colors come from a colorblind-validated categorical palette and are assigned
by *meaning*, never by plotting order: the AI era is always blue, the IT boom
is always orange.
"""

import matplotlib.pyplot as plt
import pandas as pd

from aiprod.config import FIGURES

COLUMN = 3.5
FULL = 7.0

AI_BLUE = "#2a78d6"
IT_ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

ERA_COLORS = {"AI": AI_BLUE, "IT": IT_ORANGE}


def apply_style() -> None:
    """Set matplotlib defaults for small, print-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save ``fig`` as ``output/figures/<name>.png`` and ``.pdf``, then close it.

    Args:
        fig (plt.Figure): Figure to save.
        name (str): File stem, e.g. ``"c2_event_study"``.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


def mark_event(ax: plt.Axes, x: float | pd.Timestamp, label: str) -> None:
    """Draw a thin dashed vertical reference line with a small label at the top.

    Args:
        ax (plt.Axes): Target axes.
        x (float | pd.Timestamp): Position on the x axis.
        label (str): Short annotation, e.g. ``"ChatGPT"``.
    """
    ax.axvline(x, color=MUTED, linestyle="--", linewidth=0.8)
    ax.annotate(label, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -8),
                textcoords="offset points", fontsize=6.5, color=MUTED)


def zero_line(ax: plt.Axes) -> None:
    """Draw the y = 0 reference line."""
    ax.axhline(0, color=MUTED, linewidth=0.7)


def event_study_panel(ax: plt.Axes, study: pd.DataFrame, color: str, x: str = "year") -> None:
    """Plot event-study coefficients with 95% confidence bands.

    Args:
        ax (plt.Axes): Target axes.
        study (pd.DataFrame): Columns ``x``, ``gamma`` and ``se``.
        color (str): Line and band color.
        x (str): Name of the time column.
    """
    low = study["gamma"] - 1.96 * study["se"]
    high = study["gamma"] + 1.96 * study["se"]
    ax.fill_between(study[x], low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(study[x], study["gamma"], color=color, marker="o")
    zero_line(ax)

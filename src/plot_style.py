"""Shared matplotlib style for the figures in results/ (light theme, PNG)."""
from __future__ import annotations

import matplotlib.pyplot as plt

COLORS = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#e4e3df",
    "reference": "#8a8984",   # 45-degree line, snapshot markers
    "series1": "#2a78d6",     # blue: home team / single series
    "series2": "#eb6834",     # orange: away team
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text_secondary"],
        "axes.titlecolor": COLORS["text_primary"],
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": COLORS["text_secondary"],
        "ytick.color": COLORS["text_secondary"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.labelcolor": COLORS["text_primary"],
        "font.family": "DejaVu Sans",
    })

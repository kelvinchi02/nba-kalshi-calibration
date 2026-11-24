"""Stage 9 horizon comparison (ANALYSIS_PLAN.md section 9.2).

Library for notebooks/04_horizons.ipynb.

Sample: main-sample games with p_home present at all four horizons (24h, 6h, 1h, 0h), so
every horizon uses the same games. At each horizon: Brier score, logistic recalibration and
the LR test, as in Stage 8. A paired block bootstrap by game date compares each horizon's
Brier score with tipoff.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stats_tools
from calibration import assign_bins, calibration_bins, logistic_table
from csv_io import read_csv
from plot_style import COLORS, apply_style
from stats_tools import B, SEED, brier, fit_logit, lr_test

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
RESULTS_DIR = REPO_ROOT / "results"

HORIZONS = (24, 6, 1, 0)          # hours before tipoff, earliest first
EARLIER = (24, 6, 1)              # the horizons compared with tipoff
MAX_SPREAD = 0.05
# One hue, light to dark: the closer to tipoff, the darker the line.
HORIZON_COLORS = {24: "#86b6ef", 6: "#3987e5", 1: "#1c5cab", 0: "#0d366b"}


def label(h: int) -> str:
    return "tipoff" if h == 0 else f"{h}h"


# ---------------------------------------------------------------- sample

def main_sample_rows() -> pd.DataFrame:
    """Every main-sample snapshot row (one per game per horizon)."""
    sn = read_csv(SNAP_PATH)
    return sn[sn["main_sample"]]


def complete_games(sn: pd.DataFrame) -> pd.DataFrame:
    """Rows of the games that have p_home at every horizon, sorted by game and horizon."""
    rows = sn[sn["horizon_h"].isin(HORIZONS)]
    priced = rows.dropna(subset=["p_home"]).groupby("game_id")["horizon_h"].nunique()
    keep = priced.index[priced == len(HORIZONS)]
    d = rows[rows["game_id"].isin(keep)]
    d = d[["game_id", "game_date", "home_team", "away_team", "horizon_h", "p_home",
           "home_win", "max_spread", "price_source"]].copy()
    d["home_win"] = d["home_win"].astype(int)
    if not d["p_home"].between(0, 1, inclusive="neither").all():
        raise ValueError("p_home must lie strictly between 0 and 1 for the logit")
    return d.sort_values(["game_id", "horizon_h"], ignore_index=True)


def sample_counts(sn: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Games in the main sample, with a price per horizon, and with all four."""
    rows = [{"group": "main sample", "games": sn["game_id"].nunique()}]
    for h in HORIZONS:
        priced = sn[(sn["horizon_h"] == h) & sn["p_home"].notna()]
        rows.append({"group": f"p_home present at {label(h)}", "games": len(priced)})
    rows.append({"group": "p_home present at all four horizons",
                 "games": d["game_id"].nunique()})
    return pd.DataFrame(rows)


def price_source_counts(d: pd.DataFrame) -> pd.DataFrame:
    """Games per price_source at each horizon."""
    out = d.groupby(["horizon_h", "price_source"]).size().unstack(fill_value=0)
    out = out.reindex(list(HORIZONS)).reset_index()
    out.columns.name = None
    return out


def spread_filter(d: pd.DataFrame, max_spread: float = MAX_SPREAD) -> pd.DataFrame:
    """Drop every game whose max_spread is above the limit, or missing, at any horizon."""
    bad = d["max_spread"].isna() | (d["max_spread"] > max_spread)
    return d[~d["game_id"].isin(d.loc[bad, "game_id"])].reset_index(drop=True)


# ---------------------------------------------------------------- per-horizon statistics

def horizon_metrics(d: pd.DataFrame, sample: str) -> pd.DataFrame:
    """One row per horizon: Brier score, a and b with clustered tests, and the LR test."""
    rows = []
    for h in HORIZONS:
        g = d[d["horizon_h"] == h]
        p, y = g["p_home"], g["home_win"]
        logi = logistic_table(fit_logit(p, y, groups=g["home_team"]),
                              g["home_team"].nunique()).set_index("term")
        lr = lr_test(p, y)
        a, b = logi.loc["intercept_a"], logi.loc["slope_b"]
        rows.append({"sample": sample, "horizon_h": h, **brier(p, y),
                     "intercept_a": a["estimate"], "a_se_cluster": a["se_cluster"],
                     "a_p_value": a["p_value"],
                     "slope_b": b["estimate"], "b_se_cluster": b["se_cluster"],
                     "b_p_value": b["p_value"],
                     "lr_stat": lr["lr_stat"], "lr_p_value": lr["p_value"]})
    return pd.DataFrame(rows)


def wide_prices(d: pd.DataFrame) -> pd.DataFrame:
    """One row per game: game_date, home_win and one price column per horizon."""
    prices = d.pivot(index="game_id", columns="horizon_h", values="p_home")
    info = d.drop_duplicates("game_id").set_index("game_id")[["game_date", "home_win"]]
    return info.join(prices).sort_index()


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def paired_brier_bootstrap(d: pd.DataFrame, n_boot: int = B, seed: int = SEED) -> pd.DataFrame:
    """Block bootstrap by date. Each replicate scores all four horizons on the same games."""
    w = wide_prices(d)
    y = w["home_win"].to_numpy(dtype=float)
    prices = {h: w[h].to_numpy(dtype=float) for h in HORIZONS}

    def replicate(idx: np.ndarray) -> dict:
        bs = {h: brier_score(prices[h][idx], y[idx]) for h in HORIZONS}
        rec = {f"brier_{h}h": bs[h] for h in HORIZONS}
        rec |= {f"brier_diff_{h}h_minus_0h": bs[h] - bs[0] for h in EARLIER}
        return rec

    return stats_tools.block_bootstrap(w["game_date"], replicate, n_boot=n_boot, seed=seed)


def bootstrap_table(d: pd.DataFrame, boot: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Percentile intervals for each horizon's Brier score and each difference to tipoff."""
    w = wide_prices(d)
    y = w["home_win"].to_numpy(dtype=float)
    bs = {h: brier_score(w[h].to_numpy(dtype=float), y) for h in HORIZONS}
    point = {f"brier_{h}h": bs[h] for h in HORIZONS}
    point |= {f"brier_diff_{h}h_minus_0h": bs[h] - bs[0] for h in EARLIER}
    out = stats_tools.percentile_summary(boot, point, w["game_date"].nunique())
    out.insert(0, "sample", sample)
    return out


def brier_diff_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The paired differences BS_h - BS_0 with their intervals, one row per earlier horizon."""
    s = summary.set_index("statistic")
    rows = []
    for h in EARLIER:
        r = s.loc[f"brier_diff_{h}h_minus_0h"]
        rows.append({"sample": r["sample"], "horizon_h": h,
                     "brier_h": s.loc[f"brier_{h}h", "estimate"],
                     "brier_0h": s.loc["brier_0h", "estimate"], "brier_diff": r["estimate"],
                     "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"],
                     "interval_excludes_0": int(r["boot_lo"] > 0 or r["boot_hi"] < 0)})
    return pd.DataFrame(rows)


def horizon_calibration_bins(d: pd.DataFrame) -> pd.DataFrame:
    """10 equal-count bins per horizon, built as in plan section 2.1 from that horizon's prices."""
    frames = []
    for h in HORIZONS:
        g = d[d["horizon_h"] == h].sort_values(["p_home", "game_id"]).reset_index(drop=True)
        g["bin"] = assign_bins(len(g))
        bins = calibration_bins(g)
        bins.insert(0, "horizon_h", h)
        frames.append(bins)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- figures

def brier_figure(summary: pd.DataFrame, n_games: int, path: Path) -> plt.Figure:
    """Brier score at each horizon with its bootstrap interval."""
    s = summary.set_index("statistic")
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(HORIZONS))
    est = np.array([s.loc[f"brier_{h}h", "estimate"] for h in HORIZONS])
    lo = np.array([s.loc[f"brier_{h}h", "boot_lo"] for h in HORIZONS])
    hi = np.array([s.loc[f"brier_{h}h", "boot_hi"] for h in HORIZONS])
    ax.errorbar(x, est, yerr=np.vstack([est - lo, hi - est]), fmt="o", ms=8,
                color=COLORS["series1"], ecolor=COLORS["series1"], elinewidth=2, capsize=0,
                mec=COLORS["surface"], mew=2, zorder=3)
    for xi, v in zip(x, est):
        ax.annotate(f"{v:.4f}", (xi, v), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=COLORS["text_secondary"])
    ax.set_xticks(x, [f"{h}h before" if h else "tipoff" for h in HORIZONS])
    ax.set_xlim(-0.5, len(HORIZONS) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title(f"Brier score by horizon, {n_games:,} games, 95% bootstrap intervals",
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def calibration_figure(bins: pd.DataFrame, n_games: int, path: Path) -> plt.Figure:
    """Calibration curves of the four horizons on one plot."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], lw=1.2, ls=(0, (4, 3)), zorder=1,
            label="Perfect calibration")
    for h in HORIZONS:
        b = bins[bins["horizon_h"] == h]
        ax.plot(b["mean_p"], b["win_rate"], marker="o", ms=6, lw=2, color=HORIZON_COLORS[h],
                mec=COLORS["surface"], mew=1.2, zorder=3,
                label="At tipoff" if h == 0 else f"{h}h before tipoff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Kalshi home-win price (bin mean of p_home)")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title(f"Calibration by horizon: {n_games:,} games, 10 equal-count bins each",
                 loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig

"""Stage 8: calibration at tipoff (horizon 0), exactly as in ANALYSIS_PLAN.md.

Library for notebooks/02_calibration.ipynb, which runs it and writes results/.
Main sample: main_sample games at horizon_h == 0, price p_home, outcome home_win.

  2.1 calibration plot: 10 equal-count bins, 95% Wilson intervals
  2.2 Brier score against a constant baseline (the sample home-win rate)
  2.3 logistic regression of home_win on logit(p_home), SEs clustered by home team
  2.4 likelihood ratio test of intercept 0 and slope 1
  3   block bootstrap by game date (B = 2000, seed 4981)

The statistics themselves live in stats_tools.py. This module holds what is specific to
the calibration tables and figure.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import stats_tools
from csv_io import read_csv
from plot_style import COLORS, apply_style
from stats_tools import (B, SEED, Z, brier, fit_logit, logit, loglik_restricted,  # noqa: F401
                         lr_test, percentile_summary, wilson)

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
RESULTS_DIR = REPO_ROOT / "results"

HORIZON = 0
N_BINS = 10


# ---------------------------------------------------------------- sample

def load_sample() -> pd.DataFrame:
    """One row per main-sample game at horizon 0. Drops (and reports) missing p_home."""
    sn = read_csv(SNAP_PATH)
    d = sn[(sn["horizon_h"] == HORIZON) & sn["main_sample"]]
    n_missing = int(d["p_home"].isna().sum())
    if n_missing:
        print(f"Dropping {n_missing} rows with p_home missing")
    d = d.dropna(subset=["p_home"])
    d = d[["game_id", "game_date", "home_team", "away_team", "p_home", "home_win"]]
    d = d.sort_values(["p_home", "game_id"]).reset_index(drop=True)
    d["home_win"] = d["home_win"].astype(int)
    d["bin"] = assign_bins(len(d))
    return d


def assign_bins(n: int, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-count bins for n rows already sorted by (p_home, game_id)."""
    return np.arange(n) * n_bins // n


# ---------------------------------------------------------------- tables

def calibration_bins(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, g in d.groupby("bin"):
        k, m = int(g["home_win"].sum()), len(g)
        lo, hi = wilson(k, m)
        rows.append({"bin": b + 1, "n": m, "p_min": g["p_home"].min(),
                     "p_max": g["p_home"].max(), "mean_p": g["p_home"].mean(),
                     "home_wins": k, "win_rate": k / m, "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(rows)


def logistic_table(fit, n_clusters: int) -> pd.DataFrame:
    est, se = fit.params, fit.bse
    rows = []
    for i, (term, null) in enumerate([("intercept_a", 0.0), ("slope_b", 1.0)]):
        z = (est[i] - null) / se[i]
        rows.append({"term": term, "estimate": est[i], "se_cluster": se[i],
                     "ci_lo": est[i] - Z * se[i], "ci_hi": est[i] + Z * se[i],
                     "null_value": null, "z": z, "p_value": 2 * stats.norm.sf(abs(z)),
                     "n": int(fit.nobs), "n_clusters": n_clusters})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- bootstrap

def replicate_statistics(pb, yb, bb, n_bins: int = N_BINS,
                         tolerate_failed_fit: bool = False) -> dict:
    """The statistics of one bootstrap replicate: Brier, a, b, the LR statistic and the
    win rate per bin.

    With tolerate_failed_fit (small samples, plan 10.1), a replicate whose model cannot be
    fitted keeps its Brier scores and bin rates and gets missing a, b and LR.
    """
    rec = dict(brier(pb, yb))
    try:
        with warnings.catch_warnings():
            if tolerate_failed_fit:
                warnings.simplefilter("error")      # separation or no convergence: no fit
            fit = fit_logit(pb, yb)
        rec |= {"intercept_a": fit.params[0], "slope_b": fit.params[1],
                "lr_stat": lr_test(pb, yb, fit)["lr_stat"]}
    except Exception:  # noqa: BLE001 - re-raised unless the caller asked to tolerate it
        if not tolerate_failed_fit:
            raise
        rec |= {"intercept_a": np.nan, "slope_b": np.nan, "lr_stat": np.nan}
    for k in range(n_bins):
        mask = bb == k
        rec[f"bin{k + 1}_win_rate"] = yb[mask].mean() if mask.any() else np.nan
    return rec


def block_bootstrap(d: pd.DataFrame, n_boot: int = B, seed: int = SEED, n_bins: int = N_BINS,
                    tolerate_failed_fit: bool = False) -> pd.DataFrame:
    """Resample whole game dates with replacement. Bins keep their full-sample assignment."""
    p, y, bins = d["p_home"].to_numpy(), d["home_win"].to_numpy(), d["bin"].to_numpy()

    def replicate(idx: np.ndarray) -> dict:
        return replicate_statistics(p[idx], y[idx], bins[idx], n_bins, tolerate_failed_fit)

    return stats_tools.block_bootstrap(d["game_date"], replicate, n_boot=n_boot, seed=seed)


def point_estimates(bs: dict, fit, lr: dict, bins: pd.DataFrame) -> dict:
    """The full-sample value of every statistic the bootstrap recomputes."""
    point = {k: bs[k] for k in ("brier", "brier_baseline", "brier_diff", "brier_skill")}
    point |= {"intercept_a": fit.params[0], "slope_b": fit.params[1], "lr_stat": lr["lr_stat"]}
    point |= {f"bin{k + 1}_win_rate": r for k, r in enumerate(bins["win_rate"])}
    return point


def bootstrap_summary(boot: pd.DataFrame, point: dict, n_dates: int) -> pd.DataFrame:
    return percentile_summary(boot, point, n_dates, seed=SEED)


def add_bootstrap_intervals(bins: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Append each bin's bootstrap interval for the win rate (boot_lo, boot_hi)."""
    ci = summary.set_index("statistic")
    out = bins.copy()
    out["boot_lo"] = [ci.loc[f"bin{k}_win_rate", "boot_lo"] for k in out["bin"]]
    out["boot_hi"] = [ci.loc[f"bin{k}_win_rate", "boot_hi"] for k in out["bin"]]
    return out


# ---------------------------------------------------------------- figure

def calibration_plot(bins: pd.DataFrame, bs: dict, lr: dict, logi: pd.DataFrame,
                     path: Path, title: str | None = None) -> plt.Figure:
    """Save the calibration plot to `path` and return the figure."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], lw=1.2, ls=(0, (4, 3)), zorder=1,
            label="Perfect calibration")
    yerr = np.vstack([bins["win_rate"] - bins["wilson_lo"], bins["wilson_hi"] - bins["win_rate"]])
    ax.errorbar(bins["mean_p"], bins["win_rate"], yerr=yerr, fmt="o", ms=8,
                color=COLORS["series1"], ecolor=COLORS["series1"], elinewidth=2, capsize=0,
                mec=COLORS["surface"], mew=2, zorder=3,
                label="Observed home-win rate, 95% Wilson interval")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Kalshi home-win price at tipoff (bin mean of p_home)")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title(title or f"Calibration at tipoff: {bs['n']:,} games in 10 equal-count bins",
                 loc="left")
    a, b = logi.set_index("term")["estimate"]
    ax.text(0.98, 0.04,
            f"Brier {bs['brier']:.4f} (constant baseline {bs['brier_baseline']:.4f})\n"
            f"Recalibration a = {a:.3f}, b = {b:.3f}\n"
            f"LR test of a = 0, b = 1: p = {lr['p_value']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=COLORS["text_secondary"])
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig

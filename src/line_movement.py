"""Stage 10 line movement (ANALYSIS_PLAN.md section 10.3).

Library for notebooks/06_line_movement.ipynb.

Sample: main-sample games with p_home at 24h and at tipoff. For the home side,

    move = logit(p at tipoff) - logit(p at h hours before tipoff)

and the model is

    logit P(home_win = 1) = a + b * logit(p at tipoff) + d * move

with standard errors clustered by home team and a block bootstrap by game date. This is
the model of plan 9.1 with the move in place of the popularity gap, so the fit, the
plain-terms effect and the bootstrap come from popularity.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import all_tests
import popularity
from calibration import assign_bins
from horizons import HORIZONS, MAX_SPREAD, main_sample_rows
from plot_style import COLORS, apply_style
from stats_tools import B, SEED, logit, wilson

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PART = "line_movement"
MAIN_HORIZON = 24
SECONDARY_HORIZONS = (6, 1)
N_BINS = 10
HIST_BIN_WIDTH = 0.1
_RENAME = {"gap_c": "move_d", "gap_sd": "move_sd", "c_per_sd": "d_per_sd",
           "c_per_sd_boot_lo": "d_per_sd_boot_lo", "c_per_sd_boot_hi": "d_per_sd_boot_hi"}


# ---------------------------------------------------------------- sample

def wide_table(sn: pd.DataFrame) -> pd.DataFrame:
    """One row per game: the price and max_spread at each of the four horizons."""
    rows = sn[sn["horizon_h"].isin(HORIZONS)]
    info = (rows.drop_duplicates("game_id")
            .set_index("game_id")[["game_date", "home_team", "away_team", "home_win"]])
    prices = rows.pivot(index="game_id", columns="horizon_h", values="p_home")
    spreads = rows.pivot(index="game_id", columns="horizon_h", values="max_spread")
    out = info.copy()
    out["home_win"] = out["home_win"].astype(int)
    for h in HORIZONS:
        out[f"p_{h}h"] = prices[h].astype(float)
        out[f"max_spread_{h}h"] = spreads[h].astype(float)
    return out.sort_index().reset_index()


def load_wide() -> pd.DataFrame:
    return wide_table(main_sample_rows())


def base_sample(wide: pd.DataFrame) -> pd.DataFrame:
    """Games with a price at 24h and at tipoff."""
    keep = wide[f"p_{MAIN_HORIZON}h"].notna() & wide["p_0h"].notna()
    return wide[keep].reset_index(drop=True)


def spread_ok(wide: pd.DataFrame, max_spread: float = MAX_SPREAD) -> pd.Series:
    """True when max_spread is present and at most the limit at all four horizons (9.2)."""
    spreads = wide[[f"max_spread_{h}h" for h in HORIZONS]]
    return spreads.notna().all(axis=1) & (spreads <= max_spread).all(axis=1)


def move(p_tipoff, p_earlier) -> np.ndarray:
    """logit(p at tipoff) - logit(p earlier). Positive: the home price rose."""
    return logit(p_tipoff) - logit(p_earlier)


def model_frame(d: pd.DataFrame, h: int) -> pd.DataFrame:
    """The games of d with a price h hours before tipoff, laid out for the 9.1 machinery.

    p_home is the tipoff price and gap_move is the move from h hours before to tipoff.
    """
    g = d[d[f"p_{h}h"].notna()].copy()
    g["p_home"] = g["p_0h"]
    g["gap_move"] = move(g["p_0h"], g[f"p_{h}h"])
    if not np.isfinite(g["gap_move"]).all() or not np.isfinite(logit(g["p_home"])).all():
        raise ValueError("prices must lie strictly between 0 and 1")
    return g[["game_id", "game_date", "home_team", "away_team", "home_win", "p_home",
              "gap_move"]].reset_index(drop=True)


def sample_counts(wide: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = [{"group": "main sample", "games": len(wide)},
            {"group": "price at 24h and at tipoff", "games": len(base)}]
    for h in SECONDARY_HORIZONS:
        rows.append({"group": f"of those, price at {h}h",
                     "games": int(base[f"p_{h}h"].notna().sum())})
    rows.append({"group": f"of those, max_spread at most {MAX_SPREAD} at all four horizons",
                 "games": int(spread_ok(base).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model

def analyze_move(frame: pd.DataFrame, model: str, n_boot: int = B,
                 seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Fit, bootstrap and tabulate the move model. Tables as in popularity.analyze_measure,
    with the coefficient on the move named move_d."""
    out = popularity.analyze_measure(frame, "move", n_boot=n_boot, seed=seed)
    tables = {}
    for name, t in out.items():
        t = t.rename(columns={"measure": "model", **_RENAME})
        t["model"] = model
        for col in ("term", "statistic"):
            if col in t.columns:
                t[col] = t[col].replace(_RENAME)
        tables[name] = t
    return tables


def models(base: pd.DataFrame, n_boot: int = B, seed: int = SEED) -> dict[str, dict]:
    """The primary 24h model and the three secondary models, keyed by model name."""
    out = {"24h": analyze_move(model_frame(base, MAIN_HORIZON), "24h", n_boot, seed)}
    for h in SECONDARY_HORIZONS:
        out[f"{h}h"] = analyze_move(model_frame(base, h), f"{h}h", n_boot, seed)
    filtered = base[spread_ok(base)].reset_index(drop=True)
    out["24h_spread_filter"] = analyze_move(model_frame(filtered, MAIN_HORIZON),
                                            "24h_spread_filter", n_boot, seed)
    return out


_MODEL_TEXT = {"24h": "move from 24h to tipoff",
               "6h": "move from 6h to tipoff",
               "1h": "move from 1h to tipoff",
               "24h_spread_filter": "move from 24h to tipoff, after the spread filter"}
_TERM_TEXT = {"intercept_a": ("a", "a = 0"), "slope_b": ("b", "b = 1"),
              "move_d": ("d", "d = 0")}


def tests_table(logistic: pd.DataFrame) -> pd.DataFrame:
    """Every Wald test of the four models in the common Stage 10 layout."""
    rows = []
    for r in logistic.itertuples(index=False):
        short, null_text = _TERM_TEXT[r.term]
        rows.append(all_tests.wald_row(
            PART, f"line_move_{r.model}_{short}",
            f"{short} in the model with the {_MODEL_TEXT[r.model]}", null_text, r.n,
            r.estimate, r.se_cluster, r.p_value, null_value=r.null_value,
            boot=(r.boot_lo, r.boot_hi)))
    return all_tests.table(rows)


# ---------------------------------------------------------------- description

def move_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the move across games."""
    m = frame["gap_move"]
    return pd.DataFrame([{
        "n": len(m), "mean": m.mean(), "sd": m.std(ddof=1), "min": m.min(),
        "p05": m.quantile(0.05), "q25": m.quantile(0.25), "median": m.median(),
        "q75": m.quantile(0.75), "p95": m.quantile(0.95), "max": m.max(),
        "mean_abs": m.abs().mean(),
        "share_positive": (m > 0).mean(), "share_negative": (m < 0).mean(),
        "share_zero": (m == 0).mean()}])


def move_bins(frame: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-count bins by move: home-win rate minus mean tipoff price in each bin.

    The interval is the Wilson interval of the win rate, shifted by the mean tipoff price.
    """
    g = frame.sort_values(["gap_move", "game_id"]).reset_index(drop=True)
    g["bin"] = assign_bins(len(g), n_bins)
    rows = []
    for b, x in g.groupby("bin"):
        k, m = int(x["home_win"].sum()), len(x)
        lo, hi = wilson(k, m)
        mean_p = x["p_home"].mean()
        rows.append({"bin": b + 1, "n": m, "move_min": x["gap_move"].min(),
                     "move_max": x["gap_move"].max(), "mean_move": x["gap_move"].mean(),
                     "mean_p_tipoff": mean_p, "home_wins": k, "win_rate": k / m,
                     "win_rate_minus_price": k / m - mean_p,
                     "wilson_lo_minus_price": lo - mean_p,
                     "wilson_hi_minus_price": hi - mean_p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def distribution_figure(frame: pd.DataFrame, path: Path) -> plt.Figure:
    """Histogram of the 24h move, in bins of 0.1 on the logit scale."""
    m = frame["gap_move"].to_numpy()
    reach = np.ceil(np.abs(m).max() / HIST_BIN_WIDTH) * HIST_BIN_WIDTH
    edges = np.arange(-reach, reach + HIST_BIN_WIDTH / 2, HIST_BIN_WIDTH)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(m, bins=edges, color=COLORS["series1"], edgecolor=COLORS["surface"],
            linewidth=1.0, zorder=3)
    ax.axvline(0, color=COLORS["reference"], lw=1.2, ls=(0, (4, 3)), zorder=4)
    show = max(np.quantile(np.abs(m), 0.995), 0.5)
    show = np.ceil(show / 0.5) * 0.5
    outside = int((np.abs(m) > show).sum())
    ax.set_xlim(-show, show)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Move of the home price from 24h before to tipoff (change in log odds)")
    ax.set_ylabel("Games")
    ax.set_title(f"Price moves from 24h before tipoff to tipoff, {len(m):,} games",
                 loc="left")
    note = f"SD {m.std(ddof=1):.2f}, median {np.median(m):+.3f}"
    if outside:
        note += f"\n{outside} games beyond the axis range"
    ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=COLORS["text_secondary"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def bins_figure(bins: pd.DataFrame, n_games: int, path: Path) -> plt.Figure:
    """Home-win rate minus mean tipoff price against the mean move, by bin of the move."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0, color=COLORS["reference"], lw=1.2, ls=(0, (4, 3)), zorder=1,
               label="Tipoff price is right on average")
    y = bins["win_rate_minus_price"]
    yerr = np.vstack([y - bins["wilson_lo_minus_price"], bins["wilson_hi_minus_price"] - y])
    ax.errorbar(bins["mean_move"], y, yerr=yerr, fmt="o", ms=8, color=COLORS["series1"],
                ecolor=COLORS["series1"], elinewidth=2, capsize=0, mec=COLORS["surface"],
                mew=2, zorder=3, label="Home-win rate minus mean tipoff price, 95% Wilson")
    ax.set_xlabel("Mean move of the home price from 24h before to tipoff (log odds)")
    ax.set_ylabel("Home-win rate minus tipoff price")
    ax.set_title(f"Win rate against tipoff price by size of the move: {n_games:,} games, "
                 f"{len(bins)} equal-count bins", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
              fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig

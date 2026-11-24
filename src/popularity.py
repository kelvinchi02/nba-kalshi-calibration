"""Stage 9 popularity bias test (ANALYSIS_PLAN.md section 9.1).

Library for notebooks/03_popularity.ipynb.

Two popularity measures per team and game:
  volume  leave-one-out mean of the team's own market pregame volume
  metro   metro area population (data/processed/team_metro_population.csv)

The popularity gap of a game is log(home measure) - log(away measure). The model is

    logit P(home_win = 1) = a + b * logit(p_home) + c * gap

with standard errors clustered by home team and a block bootstrap by game date.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import stats_tools
from csv_io import read_csv
from eda import team_game_volume, team_pregame_volume
from plot_style import COLORS, apply_style
from stats_tools import B, SEED, Z, fit_logit, logit

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
RESULTS_DIR = REPO_ROOT / "results"
MEASURES = ("volume", "metro")
# Scatter labels that would sit on a neighbor at the default offset (points: right, up).
LABEL_OFFSETS = {"MIA": (-24, 5), "WAS": (6, -10), "ATL": (-25, -3), "TOR": (6, 5),
                 "DAL": (-26, -4), "CHI": (6, 5), "SAC": (-25, -3), "ORL": (6, 4)}


# ---------------------------------------------------------------- measures and gaps

def leave_one_out_mean(team: pd.Series, value: pd.Series) -> pd.Series:
    """For each row, the mean of `value` over the other rows of the same team."""
    total = value.groupby(team).transform("sum")
    count = value.groupby(team).transform("size")
    return (total - value) / (count - 1)


def log_gap(home: pd.Series, away: pd.Series) -> pd.Series:
    """log(home measure) - log(away measure). Positive: the home team is more popular."""
    return np.log(home.astype(float)) - np.log(away.astype(float))


def volume_gap_leave_one_out(at0: pd.DataFrame) -> pd.DataFrame:
    """Per game: each team's leave-one-out mean volume and the gap between them."""
    long = team_game_volume(at0)
    long["loo"] = leave_one_out_mean(long["team"], long["pregame_contracts"])
    wide = long.pivot(index="game_id", columns="side", values="loo")
    out = pd.DataFrame({"home_volume_loo": wide["home"], "away_volume_loo": wide["away"]})
    out["gap_volume"] = log_gap(out["home_volume_loo"], out["away_volume_loo"])
    return out.reset_index()


def metro_gap(at0: pd.DataFrame, metro: pd.DataFrame) -> pd.DataFrame:
    """Per game: each team's metro population and the gap between them."""
    population = metro.set_index("team")["population"].astype(float)
    out = pd.DataFrame({"game_id": at0["game_id"].to_numpy(),
                        "home_metro": at0["home_team"].map(population).to_numpy(),
                        "away_metro": at0["away_team"].map(population).to_numpy()})
    if out[["home_metro", "away_metro"]].isna().any().any():
        raise ValueError("A team has no metro population")
    out["gap_metro"] = log_gap(out["home_metro"], out["away_metro"])
    return out


def load_sample(metro: pd.DataFrame) -> pd.DataFrame:
    """Main sample at horizon 0, one row per game, with both popularity gaps."""
    sn = read_csv(SNAP_PATH)
    at0 = sn[(sn["horizon_h"] == 0) & sn["main_sample"]].dropna(subset=["p_home"])
    d = at0[["game_id", "game_date", "home_team", "away_team", "p_home", "home_win"]].copy()
    d["home_win"] = d["home_win"].astype(int)
    d = d.merge(volume_gap_leave_one_out(at0), on="game_id", validate="one_to_one")
    d = d.merge(metro_gap(at0, metro), on="game_id", validate="one_to_one")
    return d.sort_values("game_id", ignore_index=True)


# ---------------------------------------------------------------- team level description

def team_measures(metro: pd.DataFrame) -> pd.DataFrame:
    """Per team: season-average volume (plan section 4) and metro population, with ranks."""
    sn = read_csv(SNAP_PATH)
    volume = team_pregame_volume(sn[sn["main_sample"]])
    out = volume[["team", "games", "mean_pregame_contracts"]].merge(
        metro[["team", "metro_area", "population"]], on="team", validate="one_to_one")
    out = out.rename(columns={"population": "metro_population"})
    out["volume_rank"] = out["mean_pregame_contracts"].rank(ascending=False).astype(int)
    out["metro_rank"] = out["metro_population"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("volume_rank", ignore_index=True)


def measure_correlation(teams: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of the logs and Spearman rank correlation, across teams."""
    volume = teams["mean_pregame_contracts"].astype(float)
    population = teams["metro_population"].astype(float)
    return pd.DataFrame([{
        "n_teams": len(teams),
        "pearson_log_log": stats.pearsonr(np.log(volume), np.log(population)).statistic,
        "spearman_rank": stats.spearmanr(volume, population).statistic}])


# ---------------------------------------------------------------- model

def fit_gap_model(d: pd.DataFrame, gap_col: str, clustered: bool = True):
    """Logit of home_win on [1, logit(p_home), gap]."""
    groups = d["home_team"] if clustered else None
    return fit_logit(d["p_home"], d["home_win"], groups=groups, extra=d[gap_col])


def model_table(fit, measure: str, n_clusters: int) -> pd.DataFrame:
    """a, b and c with clustered SEs, 95% intervals and Wald tests of a = 0, b = 1, c = 0."""
    rows = []
    terms = [("intercept_a", 0.0), ("slope_b", 1.0), ("gap_c", 0.0)]
    for i, (term, null) in enumerate(terms):
        est, se = fit.params[i], fit.bse[i]
        z = (est - null) / se
        rows.append({"measure": measure, "term": term, "estimate": est, "se_cluster": se,
                     "ci_lo": est - Z * se, "ci_hi": est + Z * se, "null_value": null,
                     "z": z, "p_value": 2 * stats.norm.sf(abs(z)), "n": int(fit.nobs),
                     "n_clusters": n_clusters})
    return pd.DataFrame(rows)


def average_effect(params, p, gap, step: float) -> float:
    """Mean change in fitted win probability when every game's gap rises by `step`.

    The price is held fixed: mean of L(eta + c * step) - L(eta), with
    eta = a + b * logit(p) + c * gap and L the logistic function.
    """
    a, b, c = params
    eta = a + b * logit(p) + c * np.asarray(gap, float)
    return float(np.mean(_logistic(eta + c * step) - _logistic(eta)))


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def effect_estimates(fit, d: pd.DataFrame, gap_col: str) -> dict:
    """The gap's SD, c per SD, and the average effect of one SD on win probability."""
    sd = float(d[gap_col].std(ddof=1))
    c = float(fit.params[2])
    return {"gap_sd": sd, "gap_c": c, "c_per_sd": c * sd,
            "avg_effect_per_sd": average_effect(fit.params, d["p_home"], d[gap_col], sd)}


def bootstrap_gap_model(d: pd.DataFrame, gap_col: str, n_boot: int = B,
                        seed: int = SEED) -> pd.DataFrame:
    """Block bootstrap by game date: refit the model in every replicate.

    The step used for the average effect stays at the full-sample SD of the gap, so every
    replicate answers the same question.
    """
    p, y, gap = (d[c].to_numpy(dtype=float) for c in ("p_home", "home_win", gap_col))
    sd = float(d[gap_col].std(ddof=1))

    def replicate(idx: np.ndarray) -> dict:
        fit = fit_logit(p[idx], y[idx], extra=gap[idx])
        a, b, c = fit.params
        return {"intercept_a": a, "slope_b": b, "gap_c": c, "c_per_sd": c * sd,
                "avg_effect_per_sd": average_effect(fit.params, p[idx], gap[idx], sd)}

    return stats_tools.block_bootstrap(d["game_date"], replicate, n_boot=n_boot, seed=seed)


def bootstrap_table(boot: pd.DataFrame, fit, effects: dict, measure: str,
                    n_dates: int) -> pd.DataFrame:
    """Percentile intervals for a, b, c, c per SD and the average effect."""
    point = {"intercept_a": fit.params[0], "slope_b": fit.params[1], "gap_c": fit.params[2],
             "c_per_sd": effects["c_per_sd"],
             "avg_effect_per_sd": effects["avg_effect_per_sd"]}
    out = stats_tools.percentile_summary(boot, point, n_dates)
    out.insert(0, "measure", measure)
    return out


def effects_table(effects: dict, boot_summary: pd.DataFrame, measure: str) -> pd.DataFrame:
    """One row per measure: the plain-terms numbers with their bootstrap intervals."""
    ci = boot_summary.set_index("statistic")
    return pd.DataFrame([{
        "measure": measure, "gap_sd": effects["gap_sd"], "gap_c": effects["gap_c"],
        "c_per_sd": effects["c_per_sd"],
        "c_per_sd_boot_lo": ci.loc["c_per_sd", "boot_lo"],
        "c_per_sd_boot_hi": ci.loc["c_per_sd", "boot_hi"],
        "avg_effect_per_sd": effects["avg_effect_per_sd"],
        "avg_effect_boot_lo": ci.loc["avg_effect_per_sd", "boot_lo"],
        "avg_effect_boot_hi": ci.loc["avg_effect_per_sd", "boot_hi"]}])


# ---------------------------------------------------------------- figure

def scatter_figure(teams: pd.DataFrame, corr: pd.DataFrame, path: Path) -> plt.Figure:
    """Season-average volume against metro population, log axes, one labeled dot per team."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    x = teams["metro_population"].astype(float) / 1e6
    y = teams["mean_pregame_contracts"].astype(float) / 1e3
    ax.scatter(x, y, s=42, color=COLORS["series1"], edgecolor=COLORS["surface"],
               linewidth=1.5, zorder=3)
    for team, xi, yi in zip(teams["team"], x, y):
        ax.annotate(team, (xi, yi), xytext=LABEL_OFFSETS.get(team, (5, 4)),
                    textcoords="offset points", fontsize=8, color=COLORS["text_secondary"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_yticks([200, 300, 500, 1000], ["200", "300", "500", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("Metro area population, July 1, 2025 (millions, log scale)")
    ax.set_ylabel("Average pregame contracts per game (thousands, log scale)")
    ax.set_title("Team trading volume against metro population, 30 teams", loc="left")
    r = corr.iloc[0]
    ax.text(0.98, 0.04, f"Pearson r of logs = {r['pearson_log_log']:.2f}\n"
                        f"Spearman rank r = {r['spearman_rank']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=COLORS["text_secondary"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ---------------------------------------------------------------- one measure, start to end

def analyze_measure(d: pd.DataFrame, measure: str, n_boot: int = B,
                    seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Fit, bootstrap and tabulate the model for one measure ("volume" or "metro").

    Returns the tables "logistic" (a, b, c with clustered tests and bootstrap intervals),
    "effects" (the plain-terms numbers) and "bootstrap" (the full bootstrap summary).
    """
    gap_col = f"gap_{measure}"
    fit = fit_gap_model(d, gap_col)
    effects = effect_estimates(fit, d, gap_col)
    boot = bootstrap_gap_model(d, gap_col, n_boot=n_boot, seed=seed)
    summary = bootstrap_table(boot, fit, effects, measure, d["game_date"].nunique())
    ci = summary.set_index("statistic")
    logistic = model_table(fit, measure, d["home_team"].nunique())
    logistic["boot_lo"] = [ci.loc[t, "boot_lo"] for t in logistic["term"]]
    logistic["boot_hi"] = [ci.loc[t, "boot_hi"] for t in logistic["term"]]
    return {"logistic": logistic, "effects": effects_table(effects, summary, measure),
            "bootstrap": summary}

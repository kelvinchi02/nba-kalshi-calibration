"""Stage 10 subgroups: where is the market less accurate (ANALYSIS_PLAN.md section 10.5).

Library for notebooks/08_subgroups.ipynb.

Sample: the main sample at tipoff. Three splits, fixed in the plan:

  liquidity    three equal-count groups by total pregame contracts (low, middle, high)
  season_half  early = before the median game date, late = on or after it
  rest         back_to_back = either team played the previous calendar day, else rested

For each split, one logistic model on all games,

    logit P(home_win = 1) = a + b * logit(p) + sum_g (alpha_g * G_g + beta_g * G_g * logit(p)),

with the first group as the reference and standard errors clustered by home team. The
test is whether the slope differs by group (beta = 0). The model is saturated in the
group, so group g's own intercept and slope are a + alpha_g and b + beta_g.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import all_tests
import stats_tools
from calibration import assign_bins, calibration_bins, logistic_table
from csv_io import read_csv
from plot_style import COLORS, apply_style
from stats_tools import B, SEED, Z, brier, fit_logit, logit, wald_joint

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
SCHEDULE_PATH = REPO_ROOT / "data" / "processed" / "nba_games_2025_26.csv"
RESULTS_DIR = REPO_ROOT / "results"
PART = "subgroups"
N_BINS = 5

# split -> group labels, the reference group first
SPLITS = {
    "liquidity": ("low", "middle", "high"),
    "season_half": ("early", "late"),
    "rest": ("rested", "back_to_back"),
}
# The Brier differences reported per split, as (group, minus group).
BRIER_PAIRS = {
    "liquidity": (("high", "low"), ("middle", "low"), ("high", "middle")),
    "season_half": (("late", "early"),),
    "rest": (("back_to_back", "rested"),),
}
SPLIT_TEXT = {"liquidity": "liquidity", "season_half": "season half", "rest": "rest"}
# Liquidity is ordered, so one hue from light to dark. The other splits use two hues.
GROUP_COLORS = {
    "low": "#86b6ef", "middle": "#3987e5", "high": "#0d366b",
    "early": COLORS["series1"], "late": COLORS["series2"],
    "rested": COLORS["series1"], "back_to_back": COLORS["series2"],
}
GROUP_LABELS = {
    "low": "Low volume", "middle": "Middle volume", "high": "High volume",
    "early": "Early (before the median date)", "late": "Late (median date and after)",
    "rested": "Neither team on a back to back", "back_to_back": "A team on a back to back",
}


# ---------------------------------------------------------------- groups

def liquidity_group(total_contracts: pd.Series, game_id: pd.Series) -> pd.Series:
    """Three equal-count groups by total pregame contracts, ties broken by game_id."""
    order = np.lexsort((game_id.to_numpy(), total_contracts.to_numpy()))  # total, then id
    labels = np.empty(len(order), dtype=object)
    labels[order] = np.array(SPLITS["liquidity"], dtype=object)[assign_bins(len(order), 3)]
    return pd.Series(labels, index=total_contracts.index)


def median_date(game_date: pd.Series) -> pd.Timestamp:
    """The median game date across games."""
    return pd.to_datetime(game_date).median()


def season_half_group(game_date: pd.Series) -> pd.Series:
    """early: before the median game date. late: on or after it."""
    dates = pd.to_datetime(game_date)
    return pd.Series(np.where(dates < median_date(game_date), "early", "late"),
                     index=game_date.index)


def played_previous_day(team: pd.Series, game_date: pd.Series,
                        schedule: pd.DataFrame) -> pd.Series:
    """True when the team has a game in `schedule` on the day before game_date."""
    played = set(zip(schedule["home_team"], pd.to_datetime(schedule["game_date"])))
    played |= set(zip(schedule["away_team"], pd.to_datetime(schedule["game_date"])))
    previous = pd.to_datetime(game_date) - pd.Timedelta(days=1)
    return pd.Series([(t, d) in played for t, d in zip(team, previous)], index=team.index)


def rest_group(d: pd.DataFrame, schedule: pd.DataFrame) -> pd.Series:
    """back_to_back: at least one of the two teams played the previous calendar day."""
    b2b = (played_previous_day(d["home_team"], d["game_date"], schedule)
           | played_previous_day(d["away_team"], d["game_date"], schedule))
    return pd.Series(np.where(b2b, "back_to_back", "rested"), index=d.index)


def add_groups(d: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["total_pregame_contracts"] = (out["home_pregame_contracts"]
                                      + out["away_pregame_contracts"])
    out["liquidity"] = liquidity_group(out["total_pregame_contracts"], out["game_id"])
    out["season_half"] = season_half_group(out["game_date"])
    out["rest"] = rest_group(out, schedule)
    return out


def load_sample() -> pd.DataFrame:
    """Main sample at horizon 0, one row per game, with the three group columns."""
    sn = read_csv(SNAP_PATH)
    at0 = sn[(sn["horizon_h"] == 0) & sn["main_sample"]].dropna(subset=["p_home"])
    d = at0[["game_id", "game_date", "home_team", "away_team", "p_home", "home_win",
             "home_pregame_contracts", "away_pregame_contracts"]].copy()
    d["home_win"] = d["home_win"].astype(int)
    d = d.sort_values("game_id", ignore_index=True)
    return add_groups(d, read_csv(SCHEDULE_PATH))


def group_definitions(d: pd.DataFrame) -> pd.DataFrame:
    """Per split and group: games, dates and the range of the variable that defines it."""
    rows = []
    for split, groups in SPLITS.items():
        for g in groups:
            x = d[d[split] == g]
            row = {"split": split, "group": g, "games": len(x),
                   "game_dates": x["game_date"].nunique(),
                   "first_game_date": str(x["game_date"].min()),
                   "last_game_date": str(x["game_date"].max()),
                   "contracts_min": np.nan, "contracts_max": np.nan}
            if split == "liquidity":
                row["contracts_min"] = x["total_pregame_contracts"].min()
                row["contracts_max"] = x["total_pregame_contracts"].max()
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per group

def group_metrics(d: pd.DataFrame, split: str) -> pd.DataFrame:
    """Per group: Brier score with its own baseline, and a and b with clustered tests."""
    rows = []
    for g in SPLITS[split]:
        x = d[d[split] == g]
        logi = logistic_table(fit_logit(x["p_home"], x["home_win"], groups=x["home_team"]),
                              x["home_team"].nunique()).set_index("term")
        a, b = logi.loc["intercept_a"], logi.loc["slope_b"]
        rows.append({"split": split, "group": g, **brier(x["p_home"], x["home_win"]),
                     "intercept_a": a["estimate"], "a_se_cluster": a["se_cluster"],
                     "a_ci_lo": a["ci_lo"], "a_ci_hi": a["ci_hi"], "a_p_value": a["p_value"],
                     "slope_b": b["estimate"], "b_se_cluster": b["se_cluster"],
                     "b_ci_lo": b["ci_lo"], "b_ci_hi": b["ci_hi"], "b_p_value": b["p_value"]})
    return pd.DataFrame(rows)


def group_calibration_bins(d: pd.DataFrame, split: str, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-count bins within each group, built as in plan section 2.1."""
    frames = []
    for g in SPLITS[split]:
        x = d[d[split] == g].sort_values(["p_home", "game_id"]).reset_index(drop=True)
        x["bin"] = assign_bins(len(x), n_bins)
        bins = calibration_bins(x)
        bins.insert(0, "group", g)
        bins.insert(0, "split", split)
        frames.append(bins)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- interaction model

def term_names(groups: tuple[str, ...]) -> list[str]:
    return (["intercept_a", "slope_b"] + [f"intercept_shift_{g}" for g in groups[1:]]
            + [f"slope_shift_{g}" for g in groups[1:]])


def design(p, group, groups: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Columns [1, logit(p), G_g ..., G_g * logit(p) ...] for every group but the first."""
    x = logit(p)
    group = np.asarray(group)
    cols = [np.ones(len(x)), x]
    cols += [(group == g).astype(float) for g in groups[1:]]
    cols += [(group == g).astype(float) * x for g in groups[1:]]
    return np.column_stack(cols), term_names(groups)


def fit_interaction(d: pd.DataFrame, split: str, clustered: bool = True):
    """The interaction model for one split. Returns (fit, term names)."""
    X, names = design(d["p_home"], d[split], SPLITS[split])
    model = sm.Logit(d["home_win"].to_numpy(dtype=float), X)
    if not clustered:
        return model.fit(disp=0), names
    codes = pd.factorize(d["home_team"])[0]
    return model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": codes}), names


def interaction_table(fit, names: list[str], split: str, n_clusters: int) -> pd.DataFrame:
    """Every coefficient with its clustered SE, 95% interval and Wald test against 0
    (b against 1). The last row is the joint Wald test that all slope shifts are 0."""
    rows = []
    for i, term in enumerate(names):
        null = 1.0 if term == "slope_b" else 0.0
        est, se = float(fit.params[i]), float(fit.bse[i])
        z = (est - null) / se
        rows.append({"split": split, "term": term, "estimate": est, "se_cluster": se,
                     "ci_lo": est - Z * se, "ci_hi": est + Z * se, "null_value": null,
                     "statistic": z, "statistic_name": "z", "df": pd.NA,
                     "p_value": 2 * stats.norm.sf(abs(z)), "n": int(fit.nobs),
                     "n_clusters": n_clusters})
    slope_terms = [i for i, t in enumerate(names) if t.startswith("slope_shift_")]
    joint = wald_joint(fit.params, fit.cov_params(), slope_terms)
    rows.append({"split": split, "term": "all_slope_shifts", "estimate": np.nan,
                 "se_cluster": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "null_value": 0.0,
                 "statistic": joint["chi2"], "statistic_name": "chi2", "df": joint["df"],
                 "p_value": joint["p_value"], "n": int(fit.nobs), "n_clusters": n_clusters})
    out = pd.DataFrame(rows)
    out["df"] = out["df"].astype("Int64")
    return out


# ---------------------------------------------------------------- bootstrap

def point_estimates(d: pd.DataFrame, split: str) -> dict:
    """Everything the bootstrap recomputes, on the full sample."""
    groups = SPLITS[split]
    fit, names = fit_interaction(d, split, clustered=False)
    return _statistics(d["p_home"].to_numpy(float), d["home_win"].to_numpy(float),
                       d[split].to_numpy(), groups, dict(zip(names, fit.params)), split)


def _statistics(p, y, group, groups, coef: dict | None, split: str) -> dict:
    rec = {}
    for g in groups:
        mask = group == g
        rec[f"brier_{g}"] = float(np.mean((p[mask] - y[mask]) ** 2)) if mask.any() else np.nan
    for hi, lo in BRIER_PAIRS[split]:
        rec[f"brier_diff_{hi}_minus_{lo}"] = rec[f"brier_{hi}"] - rec[f"brier_{lo}"]
    for name in term_names(groups):
        rec[name] = coef[name] if coef is not None else np.nan
    for g in groups:
        shift_a = 0.0 if g == groups[0] else rec[f"intercept_shift_{g}"]
        shift_b = 0.0 if g == groups[0] else rec[f"slope_shift_{g}"]
        rec[f"a_{g}"] = rec["intercept_a"] + shift_a
        rec[f"b_{g}"] = rec["slope_b"] + shift_b
    return rec


def bootstrap_split(d: pd.DataFrame, split: str, n_boot: int = B,
                    seed: int = SEED) -> pd.DataFrame:
    """Block bootstrap by game date. Group labels stay as assigned in the full sample.

    Each replicate recomputes the Brier score of every group, the differences between
    groups, and the interaction model. A replicate whose model cannot be fitted keeps its
    Brier scores and gets missing coefficients.
    """
    groups = SPLITS[split]
    p, y = d["p_home"].to_numpy(float), d["home_win"].to_numpy(float)
    group = d[split].to_numpy()

    def replicate(idx: np.ndarray) -> dict:
        pb, yb, gb = p[idx], y[idx], group[idx]
        coef = None
        try:
            X, names = design(pb, gb, groups)
            with warnings.catch_warnings():
                warnings.simplefilter("error")      # separation or no convergence: no fit
                fit = sm.Logit(yb, X).fit(disp=0)
            if np.all(np.isfinite(fit.params)):
                coef = dict(zip(names, fit.params))
        except Exception:  # noqa: BLE001 - a replicate without a fit is counted, not fatal
            coef = None
        return _statistics(pb, yb, gb, groups, coef, split)

    return stats_tools.block_bootstrap(d["game_date"], replicate, n_boot=n_boot, seed=seed)


def bootstrap_table(d: pd.DataFrame, boot: pd.DataFrame, split: str) -> pd.DataFrame:
    """Percentile intervals for every bootstrapped statistic. B counts the replicates used."""
    out = stats_tools.percentile_summary(boot, point_estimates(d, split),
                                         d["game_date"].nunique())
    out.insert(0, "split", split)
    return out


def brier_diff_table(summary: pd.DataFrame, boot: pd.DataFrame, split: str) -> pd.DataFrame:
    """Brier differences between groups with bootstrap intervals and bootstrap p values."""
    s = summary.set_index("statistic")
    rows = []
    for hi, lo in BRIER_PAIRS[split]:
        key = f"brier_diff_{hi}_minus_{lo}"
        rows.append({"split": split, "group": hi, "minus_group": lo,
                     "brier_group": s.loc[f"brier_{hi}", "estimate"],
                     "brier_minus_group": s.loc[f"brier_{lo}", "estimate"],
                     "brier_diff": s.loc[key, "estimate"], "boot_lo": s.loc[key, "boot_lo"],
                     "boot_hi": s.loc[key, "boot_hi"],
                     "p_bootstrap": stats_tools.bootstrap_p_value(boot[key])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- one split, start to end

def analyze_split(d: pd.DataFrame, split: str, n_boot: int = B,
                  seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Tables for one split: metrics, interaction, bootstrap, brier_diff, bins, tests."""
    fit, names = fit_interaction(d, split)
    interaction = interaction_table(fit, names, split, d["home_team"].nunique())
    boot = bootstrap_split(d, split, n_boot=n_boot, seed=seed)
    summary = bootstrap_table(d, boot, split)
    ci = summary.set_index("statistic")
    interaction["boot_lo"] = [ci["boot_lo"].get(t, np.nan) for t in interaction["term"]]
    interaction["boot_hi"] = [ci["boot_hi"].get(t, np.nan) for t in interaction["term"]]
    metrics = group_metrics(d, split)
    for coef in ("a", "b"):
        metrics[f"{coef}_boot_lo"] = [ci.loc[f"{coef}_{g}", "boot_lo"] for g in metrics["group"]]
        metrics[f"{coef}_boot_hi"] = [ci.loc[f"{coef}_{g}", "boot_hi"] for g in metrics["group"]]
    diffs = brier_diff_table(summary, boot, split)
    out = {"metrics": metrics, "interaction": interaction, "bootstrap": summary,
           "brier_diff": diffs, "bins": group_calibration_bins(d, split)}
    out["tests"] = tests_table(split, metrics, interaction, diffs, boot)
    return out


def tests_table(split: str, metrics: pd.DataFrame, interaction: pd.DataFrame,
                diffs: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    """The tests of one split in the common Stage 10 layout."""
    text = SPLIT_TEXT[split]
    groups = SPLITS[split]
    rows = []
    inter = interaction.set_index("term")
    n = int(inter["n"].iloc[0])
    # The primary test: the slope differs by group.
    if len(groups) == 2:
        r = inter.loc[f"slope_shift_{groups[1]}"]
        rows.append(all_tests.wald_row(
            PART, f"subgroup_{split}_slope",
            f"{text} split: slope of {groups[1]} minus slope of {groups[0]}",
            "the slope is the same in both groups", n, r["estimate"], r["se_cluster"],
            r["p_value"], boot=(r["boot_lo"], r["boot_hi"])))
    else:
        r = inter.loc["all_slope_shifts"]
        rows.append(all_tests.joint_row(
            PART, f"subgroup_{split}_slope",
            f"{text} split: the slope differs between the three groups",
            "the slope is the same in all groups", n, r["statistic"], r["df"], r["p_value"],
            "chi2", "clustered Wald chi2"))
        for g in groups[1:]:
            r = inter.loc[f"slope_shift_{g}"]
            rows.append(all_tests.wald_row(
                PART, f"subgroup_{split}_slope_{g}",
                f"{text} split: slope of {g} minus slope of {groups[0]}",
                f"same slope as {groups[0]}", n, r["estimate"], r["se_cluster"],
                r["p_value"], boot=(r["boot_lo"], r["boot_hi"])))
    for g in groups[1:]:
        r = inter.loc[f"intercept_shift_{g}"]
        rows.append(all_tests.wald_row(
            PART, f"subgroup_{split}_intercept_{g}",
            f"{text} split: intercept of {g} minus intercept of {groups[0]}",
            f"same intercept as {groups[0]}", n, r["estimate"], r["se_cluster"],
            r["p_value"], boot=(r["boot_lo"], r["boot_hi"])))
    # Recalibration within each group.
    for m in metrics.itertuples(index=False):
        rows.append(all_tests.wald_row(
            PART, f"subgroup_{split}_{m.group}_a", f"{text} split, {m.group} group: a",
            "a = 0", m.n, m.intercept_a, m.a_se_cluster, m.a_p_value,
            boot=(m.a_boot_lo, m.a_boot_hi)))
        rows.append(all_tests.wald_row(
            PART, f"subgroup_{split}_{m.group}_b", f"{text} split, {m.group} group: b",
            "b = 1", m.n, m.slope_b, m.b_se_cluster, m.b_p_value, null_value=1.0,
            boot=(m.b_boot_lo, m.b_boot_hi)))
    # Brier differences between groups.
    for r in diffs.itertuples(index=False):
        key = f"brier_diff_{r.group}_minus_{r.minus_group}"
        rows.append(all_tests.bootstrap_row(
            PART, f"subgroup_{split}_brier_{r.group}_minus_{r.minus_group}",
            f"{text} split: Brier score of {r.group} minus {r.minus_group}",
            "the Brier scores are equal", n, r.brier_diff, boot[key], unit="games"))
    return all_tests.table(rows)


# ---------------------------------------------------------------- figure

def calibration_figure(bins: pd.DataFrame, metrics: pd.DataFrame, split: str,
                       path: Path) -> plt.Figure:
    """Calibration by group: bin points with Wilson intervals, joined by lines."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], lw=1.2, ls=(0, (4, 3)), zorder=1,
            label="Perfect calibration")
    counts = metrics.set_index("group")["n"]
    for g in SPLITS[split]:
        b = bins[bins["group"] == g]
        yerr = np.vstack([b["win_rate"] - b["wilson_lo"], b["wilson_hi"] - b["win_rate"]])
        ax.errorbar(b["mean_p"], b["win_rate"], yerr=yerr, fmt="o-", ms=7, lw=2,
                    color=GROUP_COLORS[g], ecolor=GROUP_COLORS[g], elinewidth=1.5, capsize=0,
                    mec=COLORS["surface"], mew=1.5, zorder=3,
                    label=f"{GROUP_LABELS[g]} ({counts[g]:,} games)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Kalshi home-win price at tipoff (bin mean of p_home)")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title(f"Calibration at tipoff by {SPLIT_TEXT[split]}", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(0.98, 0.04, f"{N_BINS} equal-count bins per group\n95% Wilson intervals",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=COLORS["text_secondary"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ================================================================ exploratory (plan X2)

MIN_MONTH_GAMES = 100          # a smaller first or last month joins the month next to it
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December")


def month_group(game_date: pd.Series, min_games: int = MIN_MONTH_GAMES) -> pd.Series:
    """Calendar month of each game, as "2025-10". A first or last month with fewer than
    min_games games is combined with its neighbor and labeled "2025-10 to 2025-11"."""
    month = pd.to_datetime(game_date).dt.strftime("%Y-%m")
    order = sorted(month.unique())
    counts = month.value_counts()
    label = {m: m for m in order}
    if len(order) > 1 and counts[order[0]] < min_games:
        label[order[0]] = label[order[1]] = f"{order[0]} to {order[1]}"
    if len(order) > 2 and counts[order[-1]] < min_games:
        joined = f"{order[-2]} to {order[-1]}"
        label[order[-1]] = label[order[-2]] = joined
    return month.map(label)


def month_title(label: str) -> str:
    """"2025-10 to 2025-11" -> "Oct to Nov", "2026-01" -> "Jan"."""
    return " to ".join(MONTH_NAMES[int(part[5:7]) - 1][:3] for part in label.split(" to "))


def monthly_slopes(d: pd.DataFrame, min_games: int = MIN_MONTH_GAMES) -> pd.DataFrame:
    """a and b of the recalibration fit within each calendar month (plan X2), with
    standard errors clustered by home team, 95% intervals and the Wald test of b = 1."""
    groups = month_group(d["game_date"], min_games)
    rows = []
    for label in sorted(groups.unique()):
        x = d[groups == label]
        logi = logistic_table(fit_logit(x["p_home"], x["home_win"], groups=x["home_team"]),
                              x["home_team"].nunique()).set_index("term")
        a, b = logi.loc["intercept_a"], logi.loc["slope_b"]
        rows.append({"month": label, "month_label": month_title(label), "n": len(x),
                     "first_game_date": str(x["game_date"].min()),
                     "last_game_date": str(x["game_date"].max()),
                     "home_win_rate": float(x["home_win"].mean()),
                     "n_clusters": int(b["n_clusters"]),
                     "intercept_a": a["estimate"], "a_se_cluster": a["se_cluster"],
                     "a_ci_lo": a["ci_lo"], "a_ci_hi": a["ci_hi"], "a_p_value": a["p_value"],
                     "slope_b": b["estimate"], "b_se_cluster": b["se_cluster"],
                     "b_ci_lo": b["ci_lo"], "b_ci_hi": b["ci_hi"], "b_p_value": b["p_value"]})
    return pd.DataFrame(rows)


def monthly_tests(table: pd.DataFrame) -> pd.DataFrame:
    """The Wald tests of b = 1 by month in the common layout, marked exploratory."""
    rows = [all_tests.wald_row(
        PART, f"exploratory_slope_{r.month.replace(' to ', '_').replace('-', '_')}",
        f"calibration slope b in {r.month_label} ({r.month})", "b = 1", r.n, r.slope_b,
        r.b_se_cluster, r.b_p_value, null_value=1.0) for r in table.itertuples(index=False)]
    return all_tests.mark_exploratory(all_tests.table(rows))


def monthly_figure(table: pd.DataFrame, path: Path) -> plt.Figure:
    """The slope b in each month with its clustered 95% interval."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(table))
    b = table["slope_b"].to_numpy()
    yerr = np.vstack([b - table["b_ci_lo"], table["b_ci_hi"] - b])
    ax.axhline(1, color=COLORS["reference"], lw=1.2, ls=(0, (4, 3)), zorder=1,
               label="b = 1: calibrated")
    ax.errorbar(x, b, yerr=yerr, fmt="o", ms=8, color=COLORS["series1"],
                ecolor=COLORS["series1"], elinewidth=2, capsize=0, mec=COLORS["surface"],
                mew=2, zorder=3, label="Slope b, clustered 95% interval")
    for xi, v in zip(x, b):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=COLORS["text_secondary"])
    ax.set_xticks(x, [f"{r.month_label}\n{r.n} games" for r in table.itertuples()])
    ax.set_xlim(-0.5, len(table) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Calibration slope b")
    ax.set_title("Exploratory: calibration slope at tipoff by month, 2025-26", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig

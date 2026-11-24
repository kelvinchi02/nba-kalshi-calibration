"""Stage 10 replication on the 2026-27 season (ANALYSIS_PLAN.md section 10.6).

Library for notebooks/09_replication.ipynb.

Sample: the 2026-27 main sample at tipoff, from data/processed/snapshots_2026_27.csv.

  1. Calibration at tipoff as in Stage 8: Brier score against the constant baseline, a and
     b with clustered standard errors, the LR test and the block bootstrap by date.
  2. Popularity: each team's 2025-26 season-average pregame volume (plan section 4) is the
     measure. It was known before the 2026-27 season began. The model is the one of 9.1.
  3. Opposite rule: at tipoff, stake $100 on the MORE popular team when |gap| is at or
     above the 75th percentile of |gap| over the 2025-26 main sample. Taker fee as in 9.3.
     The benchmark buys the more popular team in every game.

Nothing here is new statistics: the pieces come from calibration.py, popularity.py and
strategy.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import all_tests
import calibration
import popularity
import strategy
from csv_io import read_csv
from eda import team_pregame_volume
from seasons import get_season
from stats_tools import B, SEED, brier, fit_logit, lr_test

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PART = "replication"
PREVIOUS = get_season("2025-26")
NEW = get_season("2026-27")
N_BINS = 5
MEASURE = "volume_2025_26"
GAP_COL = f"gap_{MEASURE}"
THRESHOLD_QUANTILE = strategy.THRESHOLD_QUANTILE
FEE_CHECKED = "2026-11-23"


# ---------------------------------------------------------------- data

def at_tipoff(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Main-sample rows at horizon 0."""
    return snapshots[(snapshots["horizon_h"] == 0) & snapshots["main_sample"]]


def previous_team_volume(previous_snapshots: pd.DataFrame) -> pd.Series:
    """Each team's 2025-26 season-average pregame volume (plan section 4), by team."""
    table = team_pregame_volume(previous_snapshots[previous_snapshots["main_sample"]])
    return table.set_index("team")["mean_pregame_contracts"].astype(float)


def add_gap(d: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
    """Add each team's measure and gap = log(home measure) - log(away measure)."""
    out = d.copy()
    out["home_volume_2025_26"] = out["home_team"].map(volume)
    out["away_volume_2025_26"] = out["away_team"].map(volume)
    missing = sorted(set(out.loc[out["home_volume_2025_26"].isna(), "home_team"])
                     | set(out.loc[out["away_volume_2025_26"].isna(), "away_team"]))
    if missing:
        raise ValueError(f"No 2025-26 measure for: {', '.join(missing)}")
    out[GAP_COL] = popularity.log_gap(out["home_volume_2025_26"], out["away_volume_2025_26"])
    return out


def previous_threshold(previous_at0: pd.DataFrame, volume: pd.Series,
                       quantile: float = THRESHOLD_QUANTILE) -> float:
    """The quantile of |gap| over the 2025-26 main-sample games, linear interpolation."""
    gap = add_gap(previous_at0[["home_team", "away_team"]], volume)[GAP_COL]
    return float(np.quantile(gap.abs().to_numpy(dtype=float), quantile))


def build_sample(new_snapshots: pd.DataFrame, volume: pd.Series) -> tuple[pd.DataFrame, int]:
    """(sample, number of main-sample games dropped for a missing p_home).

    One row per game of the new season's main sample at tipoff, with the gap and the
    columns the calibration, popularity and trading code expect.
    """
    at0 = at_tipoff(new_snapshots)
    n_missing = int(at0["p_home"].isna().sum())
    d = at0.dropna(subset=["p_home"])
    d = d[["game_id", "game_date", "tipoff_utc", "home_team", "away_team", "p_home",
           "home_win", "home_ask", "away_ask"]].copy()
    d["home_win"] = d["home_win"].astype(int)
    d = add_gap(d, volume)
    d = d.sort_values(["p_home", "game_id"]).reset_index(drop=True)
    d["bin"] = calibration.assign_bins(len(d), N_BINS)
    # Every sample game can be traded: the threshold comes from the season before.
    d["eligible"] = True
    d["tradable"] = True
    return d, n_missing


def load() -> dict:
    """Everything notebook 09 needs: the sample, the measure, the threshold and counts."""
    previous = read_csv(PREVIOUS.snapshots_path)
    new = read_csv(NEW.snapshots_path)
    game_map = read_csv(NEW.game_map_path)
    volume = previous_team_volume(previous)
    sample, n_missing = build_sample(new, volume)
    threshold = previous_threshold(at_tipoff(previous), volume)
    return {"sample": sample, "volume": volume, "threshold": threshold,
            "counts": sample_counts(game_map, sample, n_missing)}


def sample_counts(game_map: pd.DataFrame, sample: pd.DataFrame, n_missing: int) -> pd.DataFrame:
    rows = [
        ("final regular-season games up to the cutoff date", len(game_map)),
        ("matched to a settled Kalshi event", int(game_map["matched"].sum())),
        ("main sample (matched, not postponed_window)", int(game_map["main_sample"].sum())),
        ("main sample games with p_home missing at tipoff (dropped)", n_missing),
        ("games in the analysis", len(sample)),
        ("game dates in the analysis", sample["game_date"].nunique()),
    ]
    out = pd.DataFrame(rows, columns=["group", "count"])
    out["first_game_date"] = str(game_map["game_date"].min())
    out["cutoff_date"] = str(game_map["game_date"].max())
    return out


# ---------------------------------------------------------------- 1. calibration

def calibration_tables(d: pd.DataFrame, n_boot: int = B, seed: int = SEED) -> dict:
    """Stage 8 on the new sample: bins, Brier, logistic, LR test and bootstrap summary."""
    bins = calibration.calibration_bins(d)
    bs = brier(d["p_home"], d["home_win"])
    fit = fit_logit(d["p_home"], d["home_win"], groups=d["home_team"])
    logistic = calibration.logistic_table(fit, d["home_team"].nunique())
    lr = lr_test(d["p_home"], d["home_win"])
    boot = calibration.block_bootstrap(d, n_boot=n_boot, seed=seed, n_bins=N_BINS,
                                       tolerate_failed_fit=True)
    summary = calibration.bootstrap_summary(
        boot, calibration.point_estimates(bs, fit, lr, bins), d["game_date"].nunique())
    ci = summary.set_index("statistic")
    logistic["boot_lo"] = [ci.loc[t, "boot_lo"] for t in logistic["term"]]
    logistic["boot_hi"] = [ci.loc[t, "boot_hi"] for t in logistic["term"]]
    return {"bins": calibration.add_bootstrap_intervals(bins, summary),
            "brier": pd.DataFrame([bs]), "logistic": logistic, "lr": pd.DataFrame([lr]),
            "bootstrap": summary, "boot": boot}


def slope_width(logistic: pd.DataFrame, season: str) -> dict:
    """Width of the 95% intervals for the slope b: clustered and bootstrap."""
    b = logistic.set_index("term").loc["slope_b"]
    return {"season": season, "n": int(b["n"]), "slope_b": b["estimate"],
            "se_cluster": b["se_cluster"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
            "ci_width": b["ci_hi"] - b["ci_lo"], "boot_lo": b["boot_lo"],
            "boot_hi": b["boot_hi"], "boot_width": b["boot_hi"] - b["boot_lo"]}


def previous_slope(n_boot: int = B, seed: int = SEED) -> pd.DataFrame:
    """a and b of the 2025-26 main sample with clustered and bootstrap intervals (Stage 8)."""
    d = calibration.load_sample()
    fit = fit_logit(d["p_home"], d["home_win"], groups=d["home_team"])
    logistic = calibration.logistic_table(fit, d["home_team"].nunique())
    boot = calibration.block_bootstrap(d, n_boot=n_boot, seed=seed)
    for col, q in (("boot_lo", 0.025), ("boot_hi", 0.975)):
        logistic[col] = [boot[t].quantile(q) for t in logistic["term"]]
    return logistic


# ---------------------------------------------------------------- 3. opposite rule

def opposite_rule(d: pd.DataFrame, threshold: float, n_boot: int = B, seed: int = SEED) -> dict:
    """The opposite rule and its benchmark: trades, results rows and bootstrap replicates."""
    games = d.sort_values(["tipoff_utc", "game_id"], ignore_index=True)
    out = {"trades": [], "results": [], "boot": {}}
    for name, use_threshold in (("opposite_rule", True), ("opposite_benchmark", False)):
        table = strategy.make_trades(games, GAP_COL, use_threshold, buy="more_popular",
                                     threshold=threshold)
        table.insert(0, "strategy", name)
        boot = strategy.bootstrap_trades(table, n_boot=n_boot, seed=seed)
        out["trades"].append(table)
        out["results"].append(strategy.results_row(name, table, boot))
        out["boot"][name] = boot
    out["trades"] = pd.concat(out["trades"], ignore_index=True)
    results = pd.DataFrame(out["results"])
    before = strategy.before_fees_table(results)
    results["profit_before_fees"] = before["profit_before_fees"]
    results["return_before_fees"] = before["return_before_fees"]
    results.insert(1, "threshold", [threshold, np.nan])
    out["results"] = results
    return out


# ---------------------------------------------------------------- tests

def tests_table(cal: dict, pop: dict, rule: dict) -> pd.DataFrame:
    """Every test of the replication in the common Stage 10 layout."""
    rows = []
    lr = cal["lr"].iloc[0]
    n = int(cal["brier"].iloc[0]["n"])
    rows.append(all_tests.joint_row(
        PART, "replication_lr", "2026-27 calibration at tipoff: LR test of a = 0 and b = 1",
        "a = 0 and b = 1", n, lr["lr_stat"], 2, lr["p_value"], "LR", "LR chi2"))
    for r in cal["logistic"].itertuples(index=False):
        short = "a" if r.term == "intercept_a" else "b"
        rows.append(all_tests.wald_row(
            PART, f"replication_calibration_{short}", f"2026-27 calibration at tipoff: {short}",
            f"{short} = {r.null_value:g}", r.n, r.estimate, r.se_cluster, r.p_value,
            null_value=r.null_value, boot=(r.boot_lo, r.boot_hi)))
    bs = cal["brier"].iloc[0]
    rows.append(all_tests.bootstrap_row(
        PART, "replication_brier_diff",
        "2026-27 Brier score minus the constant baseline", "the two are equal", n,
        bs["brier_diff"], cal["boot"]["brier_diff"], unit="games"))
    names = {"intercept_a": ("a", "a = 0"), "slope_b": ("b", "b = 1"), "gap_c": ("c", "c = 0")}
    for r in pop["logistic"].itertuples(index=False):
        short, null_text = names[r.term]
        rows.append(all_tests.wald_row(
            PART, f"replication_popularity_{short}",
            f"2026-27 popularity model with the 2025-26 volume gap: {short}", null_text, r.n,
            r.estimate, r.se_cluster, r.p_value, null_value=r.null_value,
            boot=(r.boot_lo, r.boot_hi)))
    labels = {"opposite_rule": ("replication_opposite_return",
                                "opposite rule: return per dollar staked"),
              "opposite_benchmark": ("replication_opposite_benchmark_return",
                                     "opposite rule benchmark: return per dollar staked")}
    for r in rule["results"].itertuples(index=False):
        test_id, text = labels[r.strategy]
        rows.append(all_tests.bootstrap_row(
            PART, test_id, text, "the return is 0", r.n_trades, r.return_per_dollar,
            rule["boot"][r.strategy]["return_per_dollar"], unit="trades"))
    return all_tests.table(rows)

"""Stage 8: calibration analysis at tipoff (horizon 0), exactly as in ANALYSIS_PLAN.md.

Main sample: main_sample games at horizon_h == 0, price p_home, outcome home_win.

  2.1 calibration plot: 10 equal-count bins, 95% Wilson intervals
  2.2 Brier score against a constant baseline (the sample home-win rate)
  2.3 logistic regression of home_win on logit(p_home), SEs clustered by home team
  2.4 likelihood ratio test of intercept 0 and slope 1
  3   block bootstrap by game date (B = 2000, seed 4981)

Outputs in results/: calibration_bins.csv, brier.csv, logistic.csv, lr_test.csv,
bootstrap_summary.csv and calibration_plot.png.

Usage:
    python src/calibration.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

from csv_io import read_csv, write_csv  # noqa: E402
from plot_style import COLORS, apply_style  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
RESULTS_DIR = REPO_ROOT / "results"

HORIZON = 0
N_BINS = 10
Z = 1.96            # 95% intervals
B = 2000            # bootstrap replicates
SEED = 4981


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


# ---------------------------------------------------------------- statistics

def logit(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def wilson(k: int, m: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for k successes out of m."""
    if m == 0:
        return np.nan, np.nan
    phat = k / m
    denom = 1 + z**2 / m
    center = (phat + z**2 / (2 * m)) / denom
    half = z * np.sqrt(phat * (1 - phat) / m + z**2 / (4 * m**2)) / denom
    return center - half, center + half


def brier(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    ybar = y.mean()
    bs = float(np.mean((p - y) ** 2))
    bs0 = float(ybar * (1 - ybar))
    return {"n": len(y), "home_win_rate": ybar, "brier": bs, "brier_baseline": bs0,
            "brier_diff": bs - bs0, "brier_skill": 1 - bs / bs0}


def loglik_restricted(p, y) -> float:
    """Log-likelihood of taking the prices as they are (a = 0, b = 1)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logit(p, y, groups=None):
    """Logit of y on [1, logit(p)]. Cluster-robust covariance when groups is given."""
    X = np.column_stack([np.ones(len(p)), logit(p)])
    model = sm.Logit(np.asarray(y, float), X)
    if groups is None:
        return model.fit(disp=0)
    codes = pd.factorize(pd.Series(groups))[0]
    return model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": codes})


def lr_test(p, y, fit=None) -> dict:
    fit = fit if fit is not None else fit_logit(p, y)
    ll_r, ll_u = loglik_restricted(p, y), float(fit.llf)
    lr = 2 * (ll_u - ll_r)
    return {"ll_restricted": ll_r, "ll_unrestricted": ll_u, "lr_stat": lr, "df": 2,
            "p_value": float(stats.chi2.sf(lr, 2))}


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

def block_bootstrap(d: pd.DataFrame, n_boot: int = B, seed: int = SEED) -> pd.DataFrame:
    """Resample whole game dates with replacement. Bins keep their full-sample assignment."""
    rng = np.random.default_rng(seed)
    blocks = [g.index.to_numpy() for _, g in d.groupby("game_date")]
    p, y, bins = d["p_home"].to_numpy(), d["home_win"].to_numpy(), d["bin"].to_numpy()
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        pb, yb, bb = p[idx], y[idx], bins[idx]
        fit = fit_logit(pb, yb)
        rec = {**brier(pb, yb), "intercept_a": fit.params[0], "slope_b": fit.params[1],
               "lr_stat": lr_test(pb, yb, fit)["lr_stat"]}
        for k in range(N_BINS):
            mask = bb == k
            rec[f"bin{k + 1}_win_rate"] = yb[mask].mean() if mask.any() else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def bootstrap_summary(boot: pd.DataFrame, point: dict, n_dates: int) -> pd.DataFrame:
    rows = []
    for stat, est in point.items():
        x = boot[stat].dropna()
        rows.append({"statistic": stat, "estimate": est, "boot_mean": x.mean(),
                     "boot_se": x.std(ddof=1), "boot_lo": x.quantile(0.025),
                     "boot_hi": x.quantile(0.975), "B": len(x), "n_dates": n_dates,
                     "seed": SEED})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figure

def calibration_plot(bins: pd.DataFrame, bs: dict, lr: dict, logi: pd.DataFrame,
                     path: Path) -> None:
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
    ax.set_title(f"Calibration at tipoff: {bs['n']:,} games in 10 equal-count bins",
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
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    d = load_sample()
    n_dates, n_clusters = d["game_date"].nunique(), d["home_team"].nunique()

    bins = calibration_bins(d)
    bs = brier(d["p_home"], d["home_win"])
    fit = fit_logit(d["p_home"], d["home_win"], groups=d["home_team"])
    logi = logistic_table(fit, n_clusters)
    lr = lr_test(d["p_home"], d["home_win"], fit_logit(d["p_home"], d["home_win"]))

    boot = block_bootstrap(d)
    point = {k: bs[k] for k in ("brier", "brier_baseline", "brier_diff", "brier_skill")}
    point |= {"intercept_a": fit.params[0], "slope_b": fit.params[1], "lr_stat": lr["lr_stat"]}
    point |= {f"bin{k + 1}_win_rate": r for k, r in enumerate(bins["win_rate"])}
    summary = bootstrap_summary(boot, point, n_dates)
    bin_ci = summary.set_index("statistic")
    bins["boot_lo"] = [bin_ci.loc[f"bin{k}_win_rate", "boot_lo"] for k in bins["bin"]]
    bins["boot_hi"] = [bin_ci.loc[f"bin{k}_win_rate", "boot_hi"] for k in bins["bin"]]

    write_csv(bins, RESULTS_DIR / "calibration_bins.csv")
    write_csv(pd.DataFrame([bs]), RESULTS_DIR / "brier.csv")
    write_csv(logi, RESULTS_DIR / "logistic.csv")
    write_csv(pd.DataFrame([lr]), RESULTS_DIR / "lr_test.csv")
    write_csv(summary, RESULTS_DIR / "bootstrap_summary.csv")
    calibration_plot(bins, bs, lr, logi, RESULTS_DIR / "calibration_plot.png")

    pd.set_option("display.width", 160)
    print(f"Sample: {len(d)} games, {n_dates} dates, {n_clusters} home-team clusters\n")
    print(bins.round(4).to_string(index=False), "\n")
    print(pd.Series(bs).round(5).to_string(), "\n")
    print(logi.round(4).to_string(index=False), "\n")
    print(pd.Series(lr).round(4).to_string(), "\n")
    print(summary.drop(columns=["B", "n_dates", "seed"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

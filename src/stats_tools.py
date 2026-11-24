"""Reusable statistics for the analysis notebooks.

  logit              log odds of a probability
  wilson             Wilson score interval for a proportion
  brier              Brier score against a constant baseline
  loglik_restricted  log-likelihood of taking the prices as they are
  fit_logit          logistic regression on logit(p), optional extra covariates and
                     cluster-robust standard errors
  lr_test            likelihood ratio test of intercept 0 and slope 1
  date_blocks        row positions of each date
  block_bootstrap    block bootstrap that resamples whole dates
  percentile_summary percentile intervals from bootstrap replicates

Nothing here reads a file or knows a column name, so the same code serves every notebook.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

Z = 1.96            # 95% intervals
B = 2000            # bootstrap replicates
SEED = 4981


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
    """Brier score of p, and of a constant forecast equal to the sample mean of y."""
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


def fit_logit(p, y, groups=None, extra=None):
    """Logit of y on [1, logit(p)], plus the columns of `extra` when given.

    The coefficients come back in that order: intercept, slope on logit(p), then one per
    extra column. With `groups`, the covariance is cluster-robust.
    """
    columns = [np.ones(len(p)), logit(p)]
    if extra is not None:
        extra = np.asarray(extra, dtype=float)
        columns.append(extra.reshape(len(p), -1))
    X = np.column_stack(columns)
    model = sm.Logit(np.asarray(y, float), X)
    if groups is None:
        return model.fit(disp=0)
    codes = pd.factorize(pd.Series(groups))[0]
    return model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": codes})


def lr_test(p, y, fit=None) -> dict:
    """LR test of a = 0 and b = 1 in logit P(y = 1) = a + b logit(p), on 2 df."""
    fit = fit if fit is not None else fit_logit(p, y)
    ll_r, ll_u = loglik_restricted(p, y), float(fit.llf)
    lr = 2 * (ll_u - ll_r)
    return {"ll_restricted": ll_r, "ll_unrestricted": ll_u, "lr_stat": lr, "df": 2,
            "p_value": float(stats.chi2.sf(lr, 2))}


def date_blocks(dates) -> list[np.ndarray]:
    """Row positions of each distinct date, dates in sorted order, positions ascending."""
    positions = pd.Series(np.arange(len(dates)))
    return [g.to_numpy() for _, g in positions.groupby(np.asarray(dates))]


def block_bootstrap(dates, statistic: Callable[[np.ndarray], Mapping[str, float]],
                    n_boot: int = B, seed: int = SEED) -> pd.DataFrame:
    """Block bootstrap by date.

    Each replicate draws as many dates as the sample has, with replacement, and takes every
    row of each drawn date. `statistic` receives the row positions of one replicate and
    returns a dict of numbers. The result has one row per replicate.
    """
    rng = np.random.default_rng(seed)
    blocks = date_blocks(dates)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        out.append(dict(statistic(idx)))
    return pd.DataFrame(out)


def percentile_summary(boot: pd.DataFrame, point: Mapping[str, float], n_dates: int,
                       seed: int = SEED) -> pd.DataFrame:
    """Point estimate, bootstrap mean and SE, and the 95% percentile interval per statistic."""
    rows = []
    for stat, est in point.items():
        x = boot[stat].dropna()
        rows.append({"statistic": stat, "estimate": est, "boot_mean": x.mean(),
                     "boot_se": x.std(ddof=1), "boot_lo": x.quantile(0.025),
                     "boot_hi": x.quantile(0.975), "B": len(x), "n_dates": n_dates,
                     "seed": seed})
    return pd.DataFrame(rows)

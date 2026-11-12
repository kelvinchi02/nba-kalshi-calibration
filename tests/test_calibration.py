"""Checks of the Stage 8 machinery on known values and synthetic data (no real results)."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportion_confint

from calibration import (assign_bins, block_bootstrap, brier, fit_logit, logit,
                         loglik_restricted, lr_test, wilson)


@pytest.mark.parametrize("k,m", [(0, 10), (5, 10), (61, 123), (122, 122), (1, 1229)])
def test_wilson_matches_statsmodels(k, m):
    lo, hi = wilson(k, m)
    ref_lo, ref_hi = proportion_confint(k, m, alpha=0.05, method="wilson")
    # wilson() uses z = 1.96; statsmodels uses the exact normal quantile 1.95996...
    assert lo == pytest.approx(ref_lo, abs=1e-4) and hi == pytest.approx(ref_hi, abs=1e-4)
    eps = 1e-12  # float rounding at k = 0 and k = m
    assert -eps <= lo <= k / m + eps and k / m - eps <= hi <= 1 + eps


def test_equal_count_bins():
    counts = np.bincount(assign_bins(1229))
    assert sorted(counts.tolist()) == [122] + [123] * 9
    assert np.all(np.diff(assign_bins(1229)) >= 0)  # monotone in sorted order


def test_brier_and_baseline():
    y = np.array([1, 0, 1, 1])
    assert brier(y.astype(float), y)["brier"] == 0
    out = brier(np.full(4, 0.5), y)
    assert out["brier"] == pytest.approx(0.25)
    assert out["brier_baseline"] == pytest.approx(0.75 * 0.25)  # ybar (1 - ybar)
    assert out["brier_skill"] == pytest.approx(1 - 0.25 / 0.1875)


def test_restricted_loglik_equals_logit_at_a0_b1():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200)
    y = (rng.uniform(size=200) < p).astype(int)
    # logistic(0 + 1 * logit(p)) == p, so the restricted log-likelihood is the model's at (0, 1)
    eta = logit(p)
    ll_model = np.sum(y * eta - np.log1p(np.exp(eta)))
    assert loglik_restricted(p, y) == pytest.approx(ll_model)


def test_lr_test_detects_miscalibration_only_when_present():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    calibrated = lr_test(p, y)
    assert calibrated["lr_stat"] >= 0 and calibrated["p_value"] > 0.001
    fit = fit_logit(p, y)
    assert fit.params[0] == pytest.approx(0, abs=0.06)
    assert fit.params[1] == pytest.approx(1, abs=0.06)
    # Prices pushed toward 0.5 (b > 1 needed) must be rejected
    shrunk = 1 / (1 + np.exp(-0.5 * logit(p)))
    assert lr_test(shrunk, y)["p_value"] < 1e-6


def test_cluster_se_runs_and_differs_from_iid():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.1, 0.9, 600)
    y = (rng.uniform(size=600) < p).astype(int)
    groups = np.repeat(np.arange(30), 20)
    iid, clus = fit_logit(p, y), fit_logit(p, y, groups=groups)
    assert np.allclose(iid.params, clus.params)
    assert not np.allclose(iid.bse, clus.bse)


def test_block_bootstrap_reproducible_and_whole_dates():
    rng = np.random.default_rng(3)
    n = 300
    d = pd.DataFrame({"p_home": rng.uniform(0.1, 0.9, n),
                      "game_date": np.repeat(pd.date_range("2025-10-21", periods=30), 10)})
    d["home_win"] = (rng.uniform(size=n) < d["p_home"]).astype(int)
    d = d.sort_values("p_home").reset_index(drop=True)
    d["bin"] = assign_bins(n)
    a = block_bootstrap(d, n_boot=20, seed=7)
    b = block_bootstrap(d, n_boot=20, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert (a["n"] % 10 == 0).all() and (a["n"] == 300).all()  # 30 whole dates of 10 games

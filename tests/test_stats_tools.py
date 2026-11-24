"""Checks of stats_tools.py on known values and synthetic data (no real results)."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

import calibration
import stats_tools
from stats_tools import (block_bootstrap, brier, date_blocks, fit_logit, logit,
                         loglik_restricted, lr_test, percentile_summary, wilson)


def synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated prices and outcomes."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    return p, (rng.uniform(size=n) < p).astype(int)


def test_calibration_module_uses_the_same_functions():
    # calibration.py must import the statistics, not keep its own copies
    for name in ("logit", "wilson", "brier", "loglik_restricted", "fit_logit", "lr_test"):
        assert getattr(calibration, name) is getattr(stats_tools, name)
    assert (stats_tools.SEED, stats_tools.B, stats_tools.Z) == (4981, 2000, 1.96)


def test_logit_known_values():
    assert logit(0.5) == 0
    assert logit(0.75) == pytest.approx(np.log(3))
    assert np.allclose(logit([0.2, 0.8]), [-np.log(4), np.log(4)])


@pytest.mark.parametrize("k,m", [(0, 10), (5, 10), (61, 123), (122, 122), (1, 1229)])
def test_wilson_matches_statsmodels(k, m):
    lo, hi = wilson(k, m)
    ref_lo, ref_hi = proportion_confint(k, m, alpha=0.05, method="wilson")
    assert lo == pytest.approx(ref_lo, abs=1e-4) and hi == pytest.approx(ref_hi, abs=1e-4)


def test_wilson_empty_bin_is_nan():
    assert np.isnan(wilson(0, 0)).all()


def test_brier_and_baseline():
    y = np.array([1, 0, 1, 1])
    assert brier(y.astype(float), y)["brier"] == 0
    out = brier(np.full(4, 0.5), y)
    assert out["n"] == 4 and out["home_win_rate"] == 0.75
    assert out["brier"] == pytest.approx(0.25)
    assert out["brier_baseline"] == pytest.approx(0.1875)
    assert out["brier_diff"] == pytest.approx(0.0625)
    assert out["brier_skill"] == pytest.approx(1 - 0.25 / 0.1875)


def test_restricted_loglik_by_hand():
    p, y = np.array([0.8, 0.3]), np.array([1, 0])
    assert loglik_restricted(p, y) == pytest.approx(np.log(0.8) + np.log(0.7))


def test_fit_logit_matches_direct_statsmodels_fit():
    p, y = synthetic(500, 0)
    fit = fit_logit(p, y)
    ref = sm.Logit(y, sm.add_constant(logit(p))).fit(disp=0)
    assert np.allclose(fit.params, ref.params) and np.allclose(fit.bse, ref.bse)


def test_fit_logit_extra_covariate_recovers_its_coefficient():
    rng = np.random.default_rng(1)
    n = 40000
    p = rng.uniform(0.1, 0.9, n)
    gap = rng.normal(size=n)
    true_a, true_b, true_c = 0.1, 0.9, -0.3
    prob = 1 / (1 + np.exp(-(true_a + true_b * logit(p) + true_c * gap)))
    y = (rng.uniform(size=n) < prob).astype(int)
    fit = fit_logit(p, y, extra=gap)
    assert len(fit.params) == 3
    assert fit.params == pytest.approx([true_a, true_b, true_c], abs=0.05)
    # two extra columns give two extra coefficients, in column order
    two = fit_logit(p, y, extra=np.column_stack([gap, rng.normal(size=n)]))
    assert len(two.params) == 4 and two.params[2] == pytest.approx(true_c, abs=0.05)


def test_cluster_se_matches_statsmodels_and_differs_from_iid():
    p, y = synthetic(600, 2)
    groups = np.repeat([f"T{i:02d}" for i in range(30)], 20)  # team names, not codes
    iid, clus = fit_logit(p, y), fit_logit(p, y, groups=groups)
    assert np.allclose(iid.params, clus.params)
    assert not np.allclose(iid.bse, clus.bse)
    ref = sm.Logit(y, sm.add_constant(logit(p))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": np.repeat(np.arange(30), 20)})
    assert np.allclose(clus.bse, ref.bse)


def test_lr_test_statistic_and_rejection():
    p, y = synthetic(20000, 3)
    out = lr_test(p, y)
    assert out["df"] == 2 and out["lr_stat"] >= 0
    assert out["lr_stat"] == pytest.approx(2 * (out["ll_unrestricted"] - out["ll_restricted"]))
    assert out["p_value"] > 0.001                      # calibrated prices are not rejected
    shrunk = 1 / (1 + np.exp(-0.5 * logit(p)))         # prices pushed toward 0.5
    assert lr_test(shrunk, y)["p_value"] < 1e-6


def test_date_blocks_cover_every_row_once_in_date_order():
    dates = pd.to_datetime(["2025-10-22", "2025-10-21", "2025-10-22", "2025-10-23",
                            "2025-10-21"]).date
    blocks = date_blocks(dates)
    assert [b.tolist() for b in blocks] == [[1, 4], [0, 2], [3]]


def test_block_bootstrap_draws_whole_dates_and_is_reproducible():
    dates = np.repeat(pd.date_range("2025-10-21", periods=30).date, 10)
    seen = []

    def statistic(idx):
        seen.append(idx)
        return {"n": len(idx), "mean_pos": idx.mean()}

    a = block_bootstrap(dates, statistic, n_boot=25, seed=7)
    b = block_bootstrap(dates, lambda idx: {"n": len(idx), "mean_pos": idx.mean()},
                        n_boot=25, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 25 and (a["n"] == 300).all()      # 30 dates of 10 games each
    for idx in seen:                                   # each drawn date arrives complete
        day, count = np.unique(idx // 10, return_counts=True)
        assert (count % 10 == 0).all()
        assert all(set(idx[idx // 10 == d]) == set(range(d * 10, d * 10 + 10)) for d in day)
    other = block_bootstrap(dates, lambda idx: {"mean_pos": idx.mean()}, n_boot=25, seed=8)
    assert not np.allclose(a["mean_pos"], other["mean_pos"])


def test_block_bootstrap_unequal_dates_changes_sample_size():
    dates = np.array(["d1"] * 1 + ["d2"] * 5 + ["d3"] * 12)
    boot = block_bootstrap(dates, lambda idx: {"n": len(idx)}, n_boot=200, seed=1)
    assert boot["n"].nunique() > 1 and boot["n"].between(3, 36).all()


def test_percentile_summary():
    boot = pd.DataFrame({"x": np.arange(1001.0), "y": [np.nan] + [2.0] * 1000})
    out = percentile_summary(boot, {"x": 500.0, "y": 2.0}, n_dates=164).set_index("statistic")
    assert out.loc["x", "boot_lo"] == pytest.approx(25) and out.loc["x", "boot_hi"] == pytest.approx(975)
    assert out.loc["x", "estimate"] == 500 and out.loc["x", "boot_mean"] == 500
    assert out.loc["y", "B"] == 1000                   # missing replicates are dropped
    assert (out["n_dates"] == 164).all() and (out["seed"] == 4981).all()

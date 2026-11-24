"""Stage 10 shared tools: bootstrap p value, joint Wald test, Holm, and the tests table."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import all_tests
from stats_tools import bootstrap_p_value, holm, wald_joint


def test_bootstrap_p_value_by_hand():
    # 9 replicates: 2 at or below 0, 8 at or above 0 (one is exactly 0).
    x = [-1.0, 0.0, 1, 2, 3, 4, 5, 6, 7]
    assert bootstrap_p_value(x) == pytest.approx(2 * (1 + 2) / 10)
    # all on one side: the smallest value the formula can give
    assert bootstrap_p_value(np.ones(1999)) == pytest.approx(2 / 2000)
    # centered on the null: capped at 1
    assert bootstrap_p_value([-1.0, 1.0]) == 1.0


def test_bootstrap_p_value_is_symmetric_uses_the_null_and_skips_missing():
    x = np.array([0.5, 1.0, 1.5, 2.0, -0.2, np.nan])
    assert bootstrap_p_value(x) == bootstrap_p_value(-x)
    assert bootstrap_p_value(x, null=1.0) == bootstrap_p_value(x[:5] - 1.0)
    assert np.isnan(bootstrap_p_value([np.nan]))


def test_holm_by_hand_and_against_statsmodels():
    p = np.array([0.04, 0.001, 0.03, 0.5])
    # sorted: 0.001, 0.03, 0.04, 0.5 -> 4*0.001, 3*0.03, 2*0.04 (kept at 0.09), 1*0.5
    assert holm(p) == pytest.approx([0.09, 0.004, 0.09, 0.5])
    rng = np.random.default_rng(1)
    q = rng.uniform(0, 0.2, 8)
    assert holm(q) == pytest.approx(multipletests(q, method="holm")[1])
    assert holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError):
        holm([0.1, np.nan])


def test_holm_is_never_above_the_bonferroni_bound():
    rng = np.random.default_rng(2)
    q = rng.uniform(0, 0.1, 8)
    assert (holm(q) <= all_tests.bonferroni_bound(q, 8) + 1e-15).all()


def test_wald_joint_matches_statsmodels():
    rng = np.random.default_rng(3)
    X = sm.add_constant(rng.normal(size=(400, 3)))
    y = (rng.uniform(size=400) < 1 / (1 + np.exp(-(0.2 + X[:, 1])))).astype(float)
    groups = np.arange(400) % 20
    fit = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    mine = wald_joint(fit.params, fit.cov_params(), [2, 3])
    theirs = fit.wald_test(np.eye(4)[[2, 3]], use_f=False, scalar=True)
    assert mine["chi2"] == pytest.approx(float(theirs.statistic))
    assert mine["p_value"] == pytest.approx(float(theirs.pvalue))
    assert mine["df"] == 2
    # with one coefficient it is the squared z statistic
    one = wald_joint(fit.params, fit.cov_params(), [1])
    assert one["chi2"] == pytest.approx((fit.params[1] / fit.bse[1]) ** 2)


def part_tables() -> list[pd.DataFrame]:
    """One made-up table per part, covering the eight primary tests and two secondary."""
    p = dict(zip(all_tests.PRIMARY, [0.001, 0.2, 0.03, 0.5, 0.004, 0.8, 0.01, 0.06]))
    rows = [all_tests.wald_row("x", t, t, "null", 100, 0.5, 0.1, p[t])
            for t in all_tests.PRIMARY[:4]]
    rows.append(all_tests.wald_row("x", "extra_wald", "d", "null", 100, 0.5, 0.1, 0.0001))
    more = [all_tests.joint_row("y", all_tests.PRIMARY[5], "d", "null", 50, 3.2, 2, p[
        all_tests.PRIMARY[5]], "LR", "LR chi2")]
    more += [all_tests.wald_row("y", t, t, "null", 50, 0.5, 0.1, p[t])
             for t in (all_tests.PRIMARY[4], all_tests.PRIMARY[6], all_tests.PRIMARY[7])]
    more.append(all_tests.bootstrap_row("y", "extra_boot", "d", "null", 30, 0.2,
                                        np.linspace(-1, 3, 401), unit="trades"))
    return [all_tests.table(rows), all_tests.table(more)]


def test_rows_fill_the_common_layout():
    a, b = part_tables()
    assert list(a.columns) == all_tests.COLUMNS
    w = a.iloc[0]
    assert w["role"] == "primary" and w["interval_type"] == "clustered Wald"
    assert w["ci_lo"] == pytest.approx(0.5 - 1.96 * 0.1) and w["statistic"] == pytest.approx(5)
    assert a.iloc[4]["role"] == "secondary"
    j = b.iloc[0]
    assert np.isnan(j["estimate"]) and j["df"] == 2 and j["statistic_name"] == "LR"
    boot = b.iloc[-1]
    assert boot["interval_type"] == "bootstrap percentile" and boot["unit"] == "trades"
    assert boot["ci_lo"] == pytest.approx(-0.9) and boot["ci_hi"] == pytest.approx(2.9)
    assert boot["p_raw"] == pytest.approx(2 * (1 + 101) / 402)


def test_wald_row_uses_the_null_value_for_the_statistic():
    r = all_tests.wald_row("x", "t", "b", "b = 1", 10, 0.9, 0.05, 0.0455, null_value=1.0)
    assert r["statistic"] == pytest.approx(-2.0)


def test_combine_adjusts_only_the_primary_tests():
    out = all_tests.combine(part_tables())
    primary = out[out["role"] == "primary"].set_index("test_id")
    assert len(primary) == 8 and (primary["holm_family_size"] == 8).all()
    expected = multipletests(primary["p_raw"].to_numpy(), method="holm")[1]
    assert primary["p_holm"].to_numpy() == pytest.approx(expected)
    assert primary.loc[all_tests.PRIMARY[0], "p_holm"] == pytest.approx(0.008)
    assert primary["significant_holm"].tolist() == [int(v < 0.05) for v in expected]
    secondary = out[out["role"] == "secondary"]
    assert secondary["p_holm"].isna().all() and secondary["significant_holm"].isna().all()
    # the secondary p value of 0.0001 does not change the family
    assert len(out) == 10


def test_combine_refuses_an_incomplete_family_and_duplicates():
    a, b = part_tables()
    with pytest.raises(ValueError, match="missing"):
        all_tests.combine([a])
    with pytest.raises(ValueError, match="duplicate"):
        all_tests.combine([a, a, b])
    # a smaller family can be passed explicitly (plan 10.1, if D2 is stopped)
    seven = tuple(t for t in all_tests.PRIMARY if t != "injury_event_mean")
    b7 = b[b["test_id"] != "injury_event_mean"]
    out = all_tests.combine([a, b7], family=seven)
    assert (out.loc[out["role"] == "primary", "holm_family_size"] == 7).all()


def test_table_refuses_a_missing_p_value():
    row = all_tests.wald_row("x", "t", "d", "null", 10, 0.5, 0.1, np.nan)
    with pytest.raises(ValueError, match="raw p value"):
        all_tests.table([row])


def test_exploratory_tests_are_listed_but_never_adjusted():
    a, b = part_tables()
    extra = all_tests.mark_exploratory(all_tests.table([
        all_tests.wald_row("x", "exploratory_one", "d", "null", 40, 0.5, 0.1, 0.00001),
        all_tests.bootstrap_row("y", "exploratory_two", "d", "null", 30, 0.2,
                                np.linspace(-1, 3, 401), unit="events")]))
    assert (extra["role"] == "exploratory").all()
    out = all_tests.combine([a, b, extra])
    base = all_tests.combine([a, b])
    primary = out[out["role"] == "primary"].set_index("test_id")["p_holm"]
    assert primary.to_dict() == base[base["role"] == "primary"].set_index(
        "test_id")["p_holm"].to_dict()                  # the tiny exploratory p changes nothing
    exploratory = out[out["role"] == "exploratory"]
    assert len(exploratory) == 2 and exploratory["p_holm"].isna().all()
    assert exploratory["holm_family_size"].isna().all()
    assert exploratory["significant_holm"].isna().all()
    with pytest.raises(ValueError, match="primary"):
        all_tests.mark_exploratory(a)

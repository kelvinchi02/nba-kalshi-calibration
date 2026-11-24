"""Stage 10.5 machinery on small hand-made and synthetic data (no real results)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import subgroups as sg
from stats_tools import fit_logit


def test_liquidity_groups_are_equal_count_with_ties_broken_by_game_id():
    total = pd.Series([50.0, 10.0, 30.0, 30.0, 70.0, 20.0, 90.0], index=list("abcdefg"))
    ids = pd.Series(["g7", "g1", "g4", "g3", "g5", "g2", "g6"], index=total.index)
    out = sg.liquidity_group(total, ids)
    # sorted by (total, id): g1 10, g2 20, g3 30 | g4 30, g7 50 | g5 70, g6 90
    assert dict(zip(ids, out)) == {"g1": "low", "g2": "low", "g3": "low", "g4": "middle",
                                   "g7": "middle", "g5": "high", "g6": "high"}
    assert out.index.tolist() == list("abcdefg")
    big = sg.liquidity_group(pd.Series(np.arange(1229.0)),
                             pd.Series([f"{i:05d}" for i in range(1229)]))
    assert big.value_counts().to_dict() == {"low": 410, "middle": 410, "high": 409}


def test_season_half_puts_the_median_date_in_the_late_half():
    dates = pd.Series([date(2026, 1, d) for d in (1, 2, 2, 5, 9)])
    assert sg.median_date(dates) == pd.Timestamp("2026-01-02")
    assert sg.season_half_group(dates).tolist() == ["early", "late", "late", "late", "late"]
    even = pd.Series([date(2026, 1, d) for d in (1, 2, 4, 9)])     # median is January 3
    assert sg.season_half_group(even).tolist() == ["early", "early", "late", "late"]


def schedule() -> pd.DataFrame:
    return pd.DataFrame({
        "game_date": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 2), date(2026, 1, 4),
                      date(2026, 2, 1)],
        "home_team": ["AAA", "CCC", "EEE", "AAA", "BBB"],
        "away_team": ["BBB", "AAA", "FFF", "EEE", "CCC"]})


def test_back_to_back_looks_at_either_team_and_only_the_previous_day():
    s = schedule()
    out = sg.rest_group(s, s)
    # Jan 2 CCC-AAA: AAA played Jan 1 (as home). Jan 2 EEE-FFF: neither played Jan 1.
    # Jan 4 AAA-EEE: both last played Jan 2, two days before. Feb 1: Jan 31 has no games.
    assert out.tolist() == ["rested", "back_to_back", "rested", "rested", "rested"]
    away = pd.DataFrame({"game_date": [date(2026, 1, 3)], "home_team": ["ZZZ"],
                         "away_team": ["FFF"]})
    assert sg.rest_group(away, s).tolist() == ["back_to_back"]      # away team played Jan 2


def test_back_to_back_uses_games_outside_the_sample_too():
    # The sample holds only the second game. The first night is in the schedule.
    s = schedule()
    sample = s.iloc[[1]]
    assert sg.rest_group(sample, s).tolist() == ["back_to_back"]
    assert sg.rest_group(sample, s.iloc[1:]).tolist() == ["rested"]


def synthetic(n: int = 900, slope_late: float = 0.5, seed: int = 5) -> pd.DataFrame:
    """Games where the true slope is 1 in the early half and `slope_late` in the late half."""
    rng = np.random.default_rng(seed)
    lp = rng.normal(0.2, 0.9, n)
    day = np.arange(n) // 6
    late = day >= np.median(day)
    eta = np.where(late, slope_late * lp, lp)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    d = pd.DataFrame({
        "game_id": [f"g{i:04d}" for i in range(n)],
        "game_date": [date(2026, 1, 1) + timedelta(days=int(k)) for k in day],
        "home_team": [f"T{i % 30:02d}" for i in range(n)],
        "away_team": [f"T{(i + 7) % 30:02d}" for i in range(n)],
        "p_home": 1 / (1 + np.exp(-lp)), "home_win": y,
        "home_pregame_contracts": rng.uniform(1e3, 1e5, n),
        "away_pregame_contracts": rng.uniform(1e3, 1e5, n)})
    sched = d[["game_date", "home_team", "away_team"]]
    return sg.add_groups(d, sched)


def test_design_columns_and_names():
    X, names = sg.design([0.5, 0.75, 0.25], ["low", "high", "middle"], sg.SPLITS["liquidity"])
    assert names == ["intercept_a", "slope_b", "intercept_shift_middle",
                     "intercept_shift_high", "slope_shift_middle", "slope_shift_high"]
    assert X[:, 0].tolist() == [1, 1, 1]
    assert X[1].tolist() == pytest.approx([1, np.log(3), 0, 1, 0, np.log(3)])
    assert X[2].tolist() == pytest.approx([1, -np.log(3), 1, 0, -np.log(3), 0])


def test_interaction_model_reproduces_the_fits_within_each_group():
    d = synthetic()
    fit, names = sg.fit_interaction(d, "season_half")
    coef = dict(zip(names, fit.params))
    early, late = (d[d["season_half"] == g] for g in ("early", "late"))
    fe = fit_logit(early["p_home"], early["home_win"])
    fl = fit_logit(late["p_home"], late["home_win"])
    assert [coef["intercept_a"], coef["slope_b"]] == pytest.approx(fe.params, abs=1e-6)
    assert coef["intercept_a"] + coef["intercept_shift_late"] == pytest.approx(fl.params[0],
                                                                               abs=1e-6)
    assert coef["slope_b"] + coef["slope_shift_late"] == pytest.approx(fl.params[1], abs=1e-6)
    assert coef["slope_shift_late"] < -0.2              # the data were built with 1 and 0.5


def test_interaction_table_tests_and_joint_row():
    d = synthetic()
    fit, names = sg.fit_interaction(d, "liquidity")
    t = sg.interaction_table(fit, names, "liquidity", 30).set_index("term")
    assert t.index.tolist() == names + ["all_slope_shifts"]
    b = t.loc["slope_b"]
    assert b["null_value"] == 1 and b["statistic"] == pytest.approx(
        (b["estimate"] - 1) / b["se_cluster"])
    joint = t.loc["all_slope_shifts"]
    theirs = fit.wald_test(np.eye(6)[[4, 5]], use_f=False, scalar=True)
    assert joint["statistic"] == pytest.approx(float(theirs.statistic))
    assert joint["p_value"] == pytest.approx(float(theirs.pvalue)) and joint["df"] == 2
    # with two groups the joint test is the single interaction, squared
    fit2, names2 = sg.fit_interaction(d, "rest")
    t2 = sg.interaction_table(fit2, names2, "rest", 30).set_index("term")
    assert t2.loc["all_slope_shifts", "statistic"] == pytest.approx(
        t2.loc["slope_shift_back_to_back", "statistic"] ** 2)


def test_group_metrics_and_bins_use_only_the_groups_own_games():
    d = synthetic()
    m = sg.group_metrics(d, "season_half").set_index("group")
    late = d[d["season_half"] == "late"]
    assert m.loc["late", "n"] == len(late)
    assert m.loc["late", "brier"] == pytest.approx(((late["p_home"] - late["home_win"]) ** 2)
                                                   .mean())
    rate = late["home_win"].mean()
    assert m.loc["late", "brier_baseline"] == pytest.approx(rate * (1 - rate))
    fit = fit_logit(late["p_home"], late["home_win"], groups=late["home_team"])
    assert m.loc["late", "slope_b"] == pytest.approx(fit.params[1])
    assert m.loc["late", "b_se_cluster"] == pytest.approx(fit.bse[1])
    bins = sg.group_calibration_bins(d, "season_half")
    assert bins.groupby("group")["n"].sum().to_dict() == d["season_half"].value_counts().to_dict()
    assert (bins.groupby("group").size() == sg.N_BINS).all()
    assert bins[bins["group"] == "late"]["home_wins"].sum() == late["home_win"].sum()


def test_analyze_split_tables_and_tests():
    d = synthetic()
    out = sg.analyze_split(d, "season_half", n_boot=40, seed=1)
    diffs = out["brier_diff"].iloc[0]
    m = out["metrics"].set_index("group")
    assert diffs["brier_diff"] == pytest.approx(m.loc["late", "brier"] - m.loc["early", "brier"])
    assert diffs["boot_lo"] <= diffs["boot_hi"]
    boot = out["bootstrap"].set_index("statistic")
    assert boot.loc["b_late", "estimate"] == pytest.approx(m.loc["late", "slope_b"], abs=1e-6)
    assert (boot["B"] == 40).all()
    tests = out["tests"].set_index("test_id")
    primary = tests[tests["role"] == "primary"]
    assert primary.index.tolist() == ["subgroup_season_half_slope"]
    inter = out["interaction"].set_index("term")
    assert primary["p_raw"].iloc[0] == inter.loc["slope_shift_late", "p_value"]
    assert tests.loc["subgroup_season_half_late_b", "null_hypothesis"] == "b = 1"
    assert tests.loc["subgroup_season_half_brier_late_minus_early",
                     "p_method"] == "block bootstrap"
    # 1 slope, 1 intercept shift, 2 groups x (a, b), 1 Brier difference
    assert len(tests) == 7


def test_liquidity_split_has_a_joint_primary_test_and_three_brier_pairs():
    d = synthetic()
    out = sg.analyze_split(d, "liquidity", n_boot=30, seed=1)
    tests = out["tests"].set_index("test_id")
    joint = tests.loc["subgroup_liquidity_slope"]
    assert joint["role"] == "primary" and joint["statistic_name"] == "chi2" and joint["df"] == 2
    assert np.isnan(joint["estimate"])
    assert tests.loc["subgroup_liquidity_slope_high", "role"] == "secondary"
    assert len(out["brier_diff"]) == 3
    # 1 joint, 2 slopes, 2 intercept shifts, 3 groups x (a, b), 3 Brier differences
    assert len(tests) == 14


def test_bootstrap_keeps_group_labels_fixed_and_survives_a_failed_fit():
    d = synthetic(n=120)
    labels = d["season_half"].copy()
    boot = sg.bootstrap_split(d, "season_half", n_boot=25, seed=3)
    assert d["season_half"].equals(labels) and len(boot) == 25
    assert boot["brier_early"].notna().all()
    # A group with one outcome only cannot be fitted: coefficients missing, Brier kept.
    d2 = d.copy()
    d2.loc[d2["season_half"] == "late", "home_win"] = 1
    boot2 = sg.bootstrap_split(d2, "season_half", n_boot=5, seed=3)
    assert boot2["brier_late"].notna().all()
    assert boot2["slope_shift_late"].isna().all()
    assert boot2["slope_b"].isna().all()


# ---------------------------------------------------------------- exploratory (plan X2)

def month_dates(counts: dict) -> pd.Series:
    """Game dates with the given number of games per month, e.g. {"2025-10": 3}."""
    out = []
    for month, n in counts.items():
        y, m = int(month[:4]), int(month[5:])
        out += [date(y, m, 1 + i % 28) for i in range(n)]
    return pd.Series(out)


def test_month_group_joins_a_small_first_or_last_month_with_its_neighbor():
    dates = month_dates({"2025-10": 3, "2025-11": 6, "2025-12": 6, "2026-01": 6, "2026-04": 2})
    g = sg.month_group(dates, min_games=5)
    assert g.value_counts().to_dict() == {"2025-10 to 2025-11": 9, "2025-12": 6,
                                          "2026-01 to 2026-04": 8}
    assert sg.month_group(dates, min_games=2).nunique() == 5      # nothing is small
    # A small month in the middle stays as it is: only the ends are joined.
    middle = month_dates({"2025-11": 6, "2025-12": 1, "2026-01": 6})
    assert sorted(sg.month_group(middle, min_games=5).unique()) == ["2025-11", "2025-12",
                                                                    "2026-01"]


def test_month_title():
    assert sg.month_title("2026-01") == "Jan"
    assert sg.month_title("2025-10 to 2025-11") == "Oct to Nov"


def test_monthly_slopes_match_a_direct_fit_and_tests_are_exploratory():
    d = synthetic()                                  # 150 days from January 1, 6 games a day
    table = sg.monthly_slopes(d, min_games=185)       # May has 180 games: it joins April
    assert table["n"].sum() == len(d)
    assert table["month"].tolist() == ["2026-01", "2026-02", "2026-03", "2026-04 to 2026-05"]
    feb = d[pd.to_datetime(d["game_date"]).dt.month == 2]
    fit = fit_logit(feb["p_home"], feb["home_win"], groups=feb["home_team"])
    row = table.set_index("month").loc["2026-02"]
    assert row["n"] == len(feb) and row["month_label"] == "Feb"
    assert row["slope_b"] == pytest.approx(fit.params[1])
    assert row["b_se_cluster"] == pytest.approx(fit.bse[1])
    assert row["b_ci_hi"] == pytest.approx(fit.params[1] + 1.96 * fit.bse[1])
    tests = sg.monthly_tests(table)
    assert (tests["role"] == "exploratory").all() and len(tests) == 4
    assert tests["test_id"].tolist()[0] == "exploratory_slope_2026_01"
    assert tests["test_id"].tolist()[-1] == "exploratory_slope_2026_04_2026_05"
    assert (tests["null_hypothesis"] == "b = 1").all()
    feb_test = tests.set_index("test_id").loc["exploratory_slope_2026_02"]
    assert feb_test["statistic"] == pytest.approx((fit.params[1] - 1) / fit.bse[1])

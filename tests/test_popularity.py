"""Stage 9.1 machinery on small hand-made and synthetic data (no real results)."""
import numpy as np
import pandas as pd
import pytest

import metro_population as mp
import popularity as pop
from stats_tools import fit_logit, logit


def at0_three_teams() -> pd.DataFrame:
    """Four games among AAA, BBB and CCC with round volumes."""
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4"],
        "home_team": ["AAA", "BBB", "CCC", "AAA"],
        "away_team": ["BBB", "CCC", "AAA", "CCC"],
        "home_pregame_contracts": [100.0, 40.0, 10.0, 300.0],
        "away_pregame_contracts": [20.0, 30.0, 200.0, 50.0],
    })


def test_leave_one_out_mean_excludes_the_row_itself():
    team = pd.Series(["A", "A", "A", "B", "B"])
    value = pd.Series([1.0, 2.0, 6.0, 10.0, 30.0])
    assert pop.leave_one_out_mean(team, value).tolist() == [4.0, 3.5, 1.5, 30.0, 10.0]


def test_volume_gap_uses_each_teams_other_games_only():
    out = pop.volume_gap_leave_one_out(at0_three_teams()).set_index("game_id")
    # AAA plays g1 (100), g3 (200, away) and g4 (300). Without g1: mean of 200 and 300.
    assert out.loc["g1", "home_volume_loo"] == 250
    # BBB plays g1 (20, away) and g2 (40). Without g1: 40.
    assert out.loc["g1", "away_volume_loo"] == 40
    assert out.loc["g1", "gap_volume"] == pytest.approx(np.log(250 / 40))
    # CCC plays g2 (30), g3 (10) and g4 (50). Without g3: mean of 30 and 50.
    assert out.loc["g3", "home_volume_loo"] == 40
    assert out.loc["g3", "away_volume_loo"] == 200      # AAA without g3: 100 and 300


def test_changing_a_games_own_volume_does_not_change_its_own_gap():
    base = pop.volume_gap_leave_one_out(at0_three_teams()).set_index("game_id")
    changed = at0_three_teams()
    changed.loc[0, ["home_pregame_contracts", "away_pregame_contracts"]] = [9999.0, 7777.0]
    after = pop.volume_gap_leave_one_out(changed).set_index("game_id")
    assert after.loc["g1", "gap_volume"] == base.loc["g1", "gap_volume"]
    assert after.loc["g4", "gap_volume"] != base.loc["g4", "gap_volume"]


def test_metro_gap_sign_and_shared_metro():
    at0 = pd.DataFrame({"game_id": ["g1", "g2"], "home_team": ["BIG", "BIG"],
                        "away_team": ["SML", "TWN"]})
    metro = pd.DataFrame({"team": ["BIG", "SML", "TWN"], "population": [8_000_000, 2_000_000,
                                                                        8_000_000]})
    out = pop.metro_gap(at0, metro).set_index("game_id")
    assert out.loc["g1", "gap_metro"] == pytest.approx(np.log(4))   # home is the bigger metro
    assert out.loc["g2", "gap_metro"] == 0                          # same metro area
    with pytest.raises(ValueError, match="no metro population"):
        pop.metro_gap(at0, metro[metro["team"] != "SML"])


def synthetic_games(n: int, c: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"p_home": rng.uniform(0.1, 0.9, n), "gap": rng.normal(0, 0.8, n),
                      "home_team": rng.integers(0, 30, n),
                      "game_date": rng.integers(0, 150, n)})
    prob = 1 / (1 + np.exp(-(logit(d["p_home"]) + c * d["gap"])))
    d["home_win"] = (rng.uniform(size=n) < prob).astype(int)
    return d


def test_gap_model_recovers_a_planted_overpricing():
    d = synthetic_games(30000, c=-0.4, seed=0)
    fit = pop.fit_gap_model(d, "gap")
    assert fit.params == pytest.approx([0, 1, -0.4], abs=0.06)
    table = pop.model_table(fit, "volume", 30).set_index("term")
    assert table.index.tolist() == ["intercept_a", "slope_b", "gap_c"]
    assert table["null_value"].tolist() == [0, 1, 0]
    assert table.loc["gap_c", "p_value"] < 1e-6 and table.loc["gap_c", "ci_hi"] < 0
    assert table.loc["slope_b", "z"] == pytest.approx(
        (table.loc["slope_b", "estimate"] - 1) / table.loc["slope_b", "se_cluster"])
    # the clustered fit and the plain fit share coefficients, not standard errors
    plain = pop.fit_gap_model(d, "gap", clustered=False)
    assert np.allclose(plain.params, fit.params) and not np.allclose(plain.bse, fit.bse)


def test_gap_model_does_not_reject_when_there_is_no_effect():
    d = synthetic_games(30000, c=0.0, seed=1)
    table = pop.model_table(pop.fit_gap_model(d, "gap"), "volume", 30).set_index("term")
    assert table.loc["gap_c", "p_value"] > 0.001


def test_average_effect_by_hand_and_sign():
    p, gap = np.array([0.5, 0.5]), np.array([0.0, 0.0])
    # eta = 0 for both games; a step of 1 with c = log(3) moves 0.5 to 0.75
    assert pop.average_effect([0.0, 1.0, np.log(3)], p, gap, 1.0) == pytest.approx(0.25)
    assert pop.average_effect([0.0, 1.0, 0.0], p, gap, 1.0) == 0
    assert pop.average_effect([0.0, 1.0, -0.3], p, gap, 1.0) < 0


def test_effect_estimates_use_the_sample_sd():
    d = synthetic_games(2000, c=-0.4, seed=2)
    fit = pop.fit_gap_model(d, "gap")
    out = pop.effect_estimates(fit, d, "gap")
    assert out["gap_sd"] == pytest.approx(np.std(d["gap"], ddof=1))
    assert out["c_per_sd"] == pytest.approx(fit.params[2] * out["gap_sd"])
    assert out["avg_effect_per_sd"] == pytest.approx(
        pop.average_effect(fit.params, d["p_home"], d["gap"], out["gap_sd"]))


def test_bootstrap_is_reproducible_and_feeds_the_tables():
    d = synthetic_games(1500, c=-0.4, seed=3)
    fit = pop.fit_gap_model(d, "gap")
    effects = pop.effect_estimates(fit, d, "gap")
    a = pop.bootstrap_gap_model(d, "gap", n_boot=40, seed=7)
    b = pop.bootstrap_gap_model(d, "gap", n_boot=40, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert np.allclose(a["c_per_sd"], a["gap_c"] * effects["gap_sd"])  # full-sample SD
    summary = pop.bootstrap_table(a, fit, effects, "volume", d["game_date"].nunique())
    assert summary["statistic"].tolist() == ["intercept_a", "slope_b", "gap_c", "c_per_sd",
                                             "avg_effect_per_sd"]
    assert (summary["measure"] == "volume").all() and (summary["B"] == 40).all()
    row = pop.effects_table(effects, summary, "volume").iloc[0]
    assert row["c_per_sd_boot_lo"] <= row["c_per_sd_boot_hi"]
    assert row["gap_c"] == fit.params[2]


def test_measure_correlation_known_cases():
    teams = pd.DataFrame({"mean_pregame_contracts": [100.0, 200.0, 400.0, 800.0],
                          "metro_population": [1e6, 2e6, 4e6, 8e6]})
    out = pop.measure_correlation(teams).iloc[0]
    assert out["n_teams"] == 4
    assert out["pearson_log_log"] == pytest.approx(1) and out["spearman_rank"] == pytest.approx(1)
    teams["metro_population"] = teams["metro_population"].to_numpy()[::-1]
    assert pop.measure_correlation(teams).iloc[0]["spearman_rank"] == pytest.approx(-1)


# ---------------------------------------------------------------- metro population table

def census_like() -> pd.DataFrame:
    rows = [{"area_code": code, "metro_area": f"{city}-Somewhere, ST", "population": 1000 + i}
            for i, (code, city) in enumerate(sorted(set(mp.TEAM_CBSA.values())))]
    return pd.DataFrame(rows)


def test_build_table_has_30_teams_and_shared_metros():
    table = mp.build_table(census_like(), {"TOR": ("Toronto (CMA), Ontario", 7_000_000)},
                           retrieved="2026-11-23").set_index("team")
    assert len(table) == 30 and table.index.is_unique
    assert table.loc["LAL", "population"] == table.loc["LAC", "population"]
    assert table.loc["NYK", "population"] == table.loc["BKN", "population"]
    assert table.loc["TOR", "area_type"] == "CMA" and table.loc["TOR", "population"] == 7_000_000
    assert (table["reference_date"] == "2025-07-01").all()
    assert (table["retrieved"] == "2026-11-23").all()


def test_build_table_rejects_a_code_that_names_another_city():
    census = census_like()
    census.loc[census["area_code"] == "12060", "metro_area"] = "Augusta-Richmond County, GA-SC"
    with pytest.raises(ValueError, match="ATL"):
        mp.build_table(census, {"TOR": ("Toronto (CMA), Ontario", 1)}, retrieved="x")


def test_statcan_row_picks_total_all_ages_for_the_year():
    rows = [
        {"REF_DATE": "2024", "GEO": "Toronto", "DGUID": "D1", "Gender": "Total - gender",
         "Age group": "All ages", "VALUE": "10"},
        {"REF_DATE": "2025", "GEO": "Toronto", "DGUID": "D1", "Gender": "Total - gender",
         "Age group": "All ages", "VALUE": "11"},
        {"REF_DATE": "2025", "GEO": "Toronto", "DGUID": "D1", "Gender": "Men+",
         "Age group": "All ages", "VALUE": "5"},
        {"REF_DATE": "2025", "GEO": "Toronto", "DGUID": "D1", "Gender": "Total - gender",
         "Age group": "0 to 4 years", "VALUE": "1"},
    ]
    assert mp.statcan_row(rows, "D1", 2025) == ("Toronto", 11)
    with pytest.raises(ValueError, match="found 0"):
        mp.statcan_row(rows, "D2", 2025)


def test_committed_metro_table_is_complete():
    table = mp.load_metro_population()
    assert sorted(table["team"]) == sorted(list(mp.TEAM_CBSA) + list(mp.CANADA_TEAMS))
    assert (table["population"] > 500_000).all()
    assert table["source_url"].str.startswith("https://").all()
    by_team = table.set_index("team")["population"]
    assert by_team["LAL"] == by_team["LAC"] and by_team["NYK"] == by_team["BKN"]


def test_analyze_measure_returns_the_three_tables_for_the_named_gap():
    d = synthetic_games(1500, c=-0.4, seed=4).rename(columns={"gap": "gap_volume"})
    out = pop.analyze_measure(d, "volume", n_boot=30, seed=7)
    assert set(out) == {"logistic", "effects", "bootstrap"}
    logistic = out["logistic"].set_index("term")
    fit = pop.fit_gap_model(d, "gap_volume")
    assert logistic.loc["gap_c", "estimate"] == fit.params[2]
    assert logistic.loc["gap_c", "se_cluster"] == fit.bse[2]
    ci = out["bootstrap"].set_index("statistic")
    assert logistic.loc["gap_c", "boot_lo"] == ci.loc["gap_c", "boot_lo"]
    assert logistic.loc["slope_b", "boot_hi"] == ci.loc["slope_b", "boot_hi"]
    assert out["effects"]["measure"].tolist() == ["volume"]

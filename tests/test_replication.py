"""Stage 10.6 machinery on small hand-made and synthetic data (no real results)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import replication as rep
import strategy as st

TEAMS = [f"T{i:02d}" for i in range(30)]


def previous_snapshots() -> pd.DataFrame:
    """A made-up earlier season: AAA trades 4x BBB, CCC in between. Two horizons per game."""
    games = [("g1", "AAA", "BBB", 400.0, 100.0), ("g2", "BBB", "CCC", 100.0, 200.0),
             ("g3", "CCC", "AAA", 200.0, 400.0), ("g4", "AAA", "CCC", 400.0, 200.0),
             ("g5", "XXX", "AAA", 9e9, 9e9)]
    rows = []
    for gid, home, away, hv, av in games:
        for h in (24, 0):
            rows.append({"game_id": gid, "horizon_h": h, "main_sample": gid != "g5",
                         "home_team": home, "away_team": away,
                         "home_pregame_contracts": hv, "away_pregame_contracts": av})
    return pd.DataFrame(rows)


def test_previous_team_volume_is_the_season_average_over_main_sample_games():
    v = rep.previous_team_volume(previous_snapshots())
    assert v.to_dict() == {"AAA": 400.0, "BBB": 100.0, "CCC": 200.0}      # g5 is not main sample


def test_add_gap_sign_and_missing_team():
    v = rep.previous_team_volume(previous_snapshots())
    d = pd.DataFrame({"home_team": ["AAA", "BBB", "CCC"], "away_team": ["BBB", "AAA", "CCC"]})
    out = rep.add_gap(d, v)
    assert out[rep.GAP_COL].tolist() == pytest.approx([np.log(4), -np.log(4), 0.0])
    with pytest.raises(ValueError, match="No 2025-26 measure for: NEW"):
        rep.add_gap(pd.DataFrame({"home_team": ["NEW"], "away_team": ["AAA"]}), v)


def test_previous_threshold_is_the_quantile_of_absolute_gaps_of_the_earlier_season():
    sn = previous_snapshots()
    v = rep.previous_team_volume(sn)
    # |gaps| of g1..g4: log 4, log 2, log 2, log 2
    q = rep.previous_threshold(rep.at_tipoff(sn), v)
    assert q == pytest.approx(np.quantile([np.log(4), np.log(2), np.log(2), np.log(2)], 0.75))
    assert rep.previous_threshold(rep.at_tipoff(sn), v, quantile=0.5) == pytest.approx(np.log(2))


def new_snapshots(n: int = 240, seed: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    volume = pd.Series(np.exp(rng.normal(12, 0.6, 30)), index=TEAMS)
    rows = []
    for i in range(n):
        home, away = TEAMS[i % 30], TEAMS[(i * 7 + 3) % 30]
        if home == away:
            away = TEAMS[(i + 1) % 30]
        lp = rng.normal(0.2, 0.8)
        p = 1 / (1 + np.exp(-lp))
        win = int(rng.uniform() < p)
        ask_h = min(0.99, round(p + 0.01, 2))
        ask_a = min(0.99, round(1 - p + 0.01, 2))
        for h in (24, 0):
            rows.append({"game_id": f"n{i:04d}", "horizon_h": h, "main_sample": i != 5,
                         "game_date": date(2026, 10, 20) + timedelta(days=i // 8),
                         "tipoff_utc": pd.Timestamp("2026-10-21T00:00:00Z")
                         + pd.Timedelta(days=i // 8, minutes=i % 8),
                         "home_team": home, "away_team": away,
                         "p_home": np.nan if i == 9 else p, "home_win": win,
                         "home_ask": ask_h, "away_ask": ask_a})
    return pd.DataFrame(rows), volume


def test_build_sample_drops_non_main_and_missing_prices_and_bins_in_five():
    sn, volume = new_snapshots()
    d, n_missing = rep.build_sample(sn, volume)
    assert n_missing == 1 and len(d) == 238
    assert not {"n0005", "n0009"} & set(d["game_id"])
    assert d["p_home"].is_monotonic_increasing
    assert d["bin"].value_counts().sort_index().tolist() == [48, 48, 47, 48, 47]
    assert d["tradable"].all()
    row = d.iloc[0]
    assert row[rep.GAP_COL] == pytest.approx(np.log(volume[row["home_team"]]
                                                    / volume[row["away_team"]]))


def test_sample_counts_report_each_step_and_the_cutoff():
    sn, volume = new_snapshots()
    d, n_missing = rep.build_sample(sn, volume)
    gm = sn[sn["horizon_h"] == 0][["game_id", "game_date", "main_sample"]].copy()
    gm["matched"] = True
    counts = rep.sample_counts(gm, d, n_missing).set_index("group")["count"]
    assert counts["final regular-season games up to the cutoff date"] == 240
    assert counts["main sample (matched, not postponed_window)"] == 239
    assert counts["games in the analysis"] == 238


def test_calibration_tables_match_the_stage_8_functions():
    sn, volume = new_snapshots()
    d, _ = rep.build_sample(sn, volume)
    out = rep.calibration_tables(d, n_boot=30, seed=1)
    assert len(out["bins"]) == 5 and out["bins"]["n"].sum() == len(d)
    bs = out["brier"].iloc[0]
    assert bs["brier"] == pytest.approx(((d["p_home"] - d["home_win"]) ** 2).mean())
    logi = out["logistic"].set_index("term")
    assert logi.loc["slope_b", "null_value"] == 1 and "boot_lo" in logi.columns
    lr = out["lr"].iloc[0]
    assert lr["df"] == 2 and 0 <= lr["p_value"] <= 1
    stats = set(out["bootstrap"]["statistic"])
    assert {"brier_diff", "slope_b", "lr_stat", "bin5_win_rate"} <= stats
    assert "bin6_win_rate" not in stats
    width = rep.slope_width(out["logistic"], "2026-27")
    assert width["ci_width"] == pytest.approx(2 * 1.96 * logi.loc["slope_b", "se_cluster"])
    assert width["boot_width"] == pytest.approx(logi.loc["slope_b", "boot_hi"]
                                                - logi.loc["slope_b", "boot_lo"])


def tiny_games() -> pd.DataFrame:
    """Four games. gap > 0: the home team is the more popular one."""
    return pd.DataFrame({
        "game_id": ["a", "b", "c", "d"],
        "game_date": [date(2026, 11, 1), date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)],
        "tipoff_utc": pd.to_datetime(["2026-11-01T23:00Z", "2026-11-02T00:00Z",
                                      "2026-11-02T23:00Z", "2026-11-03T23:00Z"], utc=True),
        "home_team": ["H1", "H2", "H3", "H4"], "away_team": ["A1", "A2", "A3", "A4"],
        "home_win": [1, 1, 0, 1],
        "home_ask": [0.40, 0.70, 0.55, 0.61], "away_ask": [0.62, 0.32, 0.47, 0.41],
        rep.GAP_COL: [0.9, -0.8, 0.2, 0.0],
        "eligible": True, "tradable": True})


def test_opposite_rule_buys_the_more_popular_team_above_the_fixed_threshold():
    out = rep.opposite_rule(tiny_games(), threshold=0.5, n_boot=20, seed=1)
    rule = out["trades"][out["trades"]["strategy"] == "opposite_rule"].set_index("game_id")
    # a: gap 0.9, home more popular -> buy home at 0.40, 250 contracts, home won.
    a = rule.loc["a"]
    assert (a["status"], a["side"], a["team"], a["contracts"]) == ("trade", "home", "H1", 250)
    fee, _ = st.taker_fee(250, 0.40)
    assert a["profit"] == pytest.approx(250 - 100 - fee)
    # b: gap -0.8, away more popular -> buy away at 0.32, home won, so the trade lost.
    b = rule.loc["b"]
    assert (b["side"], b["team"], b["won"]) == ("away", "A2", 0)
    assert b["contracts"] == 312 and b["profit"] == pytest.approx(-312 * 0.32 - b["fee"])
    # c is below the threshold, d has no more popular team.
    assert rule.loc["c", "status"] == "no_signal" and rule.loc["d", "status"] == "no_signal"
    assert (rule["threshold"] == 0.5).all()
    res = out["results"].set_index("strategy")
    assert res.loc["opposite_rule", "n_trades"] == 2 and res.loc["opposite_rule", "wins"] == 1
    assert res.loc["opposite_rule", "threshold"] == 0.5


def test_benchmark_trades_every_game_with_a_more_popular_team():
    out = rep.opposite_rule(tiny_games(), threshold=0.5, n_boot=20, seed=1)
    bench = out["trades"][out["trades"]["strategy"] == "opposite_benchmark"].set_index("game_id")
    assert bench["status"].to_dict() == {"a": "trade", "b": "trade", "c": "trade",
                                         "d": "no_trade_gap_zero"}
    assert bench.loc["c", "side"] == "home"                 # gap 0.2: home is more popular
    res = out["results"].set_index("strategy").loc["opposite_benchmark"]
    assert res["n_trades"] == 3 and res["no_trade_gap_zero"] == 1
    trades = bench[bench["status"] == "trade"]
    assert res["profit_before_fees"] == pytest.approx(trades["payout"].sum() - trades["cost"].sum())
    assert res["return_before_fees"] == pytest.approx(res["profit_before_fees"] / res["total_cost"])


def test_the_stage_9_rule_is_unchanged_by_the_new_options():
    games = tiny_games()
    less = st.make_trades(games, rep.GAP_COL, use_threshold=False).set_index("game_id")
    more = st.make_trades(games, rep.GAP_COL, use_threshold=False, buy="more_popular",
                          threshold=0.5).set_index("game_id")
    assert less["side"].tolist()[:3] == ["away", "home", "away"]
    assert more["side"].tolist()[:3] == ["home", "away", "home"]
    with pytest.raises(ValueError):
        st.make_trades(games, rep.GAP_COL, use_threshold=True, buy="both")


def test_tests_table_has_the_three_primary_tests():
    sn, volume = new_snapshots()
    d, _ = rep.build_sample(sn, volume)
    import popularity
    cal = rep.calibration_tables(d, n_boot=30, seed=1)
    pop = popularity.analyze_measure(d, rep.MEASURE, n_boot=30, seed=1)
    rule = rep.opposite_rule(d, threshold=0.4, n_boot=30, seed=1)
    tests = rep.tests_table(cal, pop, rule).set_index("test_id")
    assert sorted(tests[tests["role"] == "primary"].index) == [
        "replication_lr", "replication_opposite_return", "replication_popularity_c"]
    assert tests.loc["replication_lr", "statistic_name"] == "LR"
    assert tests.loc["replication_lr", "p_raw"] == cal["lr"].iloc[0]["p_value"]
    c = pop["logistic"].set_index("term").loc["gap_c"]
    assert tests.loc["replication_popularity_c", "p_raw"] == c["p_value"]
    ret = tests.loc["replication_opposite_return"]
    assert ret["unit"] == "trades" and ret["p_method"] == "block bootstrap"
    assert ret["n"] == rule["results"].set_index("strategy").loc["opposite_rule", "n_trades"]
    assert tests.loc["replication_opposite_benchmark_return", "role"] == "secondary"
    assert len(tests) == 9

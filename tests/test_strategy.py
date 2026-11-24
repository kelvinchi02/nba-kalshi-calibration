"""Stage 9.3 machinery on small hand-made data (no real results)."""
import numpy as np
import pandas as pd
import pytest

import strategy as st
from eda import team_game_volume


# ---------------------------------------------------------------- fee and order size

@pytest.mark.parametrize("contracts,ask,expected", [
    (100, 0.01, 0.07),    # 0.0693 rounds up
    (100, 0.05, 0.34),    # 0.3325 rounds up
    (100, 0.10, 0.63),    # exactly 63 cents: must not become 0.64
    (100, 0.20, 1.12),
    (100, 0.25, 1.32),    # 1.3125 rounds up
    (100, 0.50, 1.75),
    (1, 0.50, 0.02),      # 0.0175 rounds up
    (1, 0.01, 0.01),
])
def test_taker_fee_matches_the_fee_schedule_table(contracts, ask, expected):
    fee, unrounded = st.taker_fee(contracts, ask)
    assert fee == pytest.approx(expected, abs=1e-12)
    assert unrounded == pytest.approx(0.07 * contracts * ask * (1 - ask))
    assert 0 <= fee - unrounded < 0.01


def test_fee_is_symmetric_in_price_and_charged_once_per_order():
    assert st.taker_fee(250, 0.3) == st.taker_fee(250, 0.7)
    whole, _ = st.taker_fee(333, 0.3)                # 4.8951 -> 4.90
    assert whole == pytest.approx(4.90)
    assert whole < 333 * st.taker_fee(1, 0.3)[0]     # not 333 one-contract fees of 2 cents


@pytest.mark.parametrize("ask,expected", [(0.25, 400), (0.29, 344), (0.05, 2000), (0.1, 1000),
                                          (0.99, 101), (0.07, 1428), (0.3, 333), (0.515, 194)])
def test_contracts_for_stake_is_an_exact_floor(ask, expected):
    c = st.contracts_for_stake(ask)
    assert c == expected and c * ask <= 100 + 1e-9 < (c + 1) * ask + 1e-9


def test_price_with_too_many_decimals_is_rejected():
    with pytest.raises(ValueError, match="four decimals"):
        st.price_units(0.123456)


# ---------------------------------------------------------------- expanding mean

def test_expanding_prior_mean_uses_only_earlier_games():
    tip = pd.to_datetime(["2025-10-21", "2025-10-23", "2025-10-25"], utc=True)
    at0 = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "tipoff_utc": tip,
                        "home_team": ["A", "A", "B"], "away_team": ["B", "C", "A"],
                        "home_pregame_contracts": [10.0, 30.0, 7.0],
                        "away_pregame_contracts": [4.0, 9.0, 80.0]})
    out = st.expanding_prior_mean(team_game_volume(at0)).set_index(["team", "game_id"])
    assert out.loc[("A", "g1"), "prior_games"] == 0 and np.isnan(out.loc[("A", "g1"), "prior_mean"])
    assert out.loc[("A", "g2"), "prior_mean"] == 10              # only g1
    assert out.loc[("A", "g3"), "prior_mean"] == 20              # g1 and g2, not g3's own 80
    assert out.loc[("B", "g3"), "prior_mean"] == 4 and out.loc[("B", "g3"), "prior_games"] == 1
    # a later game's volume cannot change an earlier game's value
    at0.loc[2, "away_pregame_contracts"] = 1e9
    again = st.expanding_prior_mean(team_game_volume(at0)).set_index(["team", "game_id"])
    assert again.loc[("A", "g2"), "prior_mean"] == 10


def test_two_games_at_the_same_tipoff_for_one_team_is_an_error():
    tip = pd.to_datetime(["2025-10-21", "2025-10-21"], utc=True)
    at0 = pd.DataFrame({"game_id": ["g1", "g2"], "tipoff_utc": tip, "home_team": ["A", "A"],
                        "away_team": ["B", "C"], "home_pregame_contracts": [1.0, 2.0],
                        "away_pregame_contracts": [1.0, 2.0]})
    with pytest.raises(ValueError, match="same tipoff"):
        st.expanding_prior_mean(team_game_volume(at0))


# ---------------------------------------------------------------- a small season

def season(n_rounds: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six teams, three games per round, one round per day. Team i trades about 100 * 2^i."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    for r in range(n_rounds):
        order = rng.permutation(6)
        for k in range(3):
            home, away = int(order[2 * k]), int(order[2 * k + 1])
            ask_home = round(float(rng.uniform(0.2, 0.8)), 2)
            rows.append({
                "game_id": f"{r:03d}{k}", "game_date": (pd.Timestamp("2025-10-21")
                                                        + pd.Timedelta(days=r)).date(),
                "tipoff_utc": pd.Timestamp("2025-10-21 23:00", tz="UTC")
                + pd.Timedelta(days=r, minutes=30 * k),
                "home_team": teams[home], "away_team": teams[away],
                "home_win": int(rng.uniform() < ask_home),
                "home_ask": ask_home, "away_ask": round(1.02 - ask_home, 2),
                "home_pregame_contracts": 100.0 * 2 ** home * rng.uniform(0.8, 1.2),
                "away_pregame_contracts": 100.0 * 2 ** away * rng.uniform(0.8, 1.2)})
    metro = pd.DataFrame({"team": teams, "population": [1e6 * (6 - i) for i in range(6)]})
    return pd.DataFrame(rows), metro


def test_eligibility_and_burn_in_counts():
    at0, metro = season()
    games = st.build_games(at0, metro)
    assert games["tipoff_utc"].is_monotonic_increasing
    assert not games.loc[games["home_prior_games"] < 10, "eligible"].any()
    assert not games.loc[games["away_prior_games"] < 10, "eligible"].any()
    assert games["eligible"].sum() == 90                     # every game from round 10 on
    assert games["tradable"].sum() == 40                     # the first 50 eligible are held out
    first_tradable = games.loc[games["tradable"], "eligible_order"].min()
    assert first_tradable == 51
    counts = st.sample_counts(games).set_index("group")["games"]
    assert counts["main sample games"] == 120 and counts["tradable games"] == 40


def test_threshold_is_the_quantile_of_earlier_eligible_games_only():
    at0, metro = season()
    games = st.build_games(at0, metro)
    thr = st.expanding_threshold(games, "gap_volume")
    assert thr[~games["eligible"]].isna().all()
    i = games.index[games["tradable"]][7]
    earlier = games[games["eligible"] & (games["tipoff_utc"] < games.loc[i, "tipoff_utc"])]
    assert thr[i] == pytest.approx(np.quantile(earlier["gap_volume"].abs(), 0.75))
    # changing this game or any later game leaves its threshold unchanged
    later = games.copy()
    later.loc[later["tipoff_utc"] >= games.loc[i, "tipoff_utc"], "gap_volume"] = 99.0
    assert st.expanding_threshold(later, "gap_volume")[i] == thr[i]


def test_rule_buys_the_less_popular_team_only_above_the_threshold():
    at0, metro = season()
    games = st.build_games(at0, metro)
    table = st.make_trades(games, "gap_volume", use_threshold=True)
    assert len(table) == games["tradable"].sum()
    trades = table[table["status"] == "trade"]
    assert 0 < len(trades) < len(table)
    assert (trades["gap"].abs() >= trades["threshold"]).all()
    assert (table.loc[table["status"] == "no_signal", "gap"].abs()
            < table.loc[table["status"] == "no_signal", "threshold"]).all()
    for _, t in trades.iterrows():
        g = games[games["game_id"] == t["game_id"]].iloc[0]
        less_popular = "home" if g["home_prior_mean"] < g["away_prior_mean"] else "away"
        assert t["side"] == less_popular and t["team"] == g[f"{less_popular}_team"]
        assert t["ask"] == g[f"{less_popular}_ask"]
        assert t["contracts"] == int(100 // t["ask"]) or t["contracts"] == st.contracts_for_stake(t["ask"])
        assert t["cost"] == pytest.approx(t["contracts"] * t["ask"]) and t["cost"] <= 100
        assert t["won"] == (g["home_win"] if less_popular == "home" else 1 - g["home_win"])
        assert t["profit"] == pytest.approx(t["contracts"] * t["won"] - t["cost"] - t["fee"])
    assert (table.loc[table["status"] != "trade", ["cost", "fee", "profit"]] == 0).all().all()


def test_benchmark_trades_every_tradable_game():
    at0, metro = season()
    games = st.build_games(at0, metro)
    bench = st.make_trades(games, "gap_volume", use_threshold=False)
    assert (bench["status"] == "trade").all() and len(bench) == games["tradable"].sum()
    rule = st.make_trades(games, "gap_volume", use_threshold=True)
    merged = rule[rule["status"] == "trade"].merge(bench, on="game_id", suffixes=("", "_b"))
    assert (merged["profit"] == merged["profit_b"]).all()    # same order where both trade


def test_bad_ask_and_zero_gap_give_no_trade():
    at0, metro = season()
    games = st.build_games(at0, metro)
    tradable = games.index[games["tradable"]]
    games.loc[tradable[0], ["home_ask", "away_ask"]] = np.nan
    games.loc[tradable[1], ["home_ask", "away_ask"]] = 1.0
    games.loc[tradable[2], "gap_volume"] = 0.0
    bench = st.make_trades(games, "gap_volume", use_threshold=False).set_index("game_id")
    ids = games.loc[tradable[:3], "game_id"].tolist()
    assert bench.loc[ids, "status"].tolist() == ["no_trade_bad_ask", "no_trade_bad_ask",
                                                 "no_trade_gap_zero"]
    summary = st.summarize(bench.reset_index())
    assert summary["no_trade_bad_ask"] == 2 and summary["no_trade_gap_zero"] == 1
    assert summary["n_trades"] == len(bench) - 3


def test_metro_rule_never_trades_a_same_metro_game():
    at0, metro = season()
    metro.loc[metro["team"] == "T1", "population"] = metro.loc[metro["team"] == "T0",
                                                               "population"].iloc[0]
    games = st.build_games(at0, metro)
    table = st.make_trades(games, "gap_metro", use_threshold=True)
    same = table[(table["gap"] == 0)]
    assert len(same) > 0 and (same["status"] != "trade").all()
    trades = table[table["status"] == "trade"]
    pop = metro.set_index("team")["population"]
    for _, t in trades.iterrows():
        other = t["away_team"] if t["side"] == "home" else t["home_team"]
        assert pop[t["team"]] < pop[other]                   # the smaller metro is bought


def test_summary_by_hand():
    table = pd.DataFrame({
        "status": ["trade", "trade", "no_signal"], "signal": [1, 1, 0],
        "ask": [0.25, 0.50, np.nan], "contracts": [400, 200, 0], "cost": [100.0, 100.0, 0.0],
        "fee": [5.25, 3.50, 0.0], "fee_unrounded": [5.25, 3.50, 0.0],
        "won": pd.array([1, 0, pd.NA], dtype="Int64"), "payout": [400.0, 0.0, 0.0],
        "profit": [294.75, -103.50, 0.0]})
    s = st.summarize(table)
    assert (s["tradable_games"], s["signals"], s["n_trades"], s["wins"]) == (3, 2, 2, 1)
    assert s["win_rate"] == 0.5 and s["mean_ask"] == 0.375 and s["contracts"] == 600
    assert s["total_cost"] == 200 and s["total_fees"] == 8.75
    assert s["profit"] == pytest.approx(191.25)
    assert s["return_per_dollar"] == pytest.approx(191.25 / 208.75)


def test_bootstrap_resamples_dates_of_tradable_games_and_keeps_trades_fixed():
    at0, metro = season()
    games = st.build_games(at0, metro)
    table = st.make_trades(games, "gap_volume", use_threshold=True)
    a, b = st.bootstrap_trades(table, n_boot=60, seed=9), st.bootstrap_trades(table, n_boot=60, seed=9)
    pd.testing.assert_frame_equal(a, b)
    assert a["profit"].nunique() > 1
    row = st.results_row("volume_rule", table, a)
    assert row["strategy"] == "volume_rule" and row["profit"] == st.summarize(table)["profit"]
    assert row["profit_boot_lo"] <= row["profit_boot_hi"]
    assert row["return_boot_lo"] <= row["return_boot_hi"]


def test_run_strategy_labels_the_trades_and_matches_the_parts():
    at0, metro = season()
    games = st.build_games(at0, metro)
    table, row = st.run_strategy(games, "volume_rule", "gap_volume", use_threshold=True,
                                 n_boot=40, seed=3)
    assert (table["strategy"] == "volume_rule").all()
    plain = st.make_trades(games, "gap_volume", use_threshold=True)
    assert row["profit"] == st.summarize(plain)["profit"] and row["n_trades"] == st.summarize(plain)["n_trades"]
    boot = st.bootstrap_trades(plain, n_boot=40, seed=3)
    assert row["profit_boot_lo"] == boot["profit"].quantile(0.025)


# ---------------------------------------------------------------- exploratory (plan E1, E2)

def test_expanding_gap_sample_is_the_eligible_games_with_their_past_only_gap():
    at0, metro = season()
    at0["p_home"] = at0["home_ask"] - 0.01
    d = st.expanding_gap_sample(at0, metro)
    games = st.build_games(at0, metro)
    eligible = games[games["eligible"]].set_index("game_id")
    assert len(d) == len(eligible) == 90 and d["game_id"].is_unique
    row = d.iloc[17]
    assert row["gap_volume_expanding"] == eligible.loc[row["game_id"], "gap_volume"]
    assert row["p_home"] == at0.set_index("game_id").loc[row["game_id"], "p_home"]
    # a later game's volume does not change an earlier game's gap
    last = at0["tipoff_utc"].idxmax()
    changed = at0.copy()
    changed.loc[last, ["home_pregame_contracts", "away_pregame_contracts"]] = 1e9
    again = st.expanding_gap_sample(changed, metro)
    assert again["gap_volume_expanding"].tolist() == d["gap_volume_expanding"].tolist()


def test_expanding_gap_sample_feeds_the_9_1_model():
    import popularity as pop
    at0, metro = season(n_rounds=80, seed=2)
    at0["p_home"] = at0["home_ask"] - 0.01
    d = st.expanding_gap_sample(at0, metro)
    out = pop.analyze_measure(d, "volume_expanding", n_boot=20, seed=1)
    assert out["logistic"]["term"].tolist() == ["intercept_a", "slope_b", "gap_c"]
    assert (out["logistic"]["measure"] == "volume_expanding").all()
    assert (out["logistic"]["n"] == len(d)).all()


def test_before_fees_table_by_hand():
    results = pd.DataFrame([{"strategy": "x", "n_trades": 2, "total_cost": 200.0,
                             "total_fees": 8.75, "profit": 191.25,
                             "return_per_dollar": 191.25 / 208.75, "wins": 1}])
    out = st.before_fees_table(results).iloc[0]
    assert out["profit_before_fees"] == pytest.approx(200.0)     # payout 400 - cost 200
    assert out["return_before_fees"] == pytest.approx(1.0)
    assert out["profit"] == 191.25 and "wins" not in out.index

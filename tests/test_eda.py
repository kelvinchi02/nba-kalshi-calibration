"""Checks of the EDA tables and of the calibration helpers the notebooks call."""
import numpy as np
import pandas as pd
import pytest

import eda
from calibration import (add_bootstrap_intervals, assign_bins, bootstrap_summary,
                         calibration_bins, point_estimates)
from stats_tools import brier, fit_logit, lr_test, wilson


def snapshots() -> pd.DataFrame:
    """Three games, two horizons each, with round numbers."""
    rows = []
    games = [("g1", "AAA", "BBB", 100.0, 10.0, 0.60), ("g2", "BBB", "CCC", 30.0, 50.0, 0.20),
             ("g3", "AAA", "CCC", 300.0, 70.0, 0.90)]
    for game_id, home, away, home_vol, away_vol, p0 in games:
        for h, p in [(24, p0 - 0.1), (0, p0)]:
            rows.append({"game_id": game_id, "event_ticker": "EV-" + game_id, "horizon_h": h,
                         "home_team": home, "away_team": away, "p_home": p,
                         "home_pregame_contracts": home_vol,
                         "away_pregame_contracts": away_vol})
    return pd.DataFrame(rows)


def test_team_pregame_volume_uses_each_teams_own_market_once_per_game():
    tv = eda.team_pregame_volume(snapshots()).set_index("team")
    assert tv.loc["AAA", "games"] == 2 and tv.loc["AAA", "mean_pregame_contracts"] == 200
    assert tv.loc["BBB", "mean_pregame_contracts"] == 20     # 10 away, 30 home
    assert tv.loc["CCC", "mean_pregame_contracts"] == 60     # 50 and 70, both away
    assert tv["rank"].tolist() == [1, 2, 3] and tv.index.tolist() == ["AAA", "CCC", "BBB"]


def test_pick_example_takes_largest_move_among_high_volume_games():
    sn = snapshots()
    sn.loc[(sn["game_id"] == "g2") & (sn["horizon_h"] == 24), "p_home"] = 0.9  # biggest move
    # total volume: g1 110, g2 80, g3 370. Only g3 is above the median of 110.
    assert eda.pick_example(sn) == "EV-g3"


def test_tipoff_prices_keeps_horizon_0_and_drops_missing():
    sn = snapshots()
    sn.loc[(sn["game_id"] == "g2") & (sn["horizon_h"] == 0), "p_home"] = np.nan
    assert eda.tipoff_prices(sn).tolist() == [0.60, 0.90]


def test_price_summary_values():
    p = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])
    out = eda.price_summary(p).iloc[0]
    assert out["n"] == 5 and out["mean"] == pytest.approx(0.5)
    assert out["median"] == pytest.approx(0.5)
    assert out["min"] == 0.1 and out["max"] == 0.9
    assert out["p25"] == pytest.approx(0.3) and out["p75"] == pytest.approx(0.7)
    assert out["share_above_half"] == pytest.approx(0.4)     # strictly above 0.5


def test_price_histogram_bins():
    p = pd.Series([0.01, 0.049, 0.05, 0.52, 0.975, 1.0])
    hist = eda.price_histogram(p)
    assert len(hist) == 20 and hist["games"].sum() == len(p)
    assert hist["bin_lo"].iloc[0] == 0 and hist["bin_hi"].iloc[-1] == 1
    games = hist.set_index("bin_lo")["games"]
    assert games[0.0] == 2 and games[0.05] == 1              # 0.05 opens the next bin
    assert games[0.5] == 1 and games[0.95] == 2              # the last bin includes 1
    assert hist["share"].sum() == pytest.approx(1)


def test_missing_raw_cache_gives_a_clear_error():
    with pytest.raises(FileNotFoundError, match="No raw cache for NOT-A-TICKER"):
        eda.load_market_or_fail("NOT-A-TICKER")


def test_minute_quotes_carries_forward_and_blanks_one_sided_books():
    candles = pd.DataFrame({"end_ts": [60, 180], "bid": [0.40, 0.0], "ask": [0.44, 0.50]})
    q = eda.minute_quotes(candles, start=0, end=240)
    assert q.index.tolist() == [60, 120, 180, 240]
    assert q["mid"].iloc[0] == pytest.approx(0.42)
    assert q["mid"].iloc[1] == pytest.approx(0.42)           # minute 120 has no candle
    assert q["mid"].iloc[2:].isna().all()                    # bid 0: no midpoint
    assert q["hours_before"].iloc[-1] == 0


def calibration_sample(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    d = pd.DataFrame({"game_id": [f"{i:04d}" for i in range(n)],
                      "p_home": rng.uniform(0.1, 0.9, n),
                      "game_date": np.repeat(pd.date_range("2025-10-21", periods=40).date, 10)})
    d["home_win"] = (rng.uniform(size=n) < d["p_home"]).astype(int)
    d = d.sort_values(["p_home", "game_id"]).reset_index(drop=True)
    d["bin"] = assign_bins(n)
    return d


def test_calibration_bins_table():
    d = calibration_sample()
    bins = calibration_bins(d)
    assert bins["bin"].tolist() == list(range(1, 11)) and (bins["n"] == 40).all()
    assert bins["home_wins"].sum() == d["home_win"].sum()
    assert (bins["p_max"].iloc[:-1].to_numpy() <= bins["p_min"].iloc[1:].to_numpy()).all()
    first = bins.iloc[0]
    assert first["win_rate"] == first["home_wins"] / first["n"]
    assert (first["wilson_lo"], first["wilson_hi"]) == wilson(int(first["home_wins"]), 40)


def test_point_estimates_and_bootstrap_intervals_line_up_by_bin():
    d = calibration_sample()
    bins = calibration_bins(d)
    bs = brier(d["p_home"], d["home_win"])
    fit = fit_logit(d["p_home"], d["home_win"])
    lr = lr_test(d["p_home"], d["home_win"], fit)
    point = point_estimates(bs, fit, lr, bins)
    assert list(point)[:7] == ["brier", "brier_baseline", "brier_diff", "brier_skill",
                               "intercept_a", "slope_b", "lr_stat"]
    assert point["bin3_win_rate"] == bins["win_rate"].iloc[2]
    assert point["slope_b"] == fit.params[1] and point["lr_stat"] == lr["lr_stat"]

    boot = pd.DataFrame({k: np.linspace(v - 1, v + 1, 201) for k, v in point.items()})
    summary = bootstrap_summary(boot, point, n_dates=40)
    out = add_bootstrap_intervals(bins, summary)
    assert "boot_lo" not in bins.columns                     # the input table is not changed
    assert np.allclose(out["boot_lo"], bins["win_rate"] - 0.95)
    assert np.allclose(out["boot_hi"], bins["win_rate"] + 0.95)

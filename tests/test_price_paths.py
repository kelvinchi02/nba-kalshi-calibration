"""Price path builder on small hand-made markets (no real data)."""
import gzip

import numpy as np
import pandas as pd
import pytest

import build_price_paths as bp
from build_snapshots import quote_before
from csv_io import read_csv, write_csv

TIP = 1_000_000_000.0          # tipoff, unix seconds


def candles(rows) -> pd.DataFrame:
    """rows: (minutes before tipoff at which the candle ends, bid, ask)."""
    return pd.DataFrame([{"end_ts": int(TIP - m * 60), "bid": b, "ask": a} for m, b, a in rows],
                        columns=["end_ts", "bid", "ask"]).sort_values("end_ts", ignore_index=True)


def trades(rows) -> pd.DataFrame:
    """rows: (trade_id, seconds before tipoff, contracts)."""
    return pd.DataFrame([{"trade_id": i, "ts": TIP - s, "price": 0.5, "count": float(c)}
                         for i, s, c in rows], columns=["trade_id", "ts", "price", "count"])


def test_grid_runs_from_30_hours_to_tipoff_in_5_minute_steps():
    m = bp.MINUTES_TO_TIPOFF
    assert len(m) == 361 and m[0] == 1800 and m[-1] == 0 and set(np.diff(m)) == {-5}


def test_quotes_at_matches_quote_before_on_random_candles():
    rng = np.random.default_rng(7)
    ends = np.sort(rng.choice(np.arange(60, 30 * 3600, 60), size=300, replace=False))
    bid = rng.integers(0, 60, 300) / 100          # some bids of 0: one-sided books
    ask = np.minimum(bid + rng.integers(1, 45, 300) / 100, 1.0)
    bid[5], ask[9] = np.nan, np.nan
    c = pd.DataFrame({"end_ts": (TIP - 30 * 3600 + ends).astype(int), "bid": bid, "ask": ask})
    times = TIP - bp.MINUTES_TO_TIPOFF * 60.0
    fast = bp.quotes_at(c, times)
    for i, s in enumerate(times):
        slow = quote_before(c, s)
        for key in ("bid", "ask", "mid", "spread", "quote_age_min"):
            a, b = fast.loc[i, key], slow[key]
            assert (np.isnan(a) and np.isnan(b)) or a == b, (i, key, a, b)
        assert bool(fast.loc[i, "one_sided"]) == slow["one_sided"]
        carried = bool(slow["quote_age_min"] > 1) if not np.isnan(slow["quote_age_min"]) else False
        assert bool(fast.loc[i, "carried"]) == carried


def test_quote_uses_only_candles_ending_strictly_before_the_step():
    c = candles([(10, 0.40, 0.42), (5, 0.50, 0.52), (0, 0.90, 0.92)])
    q = bp.quotes_at(c, np.array([TIP - 600, TIP - 300, TIP - 240, TIP]))
    assert np.isnan(q.loc[0, "mid"])                      # the candle ending at -10 is not before -10
    assert q.loc[1, "mid"] == pytest.approx(0.41)         # at -5: the candle ending at -10
    assert q.loc[2, "mid"] == pytest.approx(0.51)         # at -4: the candle ending at -5
    assert q.loc[3, "mid"] == pytest.approx(0.51)         # at tipoff: not the candle ending at tipoff
    assert q["carried"].tolist() == [False, True, False, True]
    assert q.loc[2, "quote_age_min"] == 1.0


def test_no_candles_gives_empty_quotes():
    q = bp.quotes_at(candles([]), np.array([TIP - 60, TIP]))
    assert q["mid"].isna().all() and not q["carried"].any() and not q["one_sided"].any()


def test_step_contracts_window_is_half_open():
    t = trades([("a", 600, 1), ("b", 599, 2), ("c", 301, 4), ("d", 300, 8), ("e", 1, 16)])
    out = bp.step_contracts(t, np.array([TIP - 300, TIP]), minutes=5)
    assert out.tolist() == [1 + 2 + 4, 8 + 16]            # [S - 5 min, S): a counts, d does not


def one_game(market_open_minutes: float = 3000, home_pregame=None, away_pregame=None):
    home = (trades([("h1", 1800 * 60 + 30, 5), ("h2", 100 * 60 + 10, 7), ("h3", 30, 3)]),
            candles([(1797, 0.60, 0.62), (100, 0.64, 0.66), (2, 0.0, 0.70)]))
    away = (trades([("a1", 100 * 60 + 20, 11)]),
            candles([(1790, 0.37, 0.41), (3, 0.30, 0.31)]))
    return bp.game_path("g1", TIP, TIP - market_open_minutes * 60, home, away,
                        home_pregame, away_pregame).set_index("minutes_to_tipoff")


def test_game_path_values_by_hand():
    p = one_game()
    assert list(p.columns) == bp.COLUMNS[:1] + bp.COLUMNS[2:]
    assert len(p) == 361 and p["market_open"].all()
    # 1800 and 1795 before the first home candle (ends at 1797): only 1795 has the home quote.
    assert np.isnan(p.loc[1800, "home_mid"]) and p.loc[1795, "home_mid"] == pytest.approx(0.61)
    assert np.isnan(p.loc[1795, "p_home_mid"])            # away has no quote yet
    assert p.loc[1795, "max_spread"] == 0.02              # the one spread there is
    # From 1785 on both sides have a quote: 0.61 and 0.39.
    row = p.loc[1785]
    assert row["p_home_mid"] == pytest.approx(0.61 / (0.61 + 0.39))
    assert row["max_spread"] == 0.04
    assert row["home_mid_carried_forward"] and row["away_mid_carried_forward"]
    # Step 95 follows the candle ending at 100: home 0.65.
    assert p.loc[95, "home_mid"] == pytest.approx(0.65)
    # At tipoff the home book is one-sided (bid 0): no midpoint, the spread stays.
    assert np.isnan(p.loc[0, "home_mid"]) and np.isnan(p.loc[0, "p_home_mid"])
    assert p.loc[0, "away_mid"] == pytest.approx(0.305) and p.loc[0, "max_spread"] == 0.70
    # Trades: h2 and a1 fall in [100, 105) minutes before tipoff, so in the step at 100.
    assert p.loc[100, "contracts"] == 18 and p.loc[95, "contracts"] == 0
    assert p.loc[0, "contracts"] == 3
    assert p.loc[1800, "contracts"] == 5                  # 30 seconds before the window start


def test_steps_before_the_market_opened_are_empty():
    p = one_game(market_open_minutes=1790)                # opened 1790 minutes before tipoff
    closed = p.loc[[1800, 1795, 1790]]
    assert not closed["market_open"].any()
    for col in ("home_mid", "away_mid", "p_home_mid", "max_spread", "contracts",
                "home_mid_carried_forward", "away_mid_carried_forward"):
        assert closed[col].isna().all(), col
    assert p.loc[1785, "market_open"] and p.loc[1785, "contracts"] == 0
    assert p.loc[1785, "home_mid"] == pytest.approx(0.61)


def test_pregame_trades_are_added_once():
    pre = trades([("h1", 1800 * 60 + 30, 5), ("p1", 1800 * 60 + 200, 2)])   # h1 is in both pulls
    p = one_game(home_pregame=pre)
    assert p.loc[1800, "contracts"] == 5 + 2


def test_gzipped_csv_round_trip_is_byte_reproducible(tmp_path):
    p = one_game(market_open_minutes=1790).reset_index()[bp.COLUMNS]
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    write_csv(p, a)
    write_csv(p, b)
    assert a.read_bytes() == b.read_bytes()
    text = gzip.decompress(a.read_bytes()).decode("utf-8")
    assert text.splitlines()[0] == ",".join(bp.COLUMNS)
    assert text.splitlines()[1] == "g1,1800,0,,,,,,,"
    back = read_csv(a)
    assert back["game_id"].tolist() == p["game_id"].tolist()
    assert back["market_open"].dtype == bool
    assert back["p_home_mid"].iloc[3] == p["p_home_mid"].iloc[3]
    assert back["home_mid_carried_forward"].isna().sum() == 3
    assert bool(back["home_mid_carried_forward"].iloc[3]) is True

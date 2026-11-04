"""Tests for snapshot time-window logic (no leakage), VWAP and normalization."""
import math

import numpy as np
import pandas as pd
import pytest

from build_snapshots import (choose_price, last_trade, market_snapshot, normalize,
                             quote_before, vwap, window_trades)

S = 1_000_000.0  # snapshot time, unix seconds


def trades(*rows):
    """rows of (ts, price, count)."""
    return pd.DataFrame(rows, columns=["ts", "price", "count"]).sort_values("ts",
                                                                            ignore_index=True)


def candles(*rows):
    """rows of (end_ts, bid, ask)."""
    return pd.DataFrame(rows, columns=["end_ts", "bid", "ask"])


# ---------------------------------------------------------------- no leakage

def test_last_trade_excludes_trades_at_and_after_snapshot():
    t = trades((S - 120, 0.40, 1), (S, 0.90, 5), (S + 1, 0.95, 5))
    price, age = last_trade(t, S)
    assert price == 0.40
    assert age == pytest.approx(2.0)


def test_last_trade_missing_when_nothing_before():
    t = trades((S, 0.5, 1), (S + 60, 0.6, 1))
    price, age = last_trade(t, S)
    assert math.isnan(price) and math.isnan(age)


def test_window_is_half_open():
    t = trades((S - 600, 0.1, 1), (S - 601, 0.2, 1), (S - 1, 0.3, 1), (S, 0.4, 1))
    w = window_trades(t, S, 10)
    assert sorted(w["price"]) == [0.1, 0.3]  # includes S-10min, excludes S


def test_quote_ignores_candle_ending_at_or_after_snapshot():
    c = candles((S - 120, 0.40, 0.42), (S - 60, 0.44, 0.46), (S, 0.90, 0.92), (S + 60, 0.9, 0.95))
    q = quote_before(c, S)
    assert (q["bid"], q["ask"]) == (0.44, 0.46)
    assert q["mid"] == pytest.approx(0.45)
    assert q["quote_age_min"] == pytest.approx(1.0)


def test_spread_from_same_candle_as_midpoint():
    c = candles((S - 120, 0.40, 0.48), (S - 60, 0.44, 0.49), (S, 0.10, 0.90))
    q = quote_before(c, S)
    assert q["spread"] == 0.05  # exact after rounding, so a "> 0.05" filter excludes it
    assert quote_before(candles((S - 60, 0.30, 0.31)), S)["spread"] == 0.01


def test_spread_missing_when_no_candle_before_snapshot():
    assert math.isnan(quote_before(candles((S, 0.4, 0.5)), S)["spread"])


def test_quote_carried_forward_is_flagged():
    t = trades((S - 3000, 0.5, 1))
    c = candles((S - 1800, 0.48, 0.50))
    snap = market_snapshot(t, c, S, window_start=S - 30 * 3600)
    assert snap["mid"] == pytest.approx(0.49)
    assert snap["quote_age_min"] == pytest.approx(30.0)
    assert snap["mid_carried_forward"] is True


def test_quote_missing_when_no_candle_before_snapshot():
    q = quote_before(candles((S, 0.4, 0.5)), S)
    assert math.isnan(q["mid"]) and math.isnan(q["quote_age_min"])


@pytest.mark.parametrize("bid,ask", [(0.0, 0.40), (0.60, 1.0), (np.nan, 0.5)])
def test_one_sided_book_gives_no_midpoint(bid, ask):
    q = quote_before(candles((S - 60, bid, ask)), S)
    assert q["one_sided"] is True
    assert math.isnan(q["mid"])


def test_snapshot_ignores_everything_after_tipoff():
    """Data after S must not change any snapshot value."""
    base_t = trades((S - 300, 0.50, 10), (S - 30, 0.52, 5))
    base_c = candles((S - 120, 0.49, 0.51), (S - 60, 0.51, 0.53))
    later_t = pd.concat([base_t, trades((S, 0.99, 1000), (S + 30, 0.01, 1000))],
                        ignore_index=True)
    later_c = pd.concat([base_c, candles((S, 0.98, 0.99), (S + 60, 0.01, 0.02))],
                        ignore_index=True)
    a = market_snapshot(base_t, base_c, S, window_start=S - 30 * 3600)
    b = market_snapshot(later_t, later_c, S, window_start=S - 30 * 3600)
    assert a == b


def test_cum_contracts_counts_window_start_to_snapshot():
    t = trades((S - 7200, 0.5, 100), (S - 3000, 0.5, 7), (S - 10, 0.5, 3), (S, 0.5, 50))
    snap = market_snapshot(t, candles(), S, window_start=S - 3600)
    assert snap["cum_contracts"] == 10  # excludes before window start and at S


# ---------------------------------------------------------------- VWAP

def test_vwap_weights_by_contracts():
    t = trades((S - 500, 0.40, 1), (S - 100, 0.60, 3))
    assert vwap(t, S) == pytest.approx((0.40 * 1 + 0.60 * 3) / 4)


def test_vwap_uses_only_last_ten_minutes():
    t = trades((S - 601, 0.10, 100), (S - 300, 0.50, 2), (S, 0.90, 100))
    assert vwap(t, S) == pytest.approx(0.50)


def test_vwap_missing_when_no_trades_in_window():
    assert math.isnan(vwap(trades((S - 1000, 0.5, 1)), S))


# ---------------------------------------------------------------- normalization

def test_normalize_sums_to_one():
    p, total = normalize(0.62, 0.40)
    assert total == pytest.approx(1.02)
    assert p == pytest.approx(0.62 / 1.02)
    assert p + 0.40 / 1.02 == pytest.approx(1.0)


def test_normalize_missing_side_gives_missing():
    for h, a in [(np.nan, 0.4), (0.6, np.nan)]:
        p, total = normalize(h, a)
        assert math.isnan(p) and math.isnan(total)


def test_normalize_zero_sum():
    p, total = normalize(0.0, 0.0)
    assert math.isnan(p) and total == 0


def test_choose_price_prefers_mid_then_last_then_vwap():
    row = {"p_home_mid": np.nan, "p_home_last": 0.55, "p_home_vwap_10m": 0.60}
    assert choose_price(row) == (0.55, "last")
    row["p_home_mid"] = 0.52
    assert choose_price(row) == (0.52, "mid")
    assert choose_price({"p_home_mid": np.nan, "p_home_last": np.nan,
                         "p_home_vwap_10m": np.nan})[1] is None

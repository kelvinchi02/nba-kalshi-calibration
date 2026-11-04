"""Stage 6: build pre-tipoff price snapshots for every matched game.

For each game and horizon h in (24, 6, 1, 0) hours, the snapshot time is
S = official NBA tipoff - h (tipoff_utc from game_map.parquet). For each team market:

  mid        bid/ask midpoint from the last 1-minute candle that ENDS strictly before S
             (main price). Missing candles mean the quote did not change (checked in
             Stage 2), so the last earlier candle is carried forward and its age is
             recorded. Carried quotes are flagged (quote_age_min > 1).
             A one-sided book (bid 0 or ask 1.00) gives no midpoint (flagged one_sided).
  spread     ask - bid of that same candle, in dollars (0.01 = 1 cent). Kept even for
             one-sided books. max_spread (row level) = wider of the two sides, for filtering.
  last       price of the last trade strictly before S (backup 1)
  vwap_10m   volume-weighted price of trades in [S - 10 min, S) (backup 2)
  n_trades_10m, contracts_10m   activity in [S - 10 min, S)
  cum_contracts                 contracts traded from the start of the pulled 30h window
                                (tipoff - 30h) up to S. Not since market open.
  pregame_contracts             contracts traded from market open to the official tipoff,
                                strictly before tipoff (same value at every horizon).
                                Window and pregame pulls are combined and de-duplicated by
                                trade_id.

Only data strictly before S is used. Values stay missing when no data exists; nothing is
filled forward without a flag.

Two-sided normalization, for each price type: p_home = home / (home + away), using the
raw pair sum (kept as sum_<type>). p_home is the main estimate: the midpoint pair if both
sides have one, else last trade, else 10-min VWAP. price_source says which. Both sides
always come from the same source.

Output: data/processed/snapshots.parquet, one row per game per horizon, with home_win.

Usage:
    python src/build_snapshots.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_MAP_PATH = REPO_ROOT / "data" / "processed" / "game_map.parquet"
CACHE_DIR = REPO_ROOT / "data" / "raw" / "kalshi" / "markets"
PREGAME_DIR = REPO_ROOT / "data" / "raw" / "kalshi" / "pregame"
OUT_PATH = REPO_ROOT / "data" / "processed" / "snapshots.parquet"

HORIZONS_H = (24, 6, 1, 0)
VWAP_MINUTES = 10
CANDLE_SECONDS = 60


# ---------------------------------------------------------------- loading

def trades_frame(raw_trades: list[dict]) -> pd.DataFrame:
    """trade_id, ts (unix seconds, float), price (YES price, dollars), count (contracts)."""
    created = pd.to_datetime([t["created_time"] for t in raw_trades], utc=True, format="ISO8601")
    return pd.DataFrame({
        "trade_id": [t["trade_id"] for t in raw_trades],
        "ts": (created - pd.Timestamp(0, tz="UTC")).total_seconds(),
        "price": [float(t["yes_price_dollars"]) for t in raw_trades],
        "count": [float(t["count_fp"]) for t in raw_trades],
    }, columns=["trade_id", "ts", "price", "count"]).sort_values("ts", ignore_index=True)


def load_market(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (trades, candles) from the 30h window cache, or None if not cached.

    candles: end_ts (unix seconds, int), bid, ask (YES bid/ask close, dollars)
    """
    path = CACHE_DIR / f"{ticker}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rec = json.load(f)
    trades = trades_frame(rec["trades"])

    def close(c: dict, side: str) -> float:
        value = (c.get(side) or {}).get("close")
        return float(value) if value is not None else np.nan

    candles = pd.DataFrame({
        "end_ts": [int(c["end_period_ts"]) for c in rec["candlesticks"]],
        "bid": [close(c, "yes_bid") for c in rec["candlesticks"]],
        "ask": [close(c, "yes_ask") for c in rec["candlesticks"]],
    }, columns=["end_ts", "bid", "ask"]).sort_values("end_ts", ignore_index=True)
    return trades, candles


def load_pregame(ticker: str) -> pd.DataFrame | None:
    """Trades from market open to the window start, or None if not cached."""
    path = PREGAME_DIR / f"{ticker}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return trades_frame(json.load(f)["trades"])


def pregame_contracts(window: pd.DataFrame, pregame: pd.DataFrame, tipoff: float) -> float:
    """Contracts traded from market open to tipoff, strictly before tipoff.

    The two pulls can share trades at the window boundary, so de-duplicate by trade_id.
    """
    both = pd.concat([pregame, window], ignore_index=True).drop_duplicates("trade_id")
    return float(both.loc[both["ts"] < tipoff, "count"].sum())


# ---------------------------------------------------------------- per-market measures
# All functions take a snapshot time s (unix seconds) and use only data strictly before s.

def last_trade(trades: pd.DataFrame, s: float) -> tuple[float, float]:
    """(price, age in minutes) of the last trade with ts < s; NaN if none."""
    before = trades[trades["ts"] < s]
    if before.empty:
        return np.nan, np.nan
    row = before.iloc[-1]
    return float(row["price"]), (s - float(row["ts"])) / 60


def window_trades(trades: pd.DataFrame, s: float, minutes: float) -> pd.DataFrame:
    """Trades with s - minutes*60 <= ts < s."""
    return trades[(trades["ts"] >= s - minutes * 60) & (trades["ts"] < s)]


def vwap(trades: pd.DataFrame, s: float, minutes: float = VWAP_MINUTES) -> float:
    """Volume-weighted average price over [s - minutes, s); NaN if no contracts traded."""
    w = window_trades(trades, s, minutes)
    total = w["count"].sum()
    if total <= 0:
        return np.nan
    return float((w["price"] * w["count"]).sum() / total)


def quote_before(candles: pd.DataFrame, s: float) -> dict:
    """Bid/ask from the last candle ending strictly before s.

    A candle ending at s may contain data stamped at s itself, so it is not used.
    """
    before = candles[candles["end_ts"] < s]
    if before.empty:
        return {"bid": np.nan, "ask": np.nan, "mid": np.nan, "spread": np.nan,
                "quote_age_min": np.nan, "one_sided": False}
    row = before.iloc[-1]
    bid, ask = float(row["bid"]), float(row["ask"])
    # bid 0 means no bids, ask 1.00 means no asks: no meaningful midpoint.
    one_sided = bool(np.isnan(bid) or np.isnan(ask) or bid <= 0 or ask >= 1)
    return {
        "bid": bid,
        "ask": ask,
        "mid": np.nan if one_sided else (bid + ask) / 2,
        # Rounded to 4 dp so cent-grid prices compare exactly (0.05 is not 0.0500000001).
        "spread": round(ask - bid, 4) if not (np.isnan(bid) or np.isnan(ask)) else np.nan,
        "quote_age_min": (s - float(row["end_ts"])) / 60,
        "one_sided": one_sided,
    }


def market_snapshot(trades: pd.DataFrame, candles: pd.DataFrame, s: float,
                    window_start: float) -> dict:
    last_price, last_age = last_trade(trades, s)
    w = window_trades(trades, s, VWAP_MINUTES)
    q = quote_before(candles, s)
    cum = trades[(trades["ts"] >= window_start) & (trades["ts"] < s)]["count"].sum()
    return {
        "mid": q["mid"],
        "bid": q["bid"],
        "ask": q["ask"],
        "spread": q["spread"],
        "quote_age_min": q["quote_age_min"],
        "mid_carried_forward": bool(q["quote_age_min"] > CANDLE_SECONDS / 60)
        if not np.isnan(q["quote_age_min"]) else False,
        "one_sided": q["one_sided"],
        "last": last_price,
        "last_trade_age_min": last_age,
        "vwap_10m": vwap(trades, s),
        "n_trades_10m": int(len(w)),
        "contracts_10m": float(w["count"].sum()),
        "cum_contracts": float(cum),
    }


# ---------------------------------------------------------------- normalization

def normalize(home: float, away: float) -> tuple[float, float]:
    """(p_home, raw sum). NaN unless both sides are present and the sum is positive."""
    if np.isnan(home) or np.isnan(away):
        return np.nan, np.nan
    total = home + away
    if total <= 0:
        return np.nan, total
    return home / total, total


def choose_price(row: dict) -> tuple[float, str | None]:
    """Main estimate: midpoint, then last trade, then 10-min VWAP."""
    for kind in ("mid", "last", "vwap_10m"):
        if not np.isnan(row[f"p_home_{kind}"]):
            return row[f"p_home_{kind}"], kind
    return np.nan, None


# ---------------------------------------------------------------- main

GAME_COLS = ["game_id", "event_ticker", "game_date", "tipoff_utc", "home_team", "away_team",
             "home_market", "away_market", "home_win", "postponed_window", "tipoff_moved",
             "main_sample", "market_open_time"]


def build(game_map: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows, missing_cache = [], []
    for g in game_map[game_map["matched"]][GAME_COLS].itertuples(index=False):
        tip = g.tipoff_utc.timestamp()
        window_start = tip - 30 * 3600
        data, pregame = {}, {}
        for side, ticker in (("home", g.home_market), ("away", g.away_market)):
            data[side] = load_market(ticker)
            pre = load_pregame(ticker)
            if data[side] is None or pre is None:
                missing_cache.append(ticker)
            pregame[side] = (pregame_contracts(data[side][0], pre, tip)
                             if data[side] is not None and pre is not None else np.nan)
        for h in HORIZONS_H:
            s = tip - h * 3600
            row = {c: getattr(g, c) for c in GAME_COLS}
            row.update({"horizon_h": h,
                        "snapshot_utc": pd.Timestamp(s, unit="s", tz="UTC"),
                        "market_open_before_snapshot": g.market_open_time.timestamp() < s,
                        "cache_missing": any(d is None for d in data.values())
                                         or any(np.isnan(v) for v in pregame.values())})
            for side in ("home", "away"):
                row[f"{side}_pregame_contracts"] = pregame[side]
                if data[side] is None:
                    continue
                for k, v in market_snapshot(*data[side], s, window_start).items():
                    row[f"{side}_{k}"] = v
            row["max_spread"] = np.fmax(row.get("home_spread", np.nan),
                                        row.get("away_spread", np.nan))
            for kind in ("mid", "last", "vwap_10m"):
                p, total = normalize(row.get(f"home_{kind}", np.nan),
                                     row.get(f"away_{kind}", np.nan))
                row[f"p_home_{kind}"], row[f"sum_{kind}"] = p, total
            row["p_home"], row["price_source"] = choose_price(row)
            rows.append(row)
    return pd.DataFrame(rows), missing_cache


def main() -> None:
    game_map = pd.read_parquet(GAME_MAP_PATH)
    snaps, missing = build(game_map)
    snaps.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(snaps)} rows ({snaps['game_id'].nunique()} games x "
          f"{len(HORIZONS_H)} horizons) to {OUT_PATH.relative_to(REPO_ROOT)}")
    if missing:
        print(f"WARNING: {len(missing)} markets not in the raw cache: {missing[:10]}")
    summary = snaps.groupby("horizon_h").agg(
        rows=("game_id", "size"),
        p_home_missing=("p_home", lambda x: int(x.isna().sum())),
        source_mid=("price_source", lambda x: int((x == "mid").sum())),
        source_last=("price_source", lambda x: int((x == "last").sum())),
        source_vwap=("price_source", lambda x: int((x == "vwap_10m").sum())),
    ).sort_index(ascending=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()

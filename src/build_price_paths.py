"""Stage 10: price paths on a 5-minute grid for every matched 2025-26 game (plan 10.2).

One row per game per step, from the official tipoff minus 30 hours to tipoff:
minutes_to_tipoff = 1800, 1795, ..., 0 (361 steps). The step at S = tipoff - minutes uses
only data strictly before S, with the same rules as the snapshots (build_snapshots.py):

  home_mid, away_mid   bid/ask midpoint from the last 1-minute candle ending strictly
                       before S. A one-sided book (bid 0 or ask 1.00) gives no midpoint.
  home_mid_carried_forward, away_mid_carried_forward
                       1 when that candle ended more than one minute before S, so the
                       quote was carried forward.
  p_home_mid           home_mid / (home_mid + away_mid), when both sides have a midpoint
  max_spread           the wider of the two bid/ask spreads (kept for one-sided books)
  contracts            contracts traded in both markets in [S - 5 min, S)
  market_open          1 when the game's market had opened before S

Steps before the market opened are empty. So are the quote columns of steps before the
first candle of the pulled 30h window: the raw cache holds no earlier quote, even when the
market was already open. `contracts` uses the pregame trades too, so it is complete for
every step with market_open = 1.

At 24h, 6h, 1h and 0h the values equal the snapshot table's home_mid, away_mid, p_home_mid
and max_spread.

Output: data/processed/price_paths.csv.gz (read it with csv_io.read_csv).

Usage:
    python src/build_price_paths.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_snapshots import CANDLE_SECONDS, load_market, load_pregame, normalize
from csv_io import read_csv, write_csv
from seasons import get_season

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_MAP_PATH = get_season().game_map_path
OUT_PATH = REPO_ROOT / "data" / "processed" / "price_paths.csv.gz"

WINDOW_MINUTES = 30 * 60
STEP_MINUTES = 5
MINUTES_TO_TIPOFF = np.arange(WINDOW_MINUTES, -1, -STEP_MINUTES)      # 1800, 1795, ..., 0
COLUMNS = ["game_id", "minutes_to_tipoff", "market_open", "home_mid", "away_mid",
           "p_home_mid", "max_spread", "contracts", "home_mid_carried_forward",
           "away_mid_carried_forward"]


def quotes_at(candles: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """For each time S: the quote of the last candle ending strictly before S.

    The vectorized form of build_snapshots.quote_before, with the same columns.
    """
    end = candles["end_ts"].to_numpy(dtype=float)
    pos = np.searchsorted(end, times, side="left") - 1        # last candle with end < S
    has = pos >= 0
    take = np.where(has, pos, 0)
    if len(end) == 0:
        bid = ask = quote_end = np.full(len(times), np.nan)
    else:
        bid = np.where(has, candles["bid"].to_numpy(dtype=float)[take], np.nan)
        ask = np.where(has, candles["ask"].to_numpy(dtype=float)[take], np.nan)
        quote_end = np.where(has, end[take], np.nan)
    missing = np.isnan(bid) | np.isnan(ask)
    with np.errstate(invalid="ignore"):
        one_sided = has & (missing | (bid <= 0) | (ask >= 1))
    mid = np.where(has & ~one_sided, (bid + ask) / 2, np.nan)
    # Python's round, as in quote_before, so cent-grid spreads compare exactly.
    spread = np.array([np.nan if m else round(a - b, 4) for a, b, m in zip(ask, bid, missing)])
    age = (np.asarray(times, dtype=float) - quote_end) / 60
    with np.errstate(invalid="ignore"):
        carried = has & (age > CANDLE_SECONDS / 60)
    return pd.DataFrame({"bid": bid, "ask": ask, "mid": mid, "spread": spread,
                         "quote_age_min": age, "one_sided": one_sided, "carried": carried})


def step_contracts(trades: pd.DataFrame, times: np.ndarray, minutes: float) -> np.ndarray:
    """For each time S: contracts traded in [S - minutes, S)."""
    ts = trades["ts"].to_numpy(dtype=float)
    count = trades["count"].to_numpy(dtype=float)
    order = np.argsort(ts, kind="stable")
    ts, count = ts[order], count[order]
    lo = np.searchsorted(ts, np.asarray(times, dtype=float) - minutes * 60, side="left")
    hi = np.searchsorted(ts, times, side="left")
    return np.array([count[a:b].sum() for a, b in zip(lo, hi)])


def all_trades(window: pd.DataFrame, pregame: pd.DataFrame | None) -> pd.DataFrame:
    """Window and pregame trades of one market, without the trades the two pulls share."""
    if pregame is None:
        return window
    return pd.concat([pregame, window], ignore_index=True).drop_duplicates("trade_id")


def game_path(game_id: str, tipoff: float, market_open: float, home: tuple, away: tuple,
              home_pregame: pd.DataFrame | None = None,
              away_pregame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The 361 rows of one game. home and away are (trades, candles) as from load_market."""
    times = tipoff - MINUTES_TO_TIPOFF * 60.0
    is_open = times > market_open
    q = {"home": quotes_at(home[1], times), "away": quotes_at(away[1], times)}
    out = pd.DataFrame({"game_id": game_id, "minutes_to_tipoff": MINUTES_TO_TIPOFF,
                        "market_open": is_open})
    for side in ("home", "away"):
        out[f"{side}_mid"] = q[side]["mid"]
    pairs = [normalize(h, a) for h, a in zip(q["home"]["mid"], q["away"]["mid"])]
    out["p_home_mid"] = [p for p, _ in pairs]
    out["max_spread"] = np.fmax(q["home"]["spread"], q["away"]["spread"])
    out["contracts"] = (
        step_contracts(all_trades(home[0], home_pregame), times, STEP_MINUTES)
        + step_contracts(all_trades(away[0], away_pregame), times, STEP_MINUTES))
    for side in ("home", "away"):
        out[f"{side}_mid_carried_forward"] = q[side]["carried"].astype("boolean")
    # Steps before the market opened are empty.
    closed = ~is_open
    for col in ("home_mid", "away_mid", "p_home_mid", "max_spread", "contracts"):
        out.loc[closed, col] = np.nan
    for col in ("home_mid_carried_forward", "away_mid_carried_forward"):
        out.loc[closed, col] = pd.NA
    return out[COLUMNS]


def build(game_map: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frames, missing = [], []
    for g in game_map[game_map["matched"]].sort_values("game_id").itertuples(index=False):
        home, away = load_market(g.home_market), load_market(g.away_market)
        if home is None or away is None:
            missing.append(g.game_id)
            continue
        frames.append(game_path(g.game_id, g.tipoff_utc.timestamp(),
                                g.market_open_time.timestamp(), home, away,
                                load_pregame(g.home_market), load_pregame(g.away_market)))
    return pd.concat(frames, ignore_index=True), missing


def main() -> None:
    paths, missing = build(read_csv(GAME_MAP_PATH))
    write_csv(paths, OUT_PATH)
    n_games = paths["game_id"].nunique()
    print(f"Saved {len(paths):,} rows ({n_games:,} games x {len(MINUTES_TO_TIPOFF)} steps) "
          f"to {OUT_PATH.relative_to(REPO_ROOT)}")
    if missing:
        print(f"WARNING: {len(missing)} games not in the raw cache: {missing[:10]}")
    print(f"Steps with the market open: {int(paths['market_open'].sum()):,}")
    print(f"Steps with p_home_mid: {int(paths['p_home_mid'].notna().sum()):,}")


if __name__ == "__main__":
    main()

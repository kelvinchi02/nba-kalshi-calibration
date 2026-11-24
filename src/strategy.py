"""Stage 9 strategy backtest (ANALYSIS_PLAN.md section 9.3).

Library for notebooks/05_strategy.ipynb.

One fixed rule, using only what a trader could know at tipoff:
  1. Popularity: each team's expanding mean of its own market pregame volume over its
     earlier games. A game is eligible when both teams have at least 10 earlier games.
  2. Threshold: the 75th percentile of |gap| over the eligible games with an earlier
     tipoff. The first 50 eligible games only set the threshold and are never traded.
  3. Rule: in a later eligible game with |gap| at or above the threshold, stake $100 on
     the less popular team at its tipoff ask. Contracts = floor(100 / ask).
  4. Fee: Kalshi's taker fee, 0.07 * C * P * (1 - P), rounded up to the next cent, once
     per order. No settlement fee.

The benchmark buys the less popular team in every tradable game. The robustness check
uses the metro population gap in place of the volume gap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import stats_tools
from csv_io import read_csv
from eda import team_game_volume
from popularity import log_gap, metro_gap
from stats_tools import B, SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
RESULTS_DIR = REPO_ROOT / "results"

MIN_PRIOR_GAMES = 10       # per team, before a game is eligible
BURN_IN_GAMES = 50         # first eligible games: threshold only, no trades
THRESHOLD_QUANTILE = 0.75
STAKE = 100.0              # dollars per trade
FEE_RATE_PER_10000 = 700   # 0.07, the taker fee coefficient
PRICE_UNITS = 10_000       # prices are handled in 1/100 of a cent, so sums are exact

FEE_SOURCES = [
    ("Kalshi fee schedule as filed with the CFTC: fees = round up(0.07 x C x P x (1-P)), "
     "rounded to the next cent; no settlement fee",
     "https://cftc.gov/filings/orgrules/rule091222kexdcm003.pdf"),
    ("Kalshi API, GET /series/KXNBAGAME: fee_type = quadratic_with_maker_fees, "
     "fee_multiplier = 1; GET /series/fee_changes lists no change for the series",
     "https://api.elections.kalshi.com/trade-api/v2/series/KXNBAGAME"),
]
FEE_CHECKED = "2026-11-23"


# ---------------------------------------------------------------- popularity known in time

def expanding_prior_mean(long: pd.DataFrame) -> pd.DataFrame:
    """For each team and game: how many earlier games the team has, and their mean volume.

    `long` is a team_game_volume table with tipoff_utc. Only games with an earlier tipoff
    count, so the current game and later games never enter.
    """
    if long.duplicated(["team", "tipoff_utc"]).any():
        raise ValueError("A team has two games with the same tipoff")
    out = long.sort_values(["team", "tipoff_utc"]).copy()
    by_team = out.groupby("team")["pregame_contracts"]
    out["prior_games"] = by_team.cumcount()
    # The running total up to the previous game. Shifting it, instead of subtracting the
    # current game's volume, keeps the current game out of the arithmetic altogether.
    prior_sum = by_team.cumsum().groupby(out["team"]).shift(1).fillna(0.0)
    out["prior_mean"] = (prior_sum / out["prior_games"]).where(out["prior_games"] > 0)
    return out


def load_games(metro: pd.DataFrame) -> pd.DataFrame:
    """Main sample at tipoff, one row per game, with both gaps and the eligibility flags."""
    sn = read_csv(SNAP_PATH)
    at0 = sn[(sn["horizon_h"] == 0) & sn["main_sample"]]
    return build_games(at0, metro)


def build_games(at0: pd.DataFrame, metro: pd.DataFrame) -> pd.DataFrame:
    prior = expanding_prior_mean(team_game_volume(at0))
    games = at0[["game_id", "game_date", "tipoff_utc", "home_team", "away_team", "home_win",
                 "home_ask", "away_ask"]].copy()
    games["home_win"] = games["home_win"].astype(int)
    for side in ("home", "away"):
        part = prior.loc[prior["side"] == side, ["game_id", "prior_games", "prior_mean"]]
        part = part.rename(columns={"prior_games": f"{side}_prior_games",
                                    "prior_mean": f"{side}_prior_mean"})
        games = games.merge(part, on="game_id", validate="one_to_one")
    games["eligible"] = ((games["home_prior_games"] >= MIN_PRIOR_GAMES)
                         & (games["away_prior_games"] >= MIN_PRIOR_GAMES))
    games["gap_volume"] = log_gap(games["home_prior_mean"], games["away_prior_mean"])
    games = games.merge(metro_gap(at0, metro)[["game_id", "gap_metro"]], on="game_id",
                        validate="one_to_one")
    games = games.sort_values(["tipoff_utc", "game_id"], ignore_index=True)
    games["eligible_order"] = np.where(games["eligible"], games["eligible"].cumsum(), 0)
    games["tradable"] = games["eligible"] & (games["eligible_order"] > BURN_IN_GAMES)
    return games


def expanding_threshold(games: pd.DataFrame, gap_col: str,
                        quantile: float = THRESHOLD_QUANTILE) -> pd.Series:
    """For each eligible game: the quantile of |gap| over eligible games with an earlier tipoff.

    Missing for games that are not eligible and for games with no earlier eligible game.
    """
    eligible = games[games["eligible"]].sort_values(["tipoff_utc", "game_id"])
    tipoff = eligible["tipoff_utc"].to_numpy()
    size = eligible[gap_col].abs().to_numpy(dtype=float)
    out = pd.Series(np.nan, index=games.index, dtype=float)
    for i, idx in enumerate(eligible.index):
        earlier = size[tipoff < tipoff[i]]
        if len(earlier):
            out.loc[idx] = np.quantile(earlier, quantile)
    return out


# ---------------------------------------------------------------- one order

def price_units(ask: float) -> int:
    """The ask in 1/100 of a cent, as an exact integer."""
    units = round(ask * PRICE_UNITS)
    if abs(ask * PRICE_UNITS - units) > 1e-6:
        raise ValueError(f"Ask {ask!r} has more than four decimals")
    return int(units)


def contracts_for_stake(ask: float, stake: float = STAKE) -> int:
    """floor(stake / ask)."""
    return int(round(stake * PRICE_UNITS)) // price_units(ask)


def taker_fee(contracts: int, ask: float) -> tuple[float, float]:
    """(fee, fee without rounding) in dollars for one order.

    fee = 0.07 * C * P * (1 - P), rounded up to the next cent. Computed in integers so a
    fee that is already a whole number of cents is not pushed up by float error.
    """
    u = price_units(ask)
    numerator = FEE_RATE_PER_10000 * contracts * u * (PRICE_UNITS - u)  # dollars * 1e12
    cents = -(-numerator // 10**10)
    return cents / 100, numerator / 10**12


# ---------------------------------------------------------------- trades

def make_trades(games: pd.DataFrame, gap_col: str, use_threshold: bool) -> pd.DataFrame:
    """One row per tradable game, with the order placed in it, if any.

    With use_threshold, a game trades only when |gap| is at or above its expanding
    threshold. Without it (the benchmark), every tradable game trades. The contracts bought
    are on the team with the lower measure. Columns cost, fee and profit are 0 where there
    is no trade.
    """
    threshold = expanding_threshold(games, gap_col)
    rows = []
    for i, g in games[games["tradable"]].iterrows():
        gap = g[gap_col]
        signal = (abs(gap) >= threshold[i]) if use_threshold else True
        side = "away" if gap > 0 else "home" if gap < 0 else None
        rec = {"game_id": g["game_id"], "game_date": g["game_date"],
               "tipoff_utc": g["tipoff_utc"], "home_team": g["home_team"],
               "away_team": g["away_team"], "gap": gap,
               "threshold": threshold[i] if use_threshold else np.nan,
               "signal": int(bool(signal)), "side": side, "team": None, "ask": np.nan,
               "status": "no_signal", "contracts": 0, "cost": 0.0, "fee": 0.0,
               "fee_unrounded": 0.0, "won": pd.NA, "payout": 0.0, "profit": 0.0}
        if signal:
            ask = g[f"{side}_ask"] if side else np.nan
            if side is None:
                rec["status"] = "no_trade_gap_zero"
            elif pd.isna(ask) or not 0 < ask < 1:
                rec["status"], rec["ask"] = "no_trade_bad_ask", ask
            else:
                c = contracts_for_stake(ask)
                fee, fee_unrounded = taker_fee(c, ask)
                won = int(g["home_win"] == (1 if side == "home" else 0))
                cost = c * price_units(ask) / PRICE_UNITS
                rec |= {"status": "trade", "team": g[f"{side}_team"], "ask": ask,
                        "contracts": c, "cost": cost, "fee": fee,
                        "fee_unrounded": fee_unrounded, "won": won, "payout": float(c * won),
                        "profit": c * won - cost - fee}
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["won"] = out["won"].astype("Int64")
    return out


def summarize(trade_table: pd.DataFrame) -> dict:
    """Totals over the trades of one strategy."""
    t = trade_table[trade_table["status"] == "trade"]
    outlay = t["cost"].sum() + t["fee"].sum()
    return {"tradable_games": len(trade_table), "signals": int(trade_table["signal"].sum()),
            "no_trade_bad_ask": int((trade_table["status"] == "no_trade_bad_ask").sum()),
            "no_trade_gap_zero": int((trade_table["status"] == "no_trade_gap_zero").sum()),
            "n_trades": len(t), "wins": int(t["won"].sum()),
            "win_rate": float(t["won"].mean()) if len(t) else np.nan,
            "mean_ask": float(t["ask"].mean()) if len(t) else np.nan,
            "contracts": int(t["contracts"].sum()), "total_cost": float(t["cost"].sum()),
            "total_fees": float(t["fee"].sum()),
            "total_fees_unrounded": float(t["fee_unrounded"].sum()),
            "profit": float(t["profit"].sum()),
            "return_per_dollar": float(t["profit"].sum() / outlay) if outlay else np.nan}


def bootstrap_trades(trade_table: pd.DataFrame, n_boot: int = B,
                     seed: int = SEED) -> pd.DataFrame:
    """Block bootstrap over the dates of the tradable games.

    The trades stay as the rule chose them. A replicate's return is its profit divided by
    its own cost plus fees.
    """
    profit = trade_table["profit"].to_numpy(dtype=float)
    outlay = (trade_table["cost"] + trade_table["fee"]).to_numpy(dtype=float)

    def replicate(idx: np.ndarray) -> dict:
        total, staked = profit[idx].sum(), outlay[idx].sum()
        return {"profit": total, "return_per_dollar": total / staked if staked else np.nan}

    return stats_tools.block_bootstrap(trade_table["game_date"], replicate, n_boot=n_boot,
                                       seed=seed)


def results_row(strategy: str, trade_table: pd.DataFrame, boot: pd.DataFrame) -> dict:
    """The summary of one strategy with its bootstrap intervals."""
    s = summarize(trade_table)
    point = {"profit": s["profit"], "return_per_dollar": s["return_per_dollar"]}
    ci = stats_tools.percentile_summary(boot, point,
                                        trade_table["game_date"].nunique()).set_index("statistic")
    return {"strategy": strategy, **s,
            "profit_boot_lo": ci.loc["profit", "boot_lo"],
            "profit_boot_hi": ci.loc["profit", "boot_hi"],
            "return_boot_lo": ci.loc["return_per_dollar", "boot_lo"],
            "return_boot_hi": ci.loc["return_per_dollar", "boot_hi"]}


def sample_counts(games: pd.DataFrame) -> pd.DataFrame:
    eligible = games[games["eligible"]]
    return pd.DataFrame([
        {"group": "main sample games", "games": len(games)},
        {"group": f"eligible (both teams have {MIN_PRIOR_GAMES}+ earlier games)",
         "games": len(eligible)},
        {"group": f"first {BURN_IN_GAMES} eligible games (threshold only)",
         "games": int((eligible["eligible_order"] <= BURN_IN_GAMES).sum())},
        {"group": "tradable games", "games": int(games["tradable"].sum())},
    ])


def run_strategy(games: pd.DataFrame, name: str, gap_col: str, use_threshold: bool,
                 n_boot: int = B, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Trades of one strategy and its results row, bootstrap intervals included."""
    table = make_trades(games, gap_col, use_threshold)
    table.insert(0, "strategy", name)
    boot = bootstrap_trades(table, n_boot=n_boot, seed=seed)
    return table, results_row(name, table, boot)


# ---------------------------------------------------------------- exploratory (plan E1, E2)

def expanding_gap_sample(at0: pd.DataFrame, metro: pd.DataFrame) -> pd.DataFrame:
    """Eligible games with the tipoff price and the past-games-only volume gap (plan E1).

    One row per eligible game, in the layout popularity.analyze_measure expects, with the
    gap in the column gap_volume_expanding.
    """
    games = build_games(at0, metro)
    eligible = games.loc[games["eligible"], ["game_id", "game_date", "home_team",
                                             "away_team", "home_win", "gap_volume"]]
    eligible = eligible.rename(columns={"gap_volume": "gap_volume_expanding"})
    d = eligible.merge(at0[["game_id", "p_home"]], on="game_id", validate="one_to_one")
    d = d.dropna(subset=["p_home"])
    return d.sort_values("game_id", ignore_index=True)


def load_expanding_gap_sample(metro: pd.DataFrame) -> pd.DataFrame:
    sn = read_csv(SNAP_PATH)
    return expanding_gap_sample(sn[(sn["horizon_h"] == 0) & sn["main_sample"]], metro)


def before_fees_table(results: pd.DataFrame) -> pd.DataFrame:
    """Profit and return of each strategy with the fees left out (plan E2).

    profit before fees = payout - cost = profit + total fees.
    return before fees = profit before fees / total cost.
    """
    out = results[["strategy", "n_trades", "total_cost", "total_fees", "profit",
                   "return_per_dollar"]].copy()
    out["profit_before_fees"] = out["profit"] + out["total_fees"]
    out["return_before_fees"] = out["profit_before_fees"] / out["total_cost"]
    return out

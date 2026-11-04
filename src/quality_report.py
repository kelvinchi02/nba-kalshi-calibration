"""Stage 7: data quality report for the snapshot table.

Covers coverage and data quality only: match counts, cache coverage, missing snapshots,
quote staleness, low-volume games and two-sided price sums. It deliberately does NOT
compare prices with outcomes (no calibration, Brier scores or regressions).

Output: data/processed/quality_report.md

Usage:
    python src/quality_report.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from csv_io import read_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_MAP_PATH = REPO_ROOT / "data" / "processed" / "game_map.csv"
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
CACHE_DIR = REPO_ROOT / "data" / "raw" / "kalshi" / "markets"
FAILURES_PATH = REPO_ROOT / "logs" / "pull_kalshi_failures.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "quality_report.md"

SUM_TOLERANCES = (0.05, 0.10)
LOW_VOLUME_QUANTILE = 0.02
WIDE_SPREAD = 0.05  # dollars


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> list[str]:
    if df.empty:
        return ["(none)", ""]
    df = df.reset_index() if df.index.name else df
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return out + [""]


def main() -> None:
    gm = read_csv(GAME_MAP_PATH)
    sn = read_csv(SNAP_PATH)
    L = ["# Data quality report (2025-26 regular season)", "",
         "Coverage and data quality only. No calibration results.", ""]

    # ------------------------------------------------------------ matching and cache
    expected = set(gm.loc[gm["matched"], "home_market"]) | set(gm.loc[gm["matched"], "away_market"])
    cached = {p.name.removesuffix(".json.gz") for p in CACHE_DIR.glob("*.json.gz")}
    failures = (list(csv.DictReader(FAILURES_PATH.open(encoding="utf-8")))
                if FAILURES_PATH.exists() else [])
    L += ["## Games and markets", "",
          f"- NBA regular-season games: {len(gm)}",
          f"- Matched to Kalshi: {gm['matched'].sum()} "
          f"(exact {gm['match_type'].eq('exact').sum()}, +/-1 day {gm['match_type'].eq('date_shift_1d').sum()})",
          f"- main_sample (matched, not postponed_window): {gm['main_sample'].sum()}",
          f"- postponed_window: {', '.join(gm.loc[gm['postponed_window'], 'event_ticker'])}",
          f"- tipoff_moved: {', '.join(gm.loc[gm['tipoff_moved'], 'event_ticker'])}",
          f"- Market files cached: {len(expected & cached)} of {len(expected)}; "
          f"failures in last pull run: {len(failures)}",
          f"- Snapshot rows: {len(sn)} ({sn['game_id'].nunique()} games x "
          f"{sn['horizon_h'].nunique()} horizons)", ""]

    # ------------------------------------------------------------ missing snapshots
    def per_h(f):
        return sn.groupby("horizon_h").apply(f, include_groups=False)

    missing = pd.DataFrame({
        "rows": per_h(len),
        "market_not_open": per_h(lambda d: int((~d["market_open_before_snapshot"]).sum())),
        "mid_missing": per_h(lambda d: int(d["p_home_mid"].isna().sum())),
        "one_sided_quote": per_h(lambda d: int((d["home_one_sided"] | d["away_one_sided"]).sum())),
        "last_missing": per_h(lambda d: int(d["p_home_last"].isna().sum())),
        "vwap10_missing": per_h(lambda d: int(d["p_home_vwap_10m"].isna().sum())),
        "p_home_missing": per_h(lambda d: int(d["p_home"].isna().sum())),
        "src_mid": per_h(lambda d: int((d["price_source"] == "mid").sum())),
        "src_last": per_h(lambda d: int((d["price_source"] == "last").sum())),
        "src_vwap": per_h(lambda d: int((d["price_source"] == "vwap_10m").sum())),
    }).sort_index(ascending=False)
    missing.index.name = "horizon_h"
    L += ["## Missing snapshots per horizon", "",
          "mid/last/vwap10 = that price type missing on at least one side. "
          "p_home = main estimate (mid, else last, else vwap10).", ""]
    L += md_table(missing)
    no_price = sn[sn["p_home"].isna()][["event_ticker", "horizon_h", "market_open_before_snapshot"]]
    L += ["Rows with no price at all:", ""] + md_table(no_price)

    # ------------------------------------------------------------ quote staleness
    age = sn[["horizon_h", "home_quote_age_min", "away_quote_age_min"]].copy()
    age["max_age"] = age[["home_quote_age_min", "away_quote_age_min"]].max(axis=1)
    stale = age.groupby("horizon_h")["max_age"].describe(percentiles=[0.5, 0.95, 0.99])[
        ["count", "50%", "95%", "99%", "max"]].sort_index(ascending=False)
    stale["over_30min"] = age.groupby("horizon_h")["max_age"].apply(lambda x: int((x > 30).sum()))
    L += ["## Quote age (minutes since the candle used for the midpoint, worse side)", "",
          "Minimum is 1 minute by construction (the candle ending at the snapshot is not used).",
          ""]
    L += md_table(stale, "{:.1f}")

    # ------------------------------------------------------------ bid/ask spread
    sp = pd.concat([sn[["horizon_h", "home_spread"]].rename(columns={"home_spread": "spread"}),
                    sn[["horizon_h", "away_spread"]].rename(columns={"away_spread": "spread"})])
    spread_tbl = sp.groupby("horizon_h")["spread"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])[
        ["count", "mean", "50%", "90%", "95%", "99%", "max"]].sort_index(ascending=False)
    wide = sn.assign(wide=sn["max_spread"] > WIDE_SPREAD).groupby("horizon_h").agg(
        rows=("game_id", "size"),
        rows_spread_gt_5c=("wide", "sum"),
        home_gt_5c=("home_spread", lambda x: int((x > WIDE_SPREAD).sum())),
        away_gt_5c=("away_spread", lambda x: int((x > WIDE_SPREAD).sum())),
        rows_spread_missing=("max_spread", lambda x: int(x.isna().sum())),
    ).sort_index(ascending=False)
    L += ["## Bid/ask spread (dollars, from the candle used for the midpoint)", "",
          "Distribution over both team markets:", ""]
    L += md_table(spread_tbl, "{:.3f}")
    L += [f"Rows where either side's spread is above {WIDE_SPREAD * 100:.0f} cents "
          f"(max_spread > {WIDE_SPREAD}):", ""]
    L += md_table(wide)
    widest = sn.nlargest(10, "max_spread")[
        ["event_ticker", "horizon_h", "home_bid", "home_ask", "away_bid", "away_ask",
         "max_spread", "home_one_sided", "away_one_sided"]]
    L += ["Widest spreads:", ""] + md_table(widest)

    # ------------------------------------------------------------ low volume
    at0 = sn[sn["horizon_h"] == 0].copy()
    at0["window_contracts"] = at0["home_cum_contracts"] + at0["away_cum_contracts"]
    at0["trades_10m"] = at0["home_n_trades_10m"] + at0["away_n_trades_10m"]
    q = at0["window_contracts"].describe(percentiles=[0.01, 0.05, 0.25, 0.5])
    thresh = at0["window_contracts"].quantile(LOW_VOLUME_QUANTILE)
    low = at0[at0["window_contracts"] <= thresh].sort_values("window_contracts")[
        ["event_ticker", "window_contracts", "trades_10m", "market_open_before_snapshot"]]
    L += ["## Low-volume games", "",
          "Contracts traded (both markets) in the 30h before tipoff:", "",
          f"min {q['min']:,.0f}, p1 {q['1%']:,.0f}, p5 {q['5%']:,.0f}, p25 {q['25%']:,.0f}, "
          f"median {q['50%']:,.0f}, max {q['max']:,.0f}", "",
          f"Games at or below the {LOW_VOLUME_QUANTILE:.0%} quantile ({thresh:,.0f} contracts):", ""]
    L += md_table(low, "{:,.0f}")
    L += [f"Games with no trade in the 10 minutes before tipoff (either side): "
          f"{int(((at0['home_n_trades_10m'] == 0) | (at0['away_n_trades_10m'] == 0)).sum())}", ""]

    # ------------------------------------------------------------ full pregame volume
    games0 = at0.merge(gm[["game_id", "kalshi_volume"]], on="game_id", how="left")
    games0["pregame_contracts"] = games0["home_pregame_contracts"] + games0["away_pregame_contracts"]
    pq = games0["pregame_contracts"].describe(percentiles=[0.01, 0.05, 0.25, 0.5])
    share = (games0["window_contracts"] / games0["pregame_contracts"]).describe()
    L += ["## Full pregame volume (market open to tipoff)", "",
          "Contracts traded (both markets), strictly before the official tipoff:", "",
          f"min {pq['min']:,.0f}, p1 {pq['1%']:,.0f}, p5 {pq['5%']:,.0f}, p25 {pq['25%']:,.0f}, "
          f"median {pq['50%']:,.0f}, max {pq['max']:,.0f}", "",
          f"Share of pregame volume traded in the last 30h: median {share['50%']:.3f}, "
          f"min {share['min']:.3f}", "",
          "Consistency checks (each should be 0):", "",
          f"- games with pregame volume missing: {int(games0['pregame_contracts'].isna().sum())}",
          f"- games where pregame < 30h-window volume: "
          f"{int((games0['pregame_contracts'] < games0['window_contracts'] - 1e-6).sum())}",
          f"- games where pregame > Kalshi lifetime volume (incl. in-game): "
          f"{int((games0['pregame_contracts'] > games0['kalshi_volume'] + 1e-6).sum())}", ""]

    # ------------------------------------------------------------ two-sided sums
    L += ["## Two-sided sums before normalization (home + away)", "",
          "Far from 1 means the two markets disagree (spread, stale price or thin book).", ""]
    rows = []
    for kind in ("mid", "last", "vwap_10m"):
        col = f"sum_{kind}"
        for h, d in sn.groupby("horizon_h"):
            x = d[col].dropna()
            rows.append({"price": kind, "horizon_h": h, "n": len(x),
                         "median": x.median(), "min": x.min(), "max": x.max(),
                         **{f"abs_dev_gt_{t}": int(((x - 1).abs() > t).sum())
                            for t in SUM_TOLERANCES}})
    sums = pd.DataFrame(rows).sort_values(["price", "horizon_h"], ascending=[True, False])
    L += md_table(sums)
    worst = sn.assign(dev=(sn["sum_mid"] - 1).abs()).nlargest(10, "dev")[
        ["event_ticker", "horizon_h", "home_bid", "home_ask", "away_bid", "away_ask", "sum_mid"]]
    L += ["Largest midpoint-sum deviations:", ""] + md_table(worst)

    # ------------------------------------------------------------ flagged games
    flagged = sn[sn["tipoff_moved"] | sn["postponed_window"]][
        ["event_ticker", "horizon_h", "postponed_window", "tipoff_moved", "price_source",
         "sum_mid"]]
    L += ["## Flagged games: snapshot availability", ""] + md_table(flagged)

    OUT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

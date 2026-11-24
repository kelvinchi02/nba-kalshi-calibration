"""Stage 8 EDA (descriptive only, no outcomes used).

Library for notebooks/01_eda.ipynb, which runs it and writes results/.

1. Average pregame volume by team (ANALYSIS_PLAN.md section 4): mean over the team's
   main-sample games of its own market's pregame_contracts, ranked.
2. Example game price path: both team markets' bid/ask midpoint over the 30 hours before
   the official tipoff, with the bid/ask band and the 24h/6h/1h/0h snapshot times, plus
   contracts traded per 30 minutes. The figure reads data/processed/example_price_path.csv,
   which src/export_price_path.py builds from the raw cache.
   Default game: among main-sample games with above-median pregame volume, the one whose
   p_home moved most between the 24h snapshot and tipoff.
3. Distribution of the tipoff price p_home over the main sample: summary and histogram.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_snapshots import CACHE_DIR, HORIZONS_H, load_market
from csv_io import read_csv
from plot_style import COLORS, apply_style

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
PRICE_PATH_CSV = REPO_ROOT / "data" / "processed" / "example_price_path.csv"
RESULTS_DIR = REPO_ROOT / "results"
WINDOW_HOURS = 30
VOLUME_BUCKET_MIN = 30
PRICE_BIN_WIDTH = 0.05


def main_sample() -> pd.DataFrame:
    """Every main-sample snapshot row: one row per game per horizon."""
    sn = read_csv(SNAP_PATH)
    return sn[sn["main_sample"]]


# ---------------------------------------------------------------- 1. price path

def pick_example(sn: pd.DataFrame) -> str:
    """Largest |p_home(0h) - p_home(24h)| among games with above-median pregame volume."""
    wide = sn.pivot(index="event_ticker", columns="horizon_h", values="p_home")
    at0 = sn[sn["horizon_h"] == 0].set_index("event_ticker")
    volume = at0["home_pregame_contracts"] + at0["away_pregame_contracts"]
    move = (wide[0] - wide[24]).abs()
    candidates = move[(volume > volume.median()) & move.notna()]
    return candidates.idxmax()


def minute_quotes(candles: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Close bid/ask on a full 1-minute grid. Missing minutes keep the previous quote.
    One-sided books (bid 0 or ask 1) get no midpoint."""
    grid = np.arange(start + 60, end + 1, 60)
    q = candles.set_index("end_ts")[["bid", "ask"]].reindex(grid).ffill()
    q["mid"] = (q["bid"] + q["ask"]) / 2
    q.loc[(q["bid"] <= 0) | (q["ask"] >= 1), "mid"] = np.nan
    q["hours_before"] = (q.index - end) / 3600
    return q


def load_market_or_fail(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(trades, candles) from the raw window cache, with a clear error when it is absent."""
    market = load_market(ticker)
    if market is None:
        raise FileNotFoundError(
            f"No raw cache for {ticker} in {CACHE_DIR}. data/raw/ is gitignored: "
            "pull it with Stages 5 and 4 in the README run order.")
    return market


def price_path_table(sn: pd.DataFrame, event: str) -> pd.DataFrame:
    """One game's quotes and volume per minute, from the raw cache.

    One row per team market per minute of the 30 hours before the official tipoff:
    the minute's closing bid and ask (carried forward through quiet minutes), their
    midpoint (missing for a one-sided book), and the contracts traded in the minute
    [minute_end - 60s, minute_end). This is what src/export_price_path.py saves to
    data/processed/example_price_path.csv.
    """
    row = sn[(sn["event_ticker"] == event) & (sn["horizon_h"] == 0)].iloc[0]
    tip = row["tipoff_utc"].timestamp()
    start = tip - WINDOW_HOURS * 3600
    frames = []
    for side in ("home", "away"):
        ticker = row[f"{side}_market"]
        trades, candles = load_market_or_fail(ticker)
        q = minute_quotes(candles, start, tip)
        tr = trades[(trades["ts"] >= start) & (trades["ts"] < tip)]
        edges = np.arange(start, tip + 1, 60)
        contracts = np.histogram(tr["ts"], bins=edges, weights=tr["count"])[0]
        frames.append(pd.DataFrame({
            "event_ticker": event, "game_id": row["game_id"],
            "tipoff_utc": row["tipoff_utc"], "side": side, "team": row[f"{side}_team"],
            "market_ticker": ticker,
            "minute_end_utc": pd.to_datetime(q.index.to_numpy(), unit="s", utc=True),
            "bid": q["bid"].to_numpy(), "ask": q["ask"].to_numpy(),
            "mid": q["mid"].to_numpy(), "contracts": contracts}))
    return pd.concat(frames, ignore_index=True)


def price_path_figure(pp: pd.DataFrame, sn: pd.DataFrame,
                      path: Path) -> tuple[plt.Figure, dict]:
    """Save the price-path figure of the game in `pp` (a price_path_table) to `path`.

    Returns the figure and a few numbers on it.
    """
    event = pp["event_ticker"].iloc[0]
    row = sn[(sn["event_ticker"] == event) & (sn["horizon_h"] == 0)].iloc[0]
    hours_before = (pp["minute_end_utc"] - pp["tipoff_utc"]).dt.total_seconds() / 3600

    apply_style()
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
    buckets = np.arange(-WINDOW_HOURS, 0 + 1e-9, VOLUME_BUCKET_MIN / 60)
    total = np.zeros(len(buckets) - 1)
    for side, color in (("home", COLORS["series1"]), ("away", COLORS["series2"])):
        q = pp[pp["side"] == side]
        hours, team = hours_before[q.index], q["team"].iloc[0]
        ax.fill_between(hours, q["bid"], q["ask"], step="post", color=color, alpha=0.18, lw=0)
        ax.step(hours, q["mid"], where="post", color=color, lw=2,
                label=f"{team} ({side}) midpoint, bid/ask band")
        last = q["mid"].dropna().iloc[-1]
        ax.text(0.25, last, team, color=COLORS["text_primary"], va="center", fontsize=9)
        total += q["contracts"].to_numpy().reshape(-1, VOLUME_BUCKET_MIN).sum(axis=1)

    for h in HORIZONS_H:
        for a in (ax, axv):
            a.axvline(-h, color=COLORS["reference"], lw=1, ls=(0, (4, 3)), zorder=0)
        # 1h and tipoff sit close together: push their labels apart
        ha = {1: "right", 0: "left"}.get(h, "center")
        ax.text(-h, 1.01, "tipoff" if h == 0 else f"{h}h", transform=ax.get_xaxis_transform(),
                ha=ha, va="bottom", fontsize=8.5, color=COLORS["text_secondary"])

    ax.set_ylim(0, 1)
    ax.set_ylabel("YES price (dollars)")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(f"{row['away_team']} at {row['home_team']}, {row['game_date']:%b %d, %Y}: "
                 f"Kalshi prices in the 30 hours before tipoff", loc="left", pad=18)
    axv.bar(buckets[:-1], total / 1000, width=VOLUME_BUCKET_MIN / 60 * 0.85, align="edge",
            color=COLORS["reference"])
    axv.set_ylabel(f"Contracts per\n{VOLUME_BUCKET_MIN} min (thousands)")
    axv.set_xlabel("Hours before official tipoff")
    axv.set_xlim(-WINDOW_HOURS, 1.6)
    axv.set_xticks(range(-30, 1, 3))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    info = {"event": event, "window_contracts": float(total.sum()),
            **{f"p_home_{h}h": float(sn[(sn["event_ticker"] == event) &
                                        (sn["horizon_h"] == h)]["p_home"].iloc[0])
               for h in HORIZONS_H}}
    return fig, info


# ---------------------------------------------------------------- 2. team volume

def team_game_volume(at0: pd.DataFrame) -> pd.DataFrame:
    """Each team's own market volume in each game: two rows per game (home, then away).

    `at0` holds one row per game. A team's own market is the home market when it plays at
    home and the away market when it plays away.
    """
    keep = [c for c in ("game_id", "game_date", "tipoff_utc") if c in at0.columns]
    sides = []
    for side in ("home", "away"):
        part = at0[keep + [f"{side}_team", f"{side}_pregame_contracts"]].set_axis(
            keep + ["team", "pregame_contracts"], axis=1)
        sides.append(part.assign(side=side))
    return pd.concat(sides, ignore_index=True)


def team_pregame_volume(sn: pd.DataFrame) -> pd.DataFrame:
    long = team_game_volume(sn[sn["horizon_h"] == 0])
    out = long.groupby("team")["pregame_contracts"].agg(
        games="size", mean_pregame_contracts="mean",
        median_pregame_contracts="median").reset_index()
    out = out.sort_values("mean_pregame_contracts", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def team_volume_figure(tv: pd.DataFrame, path: Path) -> plt.Figure:
    """Save the ranked team-volume bar chart to `path` and return the figure."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 8.2))
    y = np.arange(len(tv))[::-1]
    ax.barh(y, tv["mean_pregame_contracts"] / 1000, height=0.72, color=COLORS["series1"])
    ax.set_yticks(y, tv["team"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    for i in list(range(3)) + list(range(len(tv) - 3, len(tv))):
        v = tv.loc[i, "mean_pregame_contracts"] / 1000
        ax.text(v + 6, y[i], f"{v:,.0f}k", va="center", fontsize=8.5,
                color=COLORS["text_secondary"])
    league = tv["mean_pregame_contracts"].mean() / 1000
    ax.axvline(league, color=COLORS["reference"], lw=1, ls=(0, (4, 3)))
    ax.text(league, len(tv) - 0.2, f" mean of teams {league:,.0f}k", fontsize=8.5,
            color=COLORS["text_secondary"], va="bottom")
    ax.set_xlabel("Average contracts traded on the team's own market per game, "
                  "market open to tipoff (thousands)")
    ax.set_title("Average pregame volume by team, 2025-26 regular season", loc="left")
    ax.set_ylim(-0.7, len(tv) + 0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ---------------------------------------------------------------- 3. price distribution

def tipoff_prices(sn: pd.DataFrame) -> pd.Series:
    """p_home at horizon 0, one value per main-sample game, missing values dropped."""
    return sn.loc[sn["horizon_h"] == 0, "p_home"].dropna().reset_index(drop=True)


def price_summary(p: pd.Series) -> pd.DataFrame:
    """One row: count, mean, spread and quantiles of the tipoff price."""
    return pd.DataFrame([{
        "n": len(p), "mean": p.mean(), "sd": p.std(ddof=1), "min": p.min(),
        "p10": p.quantile(0.10), "p25": p.quantile(0.25), "median": p.median(),
        "p75": p.quantile(0.75), "p90": p.quantile(0.90), "max": p.max(),
        "share_above_half": (p > 0.5).mean()}])


def price_bin_edges() -> np.ndarray:
    return np.round(np.arange(0, 1 + PRICE_BIN_WIDTH / 2, PRICE_BIN_WIDTH), 10)


def price_histogram(p: pd.Series) -> pd.DataFrame:
    """Games per 0.05-wide price bin. Bins are [lo, hi), and the last one includes 1."""
    edges = price_bin_edges()
    counts = np.histogram(p, bins=edges)[0]
    return pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "games": counts,
                         "share": counts / len(p)})


def price_distribution_figure(p: pd.Series, path: Path) -> plt.Figure:
    """Save a histogram of the tipoff price to `path` and return the figure."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    hist = price_histogram(p)
    ax.bar(hist["bin_lo"] + PRICE_BIN_WIDTH * 0.05, hist["games"],
           width=PRICE_BIN_WIDTH * 0.9, align="edge", color=COLORS["series1"])
    ax.grid(axis="x", visible=False)
    median = p.median()
    ax.axvline(median, color=COLORS["reference"], lw=1, ls=(0, (4, 3)))
    ax.text(median, 1.01, f"median {median:.3f}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color=COLORS["text_secondary"])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Kalshi home-win price at tipoff (p_home), bins of 0.05")
    ax.set_ylabel("Games")
    ax.set_title(f"Distribution of the tipoff price: {len(p):,} games", loc="left", pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig

"""Stage 8 EDA figures (descriptive only, no outcomes used).

1. Example game price path from the raw cache: both team markets' bid/ask midpoint over the
   30 hours before the official tipoff, with the bid/ask band and the 24h/6h/1h/0h snapshot
   times, plus contracts traded per 30 minutes.
   Default game: among main-sample games with above-median pregame volume, the one whose
   p_home moved most between the 24h snapshot and tipoff. --event picks another game.
2. Average pregame volume by team (ANALYSIS_PLAN.md section 4): mean over the team's
   main-sample games of its own market's pregame_contracts, ranked.

Outputs in results/: eda_price_path.png, eda_team_pregame_volume.png and
eda_team_pregame_volume.csv.

Usage:
    python src/eda.py
    python src/eda.py --event KXNBAGAME-25DEC13SASOKC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_snapshots import HORIZONS_H, load_market  # noqa: E402
from csv_io import read_csv, write_csv  # noqa: E402
from plot_style import COLORS, apply_style  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = REPO_ROOT / "data" / "processed" / "snapshots.csv"
RESULTS_DIR = REPO_ROOT / "results"
WINDOW_HOURS = 30
VOLUME_BUCKET_MIN = 30


def main_sample_at_tipoff() -> pd.DataFrame:
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


def price_path_figure(sn: pd.DataFrame, event: str, path: Path) -> dict:
    row = sn[(sn["event_ticker"] == event) & (sn["horizon_h"] == 0)].iloc[0]
    tip = row["tipoff_utc"].timestamp()
    start = tip - WINDOW_HOURS * 3600
    sides = [("home", row["home_team"], row["home_market"], COLORS["series1"]),
             ("away", row["away_team"], row["away_market"], COLORS["series2"])]

    apply_style()
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
    buckets = np.arange(-WINDOW_HOURS, 0 + 1e-9, VOLUME_BUCKET_MIN / 60)
    total = np.zeros(len(buckets) - 1)
    for side, team, ticker, color in sides:
        trades, candles = load_market(ticker)
        q = minute_quotes(candles, start, tip)
        ax.fill_between(q["hours_before"], q["bid"], q["ask"], step="post", color=color,
                        alpha=0.18, lw=0)
        ax.step(q["hours_before"], q["mid"], where="post", color=color, lw=2,
                label=f"{team} ({side}) midpoint, bid/ask band")
        last = q["mid"].dropna().iloc[-1]
        ax.text(0.25, last, team, color=COLORS["text_primary"], va="center", fontsize=9)
        tr = trades[(trades["ts"] >= start) & (trades["ts"] < tip)]
        total += np.histogram((tr["ts"] - tip) / 3600, bins=buckets, weights=tr["count"])[0]

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
    plt.close(fig)
    return {"event": event, "window_contracts": float(total.sum()),
            **{f"p_home_{h}h": float(sn[(sn["event_ticker"] == event) &
                                        (sn["horizon_h"] == h)]["p_home"].iloc[0])
               for h in HORIZONS_H}}


# ---------------------------------------------------------------- 2. team volume

def team_pregame_volume(sn: pd.DataFrame) -> pd.DataFrame:
    at0 = sn[sn["horizon_h"] == 0]
    long = pd.concat([
        at0[["game_id", "home_team", "home_pregame_contracts"]].set_axis(
            ["game_id", "team", "pregame_contracts"], axis=1),
        at0[["game_id", "away_team", "away_pregame_contracts"]].set_axis(
            ["game_id", "team", "pregame_contracts"], axis=1),
    ])
    out = long.groupby("team")["pregame_contracts"].agg(
        games="size", mean_pregame_contracts="mean",
        median_pregame_contracts="median").reset_index()
    out = out.sort_values("mean_pregame_contracts", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def team_volume_figure(tv: pd.DataFrame, path: Path) -> None:
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
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", help="event ticker for the price-path figure")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    sn = main_sample_at_tipoff()

    event = args.event or pick_example(sn)
    info = price_path_figure(sn, event, RESULTS_DIR / "eda_price_path.png")
    print("Price path:", {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in info.items()})

    tv = team_pregame_volume(sn)
    write_csv(tv, RESULTS_DIR / "eda_team_pregame_volume.csv")
    team_volume_figure(tv, RESULTS_DIR / "eda_team_pregame_volume.png")
    print(tv.to_string(index=False, float_format="{:,.0f}".format))


if __name__ == "__main__":
    main()

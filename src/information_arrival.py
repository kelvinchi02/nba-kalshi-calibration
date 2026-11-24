"""Stage 10 information arrival (ANALYSIS_PLAN.md section 10.4).

Library for notebooks/07_information_arrival.ipynb.

D1 (descriptive): when do prices move? From data/processed/price_paths.csv.gz, the
absolute change in logit(p_home_mid) per 5-minute step, averaged by minutes to tipoff and
by US Eastern clock time, and the share of all movement in each window before tipoff.

D2 (event study): how does the price of a team move when the official injury report rules
one of its key players Out? Events come from injury_events.py. For each event, the change
in the affected team's normalized midpoint from 5 minutes before to 30 minutes after the
release, a placebo window 3 hours earlier, and an anticipation window from 60 to 5 minutes
before the release. Means are tested against zero with the block bootstrap by game date.
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import all_tests
import stats_tools
from csv_io import read_csv
from plot_style import COLORS, apply_style
from stats_tools import B, SEED, bootstrap_p_value, logit

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS_PATH = REPO_ROOT / "data" / "processed" / "price_paths.csv.gz"
GAME_MAP_PATH = REPO_ROOT / "data" / "processed" / "game_map.csv"
RESULTS_DIR = REPO_ROOT / "results"
PART = "information_arrival"
EASTERN = ZoneInfo("America/New_York")

STEP = 5                                   # minutes between steps
MAIN_RANGE_MINUTES = 24 * 60               # the main descriptive range: 24h to tipoff
# (label, first minute included, first minute excluded), by the step's minutes to tipoff
WINDOWS = (("24h to 6h", 360, 1440), ("6h to 1h", 60, 360), ("1h to 30 min", 30, 60),
           ("last 30 min", 0, 30))
EARLY_WINDOW = ("30h to 24h", 1440, 1800)
MIN_CLOCK_STEPS = 50                       # clock bins with fewer game-steps are not plotted

BEFORE_MIN, AFTER_MIN = 5, 30              # the main event window
ANTICIPATION_MIN = 60                      # the anticipation window starts here
PLACEBO_SHIFT_MIN = 180                    # the placebo windows are 3 hours earlier
PATH_OFFSETS = np.arange(-60, 121, STEP)   # the figure: -60 to +120 minutes


# ================================================================ D1: when prices move

def load_paths() -> pd.DataFrame:
    """Price paths of the main-sample games, with game_date and tipoff_utc."""
    paths = read_csv(PATHS_PATH)
    games = read_csv(GAME_MAP_PATH)
    games = games.loc[games["main_sample"], ["game_id", "game_date", "tipoff_utc"]]
    return paths.merge(games, on="game_id", validate="many_to_one")


def step_changes(paths: pd.DataFrame) -> pd.DataFrame:
    """Absolute change in logit(p_home_mid) from the previous step, per game and step.

    A change exists only when both steps have a midpoint. Gaps are not bridged. The first
    step of a game (1800 minutes) has no previous step.
    """
    d = paths.sort_values(["game_id", "minutes_to_tipoff"], ascending=[True, False])
    d = d[["game_id", "game_date", "tipoff_utc", "minutes_to_tipoff", "p_home_mid"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        d["logit_p"] = logit(d["p_home_mid"])
    previous = d.groupby("game_id")["logit_p"].shift(1)
    previous_minutes = d.groupby("game_id")["minutes_to_tipoff"].shift(1)
    d["abs_change"] = (d["logit_p"] - previous).abs()
    d.loc[previous_minutes != d["minutes_to_tipoff"] + STEP, "abs_change"] = np.nan
    d = d[d["minutes_to_tipoff"] < d.groupby("game_id")["minutes_to_tipoff"].transform("max")]
    return d.drop(columns="logit_p").reset_index(drop=True)


def by_minutes(changes: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute change at each step, over the games that have one."""
    out = changes.groupby("minutes_to_tipoff")["abs_change"].agg(
        mean_abs_change="mean", game_steps="count", games_possible="size").reset_index()
    out["hours_to_tipoff"] = out["minutes_to_tipoff"] / 60
    out["range"] = np.where(out["minutes_to_tipoff"] < MAIN_RANGE_MINUTES, "24h to tipoff",
                            "30h to 24h")
    return out.sort_values("minutes_to_tipoff", ascending=False, ignore_index=True)


def by_clock(changes: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute change by US Eastern clock time, for steps on the game's own date.

    A step is placed by its end time, in 5-minute clock bins.
    """
    d = changes.dropna(subset=["abs_change"]).copy()
    end = (d["tipoff_utc"] - pd.to_timedelta(d["minutes_to_tipoff"], unit="m")).dt.tz_convert(
        EASTERN)
    d = d[end.dt.date == d["game_date"]]
    end = end[d.index]
    d["clock_minute"] = (end.dt.hour * 60 + end.dt.minute) // STEP * STEP
    out = d.groupby("clock_minute")["abs_change"].agg(
        mean_abs_change="mean", game_steps="count").reset_index()
    out["games"] = d.groupby("clock_minute")["game_id"].nunique().to_numpy()
    out.insert(1, "clock_time_et", [f"{m // 60:02d}:{m % 60:02d}" for m in out["clock_minute"]])
    out["plotted"] = (out["game_steps"] >= MIN_CLOCK_STEPS).astype(int)
    return out


def window_shares(changes: pd.DataFrame) -> pd.DataFrame:
    """Share of all movement from 24h to tipoff in each window, pooled over games.

    The 30h to 24h window is a separate last row, outside the total (share left empty).
    """
    rows = []
    for label, lo, hi in (*WINDOWS, EARLY_WINDOW):
        w = changes[(changes["minutes_to_tipoff"] >= lo) & (changes["minutes_to_tipoff"] < hi)]
        rows.append({"window": label, "minutes_from": hi, "minutes_to": lo,
                     "hours": (hi - lo) / 60, "sum_abs_change": w["abs_change"].sum(),
                     "game_steps": int(w["abs_change"].notna().sum()),
                     "game_steps_possible": len(w),
                     "mean_abs_change_per_step": w["abs_change"].mean()})
    out = pd.DataFrame(rows)
    main = out["window"] != EARLY_WINDOW[0]
    total = out.loc[main, "sum_abs_change"].sum()
    out["share_of_24h_total"] = (out["sum_abs_change"] / total).where(main)
    out["share_per_hour"] = out["share_of_24h_total"] / out["hours"]
    out["in_24h_total"] = main.astype(int)
    return out


def minutes_figure(table: pd.DataFrame, n_games: int, path: Path) -> plt.Figure:
    """Mean absolute change per step against hours to tipoff. The range before 24h, which
    is outside the main descriptive range, is drawn in gray."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    main = table[table["range"] == "24h to tipoff"]
    early = table[table["range"] != "24h to tipoff"]
    ax.plot(early["hours_to_tipoff"], early["mean_abs_change"], color=COLORS["reference"],
            lw=1.5, zorder=2, label="30h to 24h (shown apart, not in the totals)")
    ax.plot(main["hours_to_tipoff"], main["mean_abs_change"], color=COLORS["series1"], lw=2,
            zorder=3, label="24h to tipoff")
    ax.axvline(24, color=COLORS["grid"], lw=1.2, zorder=1)
    ax.set_xlim(30, 0)
    ax.set_xticks([30, 24, 18, 12, 6, 3, 1, 0])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Hours to tipoff")
    ax.set_ylabel("Mean absolute change in log odds per 5 minutes")
    ax.set_title(f"When prices move: mean 5-minute price change by time to tipoff, "
                 f"{n_games:,} games", loc="left")
    ax.legend(loc="upper center", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def clock_figure(table: pd.DataFrame, path: Path) -> plt.Figure:
    """Mean absolute change per step against US Eastern clock time on the game day."""
    shown = table[table["plotted"] == 1]
    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(shown["clock_minute"] / 60, shown["mean_abs_change"], color=COLORS["series1"],
            lw=2, zorder=3)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3), [f"{h % 24:02d}:00" for h in range(0, 25, 3)])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("US Eastern clock time on the game day (end of the 5-minute step)")
    ax.set_ylabel("Mean absolute change in log odds per 5 minutes")
    ax.set_title("When prices move: mean 5-minute price change by clock time, before tipoff",
                 loc="left")
    ax.text(0.02, 0.95, f"Clock bins with at least {MIN_CLOCK_STEPS} game-steps",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color=COLORS["text_secondary"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ================================================================ D2: event study

def step_at_or_before(minutes_before_tipoff: float) -> int:
    """The grid step (minutes to tipoff) of the last 5-minute step at or before a time that
    lies `minutes_before_tipoff` before tipoff."""
    return int(np.ceil(minutes_before_tipoff / STEP - 1e-9) * STEP)


def step_at_or_after(minutes_before_tipoff: float) -> int:
    """The grid step of the first 5-minute step at or after that time."""
    return int(np.floor(minutes_before_tipoff / STEP + 1e-9) * STEP)


def window_steps(minutes_before_tipoff: float, shift: float = 0.0) -> dict:
    """Grid steps of one event's prices, with every time moved `shift` minutes earlier.

    before        last step at or before 5 minutes before the release
    after         first step at or after 30 minutes after the release
    anticipation  last step at or before 60 minutes before the release
    """
    m = minutes_before_tipoff + shift
    return {"before": step_at_or_before(m + BEFORE_MIN),
            "after": step_at_or_after(m - AFTER_MIN),
            "anticipation": step_at_or_before(m + ANTICIPATION_MIN)}


def team_price_lookup(paths: pd.DataFrame) -> dict:
    """{game_id: Series of p_home_mid indexed by minutes_to_tipoff}."""
    return {g: x.set_index("minutes_to_tipoff")["p_home_mid"]
            for g, x in paths.groupby("game_id")}


def team_price(lookup: dict, game_id: str, side: str, step: int) -> float:
    """The affected team's normalized midpoint at a grid step: p for home, 1 - p for away.
    Missing when the step is outside the path or has no midpoint."""
    series = lookup.get(game_id)
    if series is None or step not in series.index:
        return np.nan
    p = series.loc[step]
    return p if side == "home" else 1 - p


def event_changes(events: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """Per event: the main change, the anticipation change and their placebos."""
    lookup = team_price_lookup(paths)
    rows = []
    for e in events.itertuples(index=False):
        rec = {"event_id": e.event_id}
        for prefix, shift in (("", 0.0), ("placebo_", float(PLACEBO_SHIFT_MIN))):
            s = window_steps(e.minutes_before_tipoff, shift)
            price = {k: team_price(lookup, e.game_id, e.side, v) for k, v in s.items()}
            rec |= {f"{prefix}step_before": s["before"], f"{prefix}step_after": s["after"],
                    f"{prefix}step_anticipation": s["anticipation"],
                    f"{prefix}price_before": price["before"],
                    f"{prefix}price_after": price["after"],
                    f"{prefix}price_anticipation": price["anticipation"],
                    f"{prefix}change": price["after"] - price["before"],
                    f"{prefix}anticipation_change": price["before"] - price["anticipation"]}
        rows.append(rec)
    out = events.merge(pd.DataFrame(rows), on="event_id", validate="one_to_one")
    out["change_minus_placebo"] = out["change"] - out["placebo_change"]
    out["anticipation_minus_placebo"] = (out["anticipation_change"]
                                         - out["placebo_anticipation_change"])
    return out


def study_samples(changes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The events behind each test.

    main          events with both midpoints of the main window
    anticipation  events of `main` that also have both anticipation midpoints
    Each test against the placebo uses the events of its window that have the placebo.
    """
    main = changes[changes["change"].notna()]
    return {"main": main, "anticipation": main[main["anticipation_change"].notna()]}


# test_id, sample, column, description
EVENT_TESTS = (
    ("injury_event_mean", "main", "change",
     "mean change of the affected team's price, 5 minutes before to 30 minutes after release"),
    ("injury_event_placebo_mean", "main", "placebo_change",
     "mean placebo change, the same window 3 hours earlier"),
    ("injury_event_minus_placebo", "main", "change_minus_placebo",
     "mean of event change minus placebo change"),
    ("injury_anticipation_mean", "anticipation", "anticipation_change",
     "mean change of the affected team's price, 60 to 5 minutes before release"),
    ("injury_anticipation_placebo_mean", "anticipation", "placebo_anticipation_change",
     "mean placebo anticipation change, the same window 3 hours earlier"),
    ("injury_anticipation_minus_placebo", "anticipation", "anticipation_minus_placebo",
     "mean of anticipation change minus its placebo"),
)


def bootstrap_mean(d: pd.DataFrame, column: str, n_boot: int = B,
                   seed: int = SEED) -> pd.Series:
    """Block bootstrap by game date of the mean of one column (missing values dropped)."""
    x = d.dropna(subset=[column])
    values = x[column].to_numpy(dtype=float)
    boot = stats_tools.block_bootstrap(x["game_date"], lambda idx: {"mean": values[idx].mean()},
                                       n_boot=n_boot, seed=seed)
    return boot["mean"]


def event_tests(changes: pd.DataFrame, n_boot: int = B,
                seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(results table, tests in the common Stage 10 layout)."""
    samples = study_samples(changes)
    results, rows = [], []
    for test_id, sample, column, text in EVENT_TESTS:
        x = samples[sample].dropna(subset=[column])
        boot = bootstrap_mean(x, column, n_boot=n_boot, seed=seed)
        estimate = float(x[column].mean())
        results.append({"test_id": test_id, "window": sample, "statistic": column,
                        "n_events": len(x), "n_dates": x["game_date"].nunique(),
                        "mean": estimate, "median": float(x[column].median()),
                        "sd": float(x[column].std(ddof=1)),
                        "share_negative": float((x[column] < 0).mean()),
                        "boot_lo": float(boot.quantile(0.025)),
                        "boot_hi": float(boot.quantile(0.975)),
                        "p_bootstrap": bootstrap_p_value(boot)})
        rows.append(all_tests.bootstrap_row(PART, test_id, text, "the mean is 0", len(x),
                                            estimate, boot, unit="events"))
    return pd.DataFrame(results), all_tests.table(rows)


def event_counts(counts: dict, changes: pd.DataFrame) -> pd.DataFrame:
    """The number of changes and events left at each step of the definitions."""
    samples = study_samples(changes)
    rows = list(counts.items())
    rows += [("events with both midpoints of the main window", len(samples["main"])),
             ("of those, with the placebo window", int(
                 samples["main"]["placebo_change"].notna().sum())),
             ("of those (main), with both anticipation midpoints", len(samples["anticipation"])),
             ("of those, with the placebo anticipation window", int(
                 samples["anticipation"]["placebo_anticipation_change"].notna().sum())),
             ("main-window events released at the posting time", int(
                 (samples["main"]["release_source"] == "posted").sum())),
             ("main-window events released at the scheduled time", int(
                 (samples["main"]["release_source"] == "scheduled").sum()))]
    return pd.DataFrame(rows, columns=["step", "count"])


def event_paths(changes: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """One row per main-window event and offset: the affected team's price minus its value
    5 minutes before the release. Events are lined up on their step_before, which stands
    for -5 minutes. Steps after tipoff are left out."""
    lookup = team_price_lookup(paths)
    rows = []
    for e in study_samples(changes)["main"].itertuples(index=False):
        for offset in PATH_OFFSETS:
            step = e.step_before - (int(offset) + BEFORE_MIN)
            if step < 0:
                continue
            price = team_price(lookup, e.game_id, e.side, step)
            rows.append({"event_id": e.event_id, "game_date": e.game_date,
                         "offset_min": int(offset), "relative_price": price - e.price_before})
    return pd.DataFrame(rows)


def average_path(event_path_rows: pd.DataFrame, n_boot: int = B,
                 seed: int = SEED) -> pd.DataFrame:
    """Mean relative price at each offset with a pointwise 95% bootstrap band.

    One bootstrap over game dates serves every offset: each replicate draws the dates once.
    """
    wide = event_path_rows.pivot(index="event_id", columns="offset_min", values="relative_price")
    dates = event_path_rows.drop_duplicates("event_id").set_index("event_id")["game_date"]
    dates = dates.reindex(wide.index)
    values = wide.to_numpy(dtype=float)
    offsets = list(wide.columns)

    def replicate(idx: np.ndarray) -> dict:
        with np.errstate(invalid="ignore"):
            means = np.nanmean(values[idx], axis=0) if len(idx) else np.full(len(offsets), np.nan)
        return dict(zip(offsets, means))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)    # an offset with no events
        boot = stats_tools.block_bootstrap(dates.to_numpy(), replicate, n_boot=n_boot, seed=seed)
        mean = np.nanmean(values, axis=0)
    return pd.DataFrame({"offset_min": offsets, "mean_relative_price": mean,
                         "n_events": np.sum(~np.isnan(values), axis=0),
                         "boot_lo": [boot[o].quantile(0.025) for o in offsets],
                         "boot_hi": [boot[o].quantile(0.975) for o in offsets]})


def path_figure(path_table: pd.DataFrame, n_events: int, path: Path) -> plt.Figure:
    """The average path of the affected team's price around the release."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = path_table["offset_min"]
    ax.fill_between(x, path_table["boot_lo"] * 100, path_table["boot_hi"] * 100,
                    color=COLORS["series1"], alpha=0.18, lw=0, zorder=2,
                    label="95% bootstrap band")
    ax.plot(x, path_table["mean_relative_price"] * 100, color=COLORS["series1"], lw=2,
            marker="o", ms=4, mec=COLORS["surface"], mew=0.8, zorder=3,
            label="Mean price of the affected team")
    ax.axhline(0, color=COLORS["reference"], lw=1.0, zorder=1)
    ax.axvline(0, color=COLORS["reference"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("report released", (0, 1), xycoords=("data", "axes fraction"),
                xytext=(5, -4), textcoords="offset points", va="top", fontsize=8.5,
                color=COLORS["text_secondary"])
    ax.set_xticks(np.arange(-60, 121, 30))
    ax.set_xlabel("Minutes from the release of the injury report")
    ax.set_ylabel("Price minus its value 5 minutes before\nthe release (percentage points)")
    ax.set_title(f"Price of a team after a key player is ruled Out: {n_events:,} events",
                 loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig

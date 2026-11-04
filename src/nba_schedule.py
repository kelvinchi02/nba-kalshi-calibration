"""Stage 3: 2025-26 NBA regular season schedule and results from stats.nba.com.

Source: nba_api ScheduleLeagueV2 (official stats.nba.com endpoint). No unofficial fallback:
if stats.nba.com cannot be reached after retries, the script stops with an error.

Output: data/processed/nba_games_2025_26.parquet, one row per regular-season game
(NBA game ID prefix 002). The NBA Cup final (prefix 006) does not count toward the regular
season and is excluded. Postponed games appear once, at their replayed date and tipoff.

Columns:
  game_id        NBA game ID, e.g. 0022500529
  game_date      US Eastern game date
  tipoff_utc     official scheduled tipoff, UTC (tz-aware)
  away_team      NBA tricode, e.g. MIA
  home_team      NBA tricode, e.g. CHI
  away_score, home_score   final scores
  home_win       1 if the home team won, else 0
  status         Final, Final/OT, Final/OT2, ...
  game_label     e.g. "Emirates NBA Cup", "NBA Berlin Game" (empty for ordinary games)
  game_subtype   e.g. "in-season", "Global Games" (empty for ordinary games)
  arena_city

Usage:
    python src/nba_schedule.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

SEASON = "2025-26"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "nba_schedule_2025_26.json"
OUT_PATH = REPO_ROOT / "data" / "processed" / "nba_games_2025_26.parquet"
EXPECTED_GAMES = 1230

# Headers: stats.nba.com silently drops requests that do not look like a current browser
# on nba.com. nba_api ships a maintained browser-like header set (recent Chrome User-Agent,
# Referer, Sec-Ch-Ua / Sec-Fetch fields), so we use it (headers=None) instead of a
# hand-written set; a hand-written set without the Sec-* fields timed out in testing.
HEADERS = None

log = logging.getLogger("nba_schedule")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60),
       before_sleep=before_sleep_log(log, logging.WARNING), reraise=True)
def fetch_schedule(season: str = SEASON) -> scheduleleaguev2.ScheduleLeagueV2:
    """Fetch ScheduleLeagueV2 from stats.nba.com, retrying with backoff."""
    return scheduleleaguev2.ScheduleLeagueV2(season=season, headers=HEADERS, timeout=60)


def build_regular_season(sched: pd.DataFrame) -> pd.DataFrame:
    reg = sched[sched["gameId"].str[:3] == "002"].copy()
    games = pd.DataFrame({
        "game_id": reg["gameId"],
        "game_date": pd.to_datetime(reg["gameDateEst"].str[:10]).dt.date,
        "tipoff_utc": pd.to_datetime(reg["gameDateTimeUTC"], utc=True),
        "away_team": reg["awayTeam_teamTricode"],
        "home_team": reg["homeTeam_teamTricode"],
        "away_score": reg["awayTeam_score"].astype("Int64"),
        "home_score": reg["homeTeam_score"].astype("Int64"),
        "status": reg["gameStatusText"].str.strip(),
        "game_label": reg["gameLabel"].fillna("").str.strip(),
        "game_subtype": reg["gameSubtype"].fillna("").str.strip(),
        "arena_city": reg["arenaCity"],
    })
    games["home_win"] = (games["home_score"] > games["away_score"]).astype("Int64")
    # Unfinished games get no outcome rather than a false 0.
    games.loc[~games["status"].str.startswith("Final"), "home_win"] = pd.NA
    cols = ["game_id", "game_date", "tipoff_utc", "away_team", "home_team",
            "away_score", "home_score", "home_win", "status", "game_label",
            "game_subtype", "arena_city"]
    return games[cols].sort_values(["tipoff_utc", "game_id"]).reset_index(drop=True)


def check(games: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty if all checks pass)."""
    problems = []
    if len(games) != EXPECTED_GAMES:
        problems.append(f"row count {len(games)} != {EXPECTED_GAMES}")
    if games["game_id"].duplicated().any():
        problems.append("duplicate game IDs")
    if games["tipoff_utc"].isna().any():
        problems.append(f"{games['tipoff_utc'].isna().sum()} games without a tipoff time")
    not_final = (~games["status"].str.startswith("Final")).sum()
    if not_final:
        problems.append(f"{not_final} games not final")
    ties = (games["home_score"] == games["away_score"]).sum()
    if ties:
        problems.append(f"{ties} games with equal scores")
    per_team = pd.concat([games["home_team"], games["away_team"]]).value_counts()
    if per_team.nunique() != 1 or per_team.iloc[0] != 82 or len(per_team) != 30:
        problems.append(f"games per team not 30 x 82: {per_team.to_dict()}")
    # Tipoff in UTC should fall on the Eastern game date (UTC-6h is a safe ET-date proxy).
    off = (games["tipoff_utc"] - pd.Timedelta(hours=6)).dt.date != games["game_date"]
    if off.any():
        problems.append(f"{off.sum()} games whose UTC tipoff does not match the game date")
    return problems


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        endpoint = fetch_schedule()
    except Exception as exc:  # noqa: BLE001 - report and stop, no silent fallback
        raise SystemExit(
            f"stats.nba.com request failed after retries: {exc!r}\n"
            "Not switching to an unofficial source. Try again later or from another network."
        ) from exc

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(endpoint.get_dict()))

    sched = endpoint.get_data_frames()[0]  # full schedule, all game types
    print("All games by NBA game ID prefix (001 pre, 002 regular, 003 all-star, "
          "004 playoffs, 005 play-in, 006 Cup final):")
    print(sched["gameId"].str[:3].value_counts().sort_index().to_string())

    games = build_regular_season(sched)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(games)} regular-season games to {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Date range: {games['game_date'].min()} to {games['game_date'].max()}")
    print(f"Home win rate: {games['home_win'].mean():.3f}")
    print(f"Overtime games: {(games['status'] != 'Final').sum()}")
    print(f"Labelled games: {games['game_label'].replace('', pd.NA).value_counts().to_dict()}")

    problems = check(games)
    if problems:
        print("\nCHECKS FAILED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nAll checks passed (1,230 games, 30 teams x 82, all final, no ties, "
          "tipoffs consistent with game dates).")


if __name__ == "__main__":
    main()

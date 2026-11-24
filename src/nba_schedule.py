"""Stage 3: NBA regular season schedule and results from stats.nba.com (default 2025-26).

Source: nba_api ScheduleLeagueV2 (official stats.nba.com endpoint). No unofficial fallback:
if stats.nba.com cannot be reached after retries, the script stops with an error.

Output: data/processed/nba_games_<season>.csv (nba_games_2025_26.csv by default), one row
per regular-season game (NBA game ID prefix 002). The NBA Cup final (prefix 006) does not
count toward the regular season and is excluded. Postponed games appear once, at their
replayed date and tipoff.

A season still being played needs --through: the table then holds the games up to that US
Eastern date, and every one of them must be final.

Columns:
  game_id        NBA game ID, e.g. 0022500529
  game_date      US Eastern game date
  tipoff_utc     official scheduled tipoff, UTC (ISO 8601 with Z in the CSV)
  away_team      NBA tricode, e.g. MIA
  home_team      NBA tricode, e.g. CHI
  away_score, home_score   final scores
  home_win       1 if the home team won, else 0
  status         Final, Final/OT, Final/OT2, ...
  game_label     e.g. "Emirates NBA Cup", "NBA Berlin Game" (empty for ordinary games)
  game_subtype   e.g. "in-season", "Global Games" (empty for ordinary games)
  arena_city

The raw stats.nba.com response is cached in data/raw/nba_schedule_<season>.json, so the
table can be rebuilt offline with --cached.

Usage:
    python src/nba_schedule.py            # fetch from stats.nba.com
    python src/nba_schedule.py --cached   # rebuild from the cached raw response
    python src/nba_schedule.py --season 2026-27 --through 2026-11-22
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2
from nba_api.stats.library.http import NBAStatsResponse
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from csv_io import write_csv
from seasons import DEFAULT_SEASON, Season, add_season_argument, get_season

SEASON = DEFAULT_SEASON
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = get_season().nba_raw_path
OUT_PATH = get_season().nba_games_path
EXPECTED_GAMES = get_season().expected_games

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


def load_cached_schedule(season: str = SEASON,
                         raw_path: Path = RAW_PATH) -> scheduleleaguev2.ScheduleLeagueV2:
    """Rebuild the endpoint object from the cached raw response, without a request."""
    endpoint = scheduleleaguev2.ScheduleLeagueV2(season=season, get_request=False)
    endpoint.nba_response = NBAStatsResponse(response=raw_path.read_text(encoding="utf-8"),
                                             status_code=200, url=str(raw_path))
    endpoint.load_response()
    return endpoint


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


def played_through(games: pd.DataFrame, through: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For a season in progress: (games up to `through` that are final, games up to
    `through` that are not final). Later games are left out of both."""
    upto = games[games["game_date"] <= through]
    final = upto["status"].str.startswith("Final")
    return upto[final].reset_index(drop=True), upto[~final].reset_index(drop=True)


def check(games: pd.DataFrame, expected_games: int | None = EXPECTED_GAMES) -> list[str]:
    """Return a list of problems (empty if all checks pass).

    expected_games None means a season in progress: the game count and the 30 x 82 check
    do not apply.
    """
    problems = []
    if expected_games is not None and len(games) != expected_games:
        problems.append(f"row count {len(games)} != {expected_games}")
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
    if expected_games is not None and (per_team.nunique() != 1 or per_team.iloc[0] != 82
                                       or len(per_team) != 30):
        problems.append(f"games per team not 30 x 82: {per_team.to_dict()}")
    if expected_games is None and len(per_team) != 30:
        problems.append(f"{len(per_team)} teams have played, expected 30")
    # Tipoff in UTC should fall on the Eastern game date (UTC-6h is a safe ET-date proxy).
    off = (games["tipoff_utc"] - pd.Timedelta(hours=6)).dt.date != games["game_date"]
    if off.any():
        problems.append(f"{off.sum()} games whose UTC tipoff does not match the game date")
    return problems


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true",
                        help="rebuild from the cached raw response instead of fetching")
    parser.add_argument("--through", type=date.fromisoformat,
                        help="last US Eastern game date to keep (season in progress)")
    add_season_argument(parser)
    args = parser.parse_args()
    season: Season = get_season(args.season)
    raw_path, out_path = season.nba_raw_path, season.nba_games_path
    if season.complete == (args.through is not None):
        raise SystemExit("--through is for a season in progress, and is required for one")

    if args.cached:
        endpoint = load_cached_schedule(season.label, raw_path)
        print(f"Using cached response {raw_path.relative_to(REPO_ROOT)}")
    else:
        try:
            endpoint = fetch_schedule(season.label)
        except Exception as exc:  # noqa: BLE001 - report and stop, no silent fallback
            raise SystemExit(
                f"stats.nba.com request failed after retries: {exc!r}\n"
                "Not switching to an unofficial source. Try again later, from another "
                "network, or rebuild from the cached response with --cached."
            ) from exc
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(endpoint.get_dict()), encoding="utf-8")

    sched = endpoint.get_data_frames()[0]  # full schedule, all game types
    print("All games by NBA game ID prefix (001 pre, 002 regular, 003 all-star, "
          "004 playoffs, 005 play-in, 006 Cup final):")
    print(sched["gameId"].str[:3].value_counts().sort_index().to_string())

    games = build_regular_season(sched)
    if not season.complete:
        games, not_final = played_through(games, args.through)
        print(f"\nSeason in progress: keeping the final games up to {args.through}.")
        if len(not_final):
            print(f"NOT FINAL on or before {args.through} (left out):")
            print(not_final[["game_id", "game_date", "away_team", "home_team",
                             "status"]].to_string(index=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(games, out_path)

    print(f"\nSaved {len(games)} regular-season games to {out_path.relative_to(REPO_ROOT)}")
    print(f"Date range: {games['game_date'].min()} to {games['game_date'].max()}")
    print(f"Home win rate: {games['home_win'].mean():.3f}")
    print(f"Overtime games: {(games['status'] != 'Final').sum()}")
    print(f"Labelled games: {games['game_label'].replace('', pd.NA).value_counts().to_dict()}")

    problems = check(games, season.expected_games)
    if problems:
        print("\nCHECKS FAILED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    if season.complete:
        print("\nAll checks passed (1,230 games, 30 teams x 82, all final, no ties, "
              "tipoffs consistent with game dates).")
    else:
        print(f"\nAll checks passed ({len(games):,} games through {args.through}, all final, "
              "no ties, 30 teams, tipoffs consistent with game dates).")


if __name__ == "__main__":
    main()

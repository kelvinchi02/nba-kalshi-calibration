"""Stage 10: minutes played per player per game, 2025-26 regular season (plan 10.4, D2).

Source: nba_api LeagueGameLog in player mode (official stats.nba.com endpoint). One request
returns every player's line in every regular-season game. No unofficial fallback: if
stats.nba.com cannot be reached after retries, the script stops with an error.

Output: data/processed/player_minutes_2025_26.csv, one row per player per game played:
  game_id, game_date (US Eastern), team (NBA tricode), player_id, player_name, minutes

The raw response is cached in data/raw/player_gamelog_2025_26.json, so the table can be
rebuilt offline with --cached.

Usage:
    python src/player_minutes.py            # fetch from stats.nba.com
    python src/player_minutes.py --cached   # rebuild from the cached raw response
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from csv_io import write_csv

SEASON = "2025-26"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "player_gamelog_2025_26.json"
OUT_PATH = REPO_ROOT / "data" / "processed" / "player_minutes_2025_26.csv"

log = logging.getLogger("player_minutes")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60),
       before_sleep=before_sleep_log(log, logging.WARNING), reraise=True)
def fetch(season: str = SEASON) -> dict:
    """The raw LeagueGameLog response for players, regular season."""
    endpoint = leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star="Regular Season",
        player_or_team_abbreviation="P", timeout=60)
    return endpoint.get_dict()


def raw_frame(raw: dict) -> pd.DataFrame:
    result = raw["resultSets"][0]
    return pd.DataFrame(result["rowSet"], columns=result["headers"])


def build(log_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per player per game in which he played (minutes above 0)."""
    out = pd.DataFrame({
        "game_id": log_rows["GAME_ID"].astype(str),
        "game_date": pd.to_datetime(log_rows["GAME_DATE"]).dt.date,
        "team": log_rows["TEAM_ABBREVIATION"],
        "player_id": log_rows["PLAYER_ID"].astype(int),
        "player_name": log_rows["PLAYER_NAME"],
        "minutes": pd.to_numeric(log_rows["MIN"], errors="coerce"),
    })
    out = out[out["minutes"] > 0]
    if out.duplicated(["game_id", "player_id"]).any():
        raise ValueError("a player appears twice in one game")
    return out.sort_values(["game_id", "team", "player_id"], ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true",
                        help="rebuild from the cached raw response instead of fetching")
    args = parser.parse_args()
    if args.cached:
        raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    else:
        try:
            raw = fetch()
        except Exception as exc:  # noqa: BLE001 - report and stop, no silent fallback
            raise SystemExit(f"stats.nba.com request failed after retries: {exc!r}\n"
                             "Not switching to another source.") from exc
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text(json.dumps(raw), encoding="utf-8")
    table = build(raw_frame(raw))
    write_csv(table, OUT_PATH)
    print(f"Saved {len(table):,} player-game rows to {OUT_PATH.relative_to(REPO_ROOT)}: "
          f"{table['game_id'].nunique():,} games, {table['player_id'].nunique()} players, "
          f"{table['team'].nunique()} teams")


if __name__ == "__main__":
    main()

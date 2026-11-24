"""Season settings and file names for the data pipeline.

The pipeline scripts (nba_schedule.py, match_games.py, pull_kalshi.py, build_snapshots.py,
quality_report.py) take --season. Everything that differs between seasons is here, so no
script is copied per season.

2025-26 is the default and keeps the file names it had before a second season existed
(game_map.csv, snapshots.csv, ...). Other seasons carry the season in the file name
(game_map_2026_27.csv, snapshots_2026_27.csv, ...).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
RAW = REPO_ROOT / "data" / "raw"
SERIES = "KXNBAGAME"
DEFAULT_SEASON = "2025-26"


@dataclass(frozen=True)
class Season:
    label: str                       # NBA season label, e.g. "2025-26"
    kalshi_start: date               # Kalshi events dated from here on belong to the season
    kalshi_end: date | None          # last Kalshi event date; None = last game date in the
                                     # NBA games file (a season still being played)
    regular_start: date | None       # first and last regular-season date and first playoff
    regular_end: date | None         # date, for the labels in the match report; None = taken
    playoffs_start: date | None      # from the NBA games file
    expected_games: int | None       # 1230 for a complete season, None while in progress
    source: str                      # Kalshi endpoints holding the data: "historical" or "live"
    legacy_names: bool = False       # True: file names without the season

    @property
    def suffix(self) -> str:
        return self.label.replace("-", "_")

    @property
    def complete(self) -> bool:
        return self.expected_games is not None

    def _processed(self, stem: str, ext: str) -> Path:
        name = f"{stem}.{ext}" if self.legacy_names else f"{stem}_{self.suffix}.{ext}"
        return PROCESSED / name

    @property
    def nba_games_path(self) -> Path:
        return PROCESSED / f"nba_games_{self.suffix}.csv"

    @property
    def nba_raw_path(self) -> Path:
        return RAW / f"nba_schedule_{self.suffix}.json"

    @property
    def game_map_path(self) -> Path:
        return self._processed("game_map", "csv")

    @property
    def snapshots_path(self) -> Path:
        return self._processed("snapshots", "csv")

    @property
    def match_report_path(self) -> Path:
        return self._processed("match_report", "md")

    @property
    def quality_report_path(self) -> Path:
        return self._processed("quality_report", "md")

    @property
    def markets_cache_path(self) -> Path:
        stem = f"kalshi_markets_{SERIES}"
        return RAW / (f"{stem}.json" if self.legacy_names else f"{stem}_{self.suffix}.json")


SEASONS = {
    "2025-26": Season(
        label="2025-26", kalshi_start=date(2025, 10, 1), kalshi_end=date(2026, 6, 30),
        regular_start=date(2025, 10, 21), regular_end=date(2026, 4, 12),
        playoffs_start=date(2026, 4, 18), expected_games=1230, source="historical",
        legacy_names=True),
    # In progress when it was added (plan section 10.6). The games file holds the final
    # games up to the cutoff date given to nba_schedule.py, and that date ends the season
    # for every later step.
    "2026-27": Season(
        label="2026-27", kalshi_start=date(2026, 10, 1), kalshi_end=None,
        regular_start=None, regular_end=None, playoffs_start=None, expected_games=None,
        source="live"),
}


def get_season(label: str = DEFAULT_SEASON) -> Season:
    if label not in SEASONS:
        raise SystemExit(f"Unknown season {label!r}. Known: {', '.join(SEASONS)}")
    return SEASONS[label]


def add_season_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--season", default=DEFAULT_SEASON, choices=sorted(SEASONS),
                        help=f"NBA season (default {DEFAULT_SEASON})")

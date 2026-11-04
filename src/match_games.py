"""Stage 5: map Kalshi KXNBAGAME events to NBA 2025-26 regular-season game IDs.

Matching key: game date + away team + home team, using an explicit Kalshi -> NBA team
code table. Kalshi event tickers look like KXNBAGAME-26JAN15MILSAS (date, AWAY, HOME).

Rules (agreed with the project owner):
  * Kalshi events settled 'scalar' are postponed originals: dropped, listed in the report.
    Their replayed games have their own Kalshi events and are kept.
  * If no exact date match exists, the same away/home pairing within +/- 1 day is accepted
    and flagged postponed_window (game moved after the market opened). Excluded from the
    main sample.
  * tipoff_moved flags games whose official NBA tipoff differs from the time Kalshi listed
    (expiration_time - 14 days) by 1 to 12 hours: same-day time changes. Kept in the main
    sample, to be checked separately.
  * Nothing is dropped silently: every unmatched game on either side is in the report.

Inputs:
  data/processed/nba_games_2025_26.parquet          (Stage 3)
  Kalshi /historical/markets for KXNBAGAME          (cached to data/raw/kalshi_markets_KXNBAGAME.json)
Outputs:
  data/processed/team_code_map.csv
  data/processed/game_map.parquet     one row per NBA regular-season game
  data/processed/match_report.md

Usage:
    python src/match_games.py [--refresh]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from kalshi_client import KalshiClient

SERIES = "KXNBAGAME"
SEASON_START, SEASON_END = date(2025, 10, 1), date(2026, 6, 30)
REPO_ROOT = Path(__file__).resolve().parents[1]
NBA_PATH = REPO_ROOT / "data" / "processed" / "nba_games_2025_26.parquet"
MARKETS_CACHE = REPO_ROOT / "data" / "raw" / f"kalshi_markets_{SERIES}.json"
TEAM_MAP_PATH = REPO_ROOT / "data" / "processed" / "team_code_map.csv"
GAME_MAP_PATH = REPO_ROOT / "data" / "processed" / "game_map.parquet"
REPORT_PATH = REPO_ROOT / "data" / "processed" / "match_report.md"

# Kalshi team code -> (NBA tricode, names Kalshi uses in yes_sub_title).
# For 2025-26 the codes happen to be identical; the table stays explicit so a future
# difference (e.g. GS vs GSW) is a one-line change and is validated below.
TEAM_MAP = {
    "ATL": ("ATL", ["Atlanta"]),
    "BKN": ("BKN", ["Brooklyn"]),
    "BOS": ("BOS", ["Boston"]),
    "CHA": ("CHA", ["Charlotte"]),
    "CHI": ("CHI", ["Chicago"]),
    "CLE": ("CLE", ["Cleveland"]),
    "DAL": ("DAL", ["Dallas"]),
    "DEN": ("DEN", ["Denver"]),
    "DET": ("DET", ["Detroit"]),
    "GSW": ("GSW", ["Golden State"]),
    "HOU": ("HOU", ["Houston"]),
    "IND": ("IND", ["Indiana"]),
    "LAC": ("LAC", ["Los Angeles C", "LA"]),
    "LAL": ("LAL", ["Los Angeles L", "Los Angeles"]),
    "MEM": ("MEM", ["Memphis"]),
    "MIA": ("MIA", ["Miami"]),
    "MIL": ("MIL", ["Milwaukee"]),
    "MIN": ("MIN", ["Minnesota"]),
    "NOP": ("NOP", ["New Orleans"]),
    "NYK": ("NYK", ["New York", "New York K"]),
    "OKC": ("OKC", ["Oklahoma City"]),
    "ORL": ("ORL", ["Orlando"]),
    "PHI": ("PHI", ["Philadelphia"]),
    "PHX": ("PHX", ["Phoenix"]),
    "POR": ("POR", ["Portland"]),
    "SAC": ("SAC", ["Sacramento"]),
    "SAS": ("SAS", ["San Antonio"]),
    "TOR": ("TOR", ["Toronto"]),
    "UTA": ("UTA", ["Utah"]),
    "WAS": ("WAS", ["Washington"]),
}
TIPOFF_MOVED_MIN_H, TIPOFF_MOVED_MAX_H = 1, 12


# ---------------------------------------------------------------- Kalshi side

def load_markets(refresh: bool) -> list[dict]:
    if MARKETS_CACHE.exists() and not refresh:
        return json.loads(MARKETS_CACHE.read_text())
    markets = KalshiClient().historical_markets(SERIES)
    MARKETS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    MARKETS_CACHE.write_text(json.dumps(markets))
    return markets


def parse_event_ticker(event_ticker: str) -> tuple[date, str, str]:
    """KXNBAGAME-26JAN15MILSAS -> (2026-01-15, 'MIL', 'SAS')."""
    m = re.fullmatch(rf"{SERIES}-(\d\d[A-Z]{{3}}\d\d)([A-Z]{{3}})([A-Z]{{3}})", event_ticker)
    if not m:
        raise ValueError(f"unexpected event ticker {event_ticker}")
    return datetime.strptime(m.group(1), "%y%b%d").date(), m.group(2), m.group(3)


def kalshi_events(markets: list[dict]) -> pd.DataFrame:
    """One row per 2025-26 Kalshi event with its two team markets."""
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        by_event.setdefault(m["event_ticker"], []).append(m)

    rows = []
    for et, ms in by_event.items():
        gdate, away, home = parse_event_ticker(et)
        if not SEASON_START <= gdate <= SEASON_END:
            continue
        by_code = {m["ticker"].rsplit("-", 1)[1]: m for m in ms}
        if set(by_code) != {away, home}:
            raise ValueError(f"{et}: market codes {sorted(by_code)} != ticker teams {away},{home}")
        h, a = by_code[home], by_code[away]
        rows.append({
            "event_ticker": et,
            "kalshi_game_date": gdate,
            "kalshi_away": away,
            "kalshi_home": home,
            "home_market": h["ticker"],
            "away_market": a["ticker"],
            "home_yes_name": h["yes_sub_title"],
            "away_yes_name": a["yes_sub_title"],
            "home_result": h["result"],
            "away_result": a["result"],
            "kalshi_tipoff_estimate": pd.Timestamp(h["expiration_time"]) - pd.Timedelta(days=14),
            "market_open_time": min(pd.Timestamp(h["open_time"]), pd.Timestamp(a["open_time"])),
            "kalshi_volume": float(h["volume_fp"]) + float(a["volume_fp"]),
        })
    return pd.DataFrame(rows)


def check_team_names(ev: pd.DataFrame) -> list[str]:
    """Every NBA code must be in TEAM_MAP and its display name must be a known alias."""
    problems = []
    for side in ["home", "away"]:
        for code, name in ev[[f"kalshi_{side}", f"{side}_yes_name"]].drop_duplicates().itertuples(
                index=False):
            if code in TEAM_MAP and name not in TEAM_MAP[code][1]:
                problems.append(f"{code}: unexpected Kalshi name {name!r}")
    return problems


# ---------------------------------------------------------------- matching

def match(nba: pd.DataFrame, ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = ev.copy()
    known = ev["kalshi_home"].isin(TEAM_MAP) & ev["kalshi_away"].isin(TEAM_MAP)
    ev["away_team"] = ev["kalshi_away"].map(lambda c: TEAM_MAP.get(c, (None,))[0])
    ev["home_team"] = ev["kalshi_home"].map(lambda c: TEAM_MAP.get(c, (None,))[0])
    postponed = ev["home_result"].eq("scalar") | ev["away_result"].eq("scalar")
    ev["status"] = None
    ev.loc[~known, "status"] = "non_nba_team"
    ev.loc[known & postponed, "status"] = "dropped_postponed_original"
    candidates = ev[ev["status"].isna()]

    # Pass 1: exact date + teams.
    exact = nba.merge(candidates, left_on=["game_date", "away_team", "home_team"],
                      right_on=["kalshi_game_date", "away_team", "home_team"], how="inner")
    exact["match_type"] = "exact"

    # Pass 2: same teams within +/- 1 day, for games and events still unmatched.
    nba_left = nba[~nba["game_id"].isin(exact["game_id"])]
    ev_left = candidates[~candidates["event_ticker"].isin(exact["event_ticker"])]
    near = nba_left.merge(ev_left, on=["away_team", "home_team"], how="inner")
    near = near[(pd.to_datetime(near["game_date"]) - pd.to_datetime(near["kalshi_game_date"]))
                .abs() <= pd.Timedelta(days=1)]
    if near["game_id"].duplicated().any() or near["event_ticker"].duplicated().any():
        raise ValueError(f"ambiguous +/-1 day matches:\n{near[['game_id', 'event_ticker']]}")
    near["match_type"] = "date_shift_1d"

    matched = pd.concat([exact, near], ignore_index=True)
    if matched["game_id"].duplicated().any() or matched["event_ticker"].duplicated().any():
        raise ValueError("a game or event matched twice")
    ev.loc[ev["event_ticker"].isin(matched["event_ticker"]), "status"] = "matched"
    ev.loc[ev["status"].isna(), "status"] = "unmatched"

    game_map = nba.merge(matched[["game_id", "event_ticker", "kalshi_game_date", "home_market",
                                  "away_market", "home_result", "kalshi_tipoff_estimate",
                                  "market_open_time", "kalshi_volume", "match_type"]],
                         on="game_id", how="left")
    shift_h = (game_map["tipoff_utc"] - game_map["kalshi_tipoff_estimate"]).dt.total_seconds() / 3600
    game_map["tipoff_shift_hours"] = shift_h
    game_map["postponed_window"] = game_map["match_type"].eq("date_shift_1d")
    game_map["tipoff_moved"] = (shift_h.abs() >= TIPOFF_MOVED_MIN_H) & \
                               (shift_h.abs() < TIPOFF_MOVED_MAX_H) & ~game_map["postponed_window"]
    game_map["hours_open_before_tipoff"] = (
        (game_map["tipoff_utc"] - game_map["market_open_time"]).dt.total_seconds() / 3600)
    game_map["matched"] = game_map["event_ticker"].notna()
    # Sanity check on orientation: Kalshi's home market settles yes iff the NBA home team won.
    game_map["kalshi_home_win"] = game_map["home_result"].map({"yes": 1, "no": 0}).astype("Int64")
    game_map = game_map.drop(columns="home_result")
    game_map["main_sample"] = game_map["matched"] & ~game_map["postponed_window"]
    return game_map, ev


def season_phase(d: date) -> str:
    """Provisional label for unmatched Kalshi events, by US game date."""
    if d < date(2025, 10, 21):
        return "preseason"
    if d <= date(2026, 4, 12):
        return "regular_season"
    if d < date(2026, 4, 18):
        return "play_in"
    return "playoffs"


# ---------------------------------------------------------------- report

def write_report(game_map: pd.DataFrame, ev: pd.DataFrame, name_problems: list[str]) -> str:
    lines = ["# Kalshi to NBA match report (2025-26 regular season)", ""]
    n = len(game_map)
    lines += [
        f"- NBA regular-season games: {n}",
        f"- Matched to a Kalshi event: {game_map['matched'].sum()} "
        f"(exact {game_map['match_type'].eq('exact').sum()}, "
        f"+/-1 day {game_map['match_type'].eq('date_shift_1d').sum()})",
        f"- Main sample (matched, not postponed_window): {game_map['main_sample'].sum()}",
        f"- tipoff_moved (official tipoff 1-12h from Kalshi's listed time): "
        f"{game_map['tipoff_moved'].sum()}",
        f"- Kalshi settlement agrees with NBA home_win: "
        f"{(game_map['kalshi_home_win'] == game_map['home_win']).sum()} of {game_map['matched'].sum()}",
        f"- Kalshi 2025-26 events: {len(ev)}; status counts: "
        + ", ".join(f"{k} {v}" for k, v in ev["status"].value_counts().items()),
        "",
    ]

    def table(df: pd.DataFrame, cols: list[str]) -> list[str]:
        if df.empty:
            return ["(none)", ""]
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in df[cols].astype(str).itertuples(index=False):
            out.append("| " + " | ".join(row) + " |")
        return out + [""]

    lines += ["## NBA games with no Kalshi event", ""]
    lines += table(game_map[~game_map["matched"]],
                   ["game_id", "game_date", "away_team", "home_team", "tipoff_utc"])

    lines += ["## Flagged: postponed_window (excluded from main sample)", ""]
    lines += table(game_map[game_map["postponed_window"]],
                   ["game_id", "event_ticker", "kalshi_game_date", "game_date", "tipoff_utc",
                    "kalshi_tipoff_estimate"])

    lines += ["## Flagged: tipoff_moved (kept in main sample)", ""]
    lines += table(game_map[game_map["tipoff_moved"]],
                   ["game_id", "event_ticker", "tipoff_utc", "kalshi_tipoff_estimate",
                    "tipoff_shift_hours"])

    lines += ["## Kalshi events dropped: postponed originals (settled scalar)", ""]
    lines += table(ev[ev["status"] == "dropped_postponed_original"],
                   ["event_ticker", "home_result", "away_result"])

    unmatched = ev[ev["status"].isin(["unmatched", "non_nba_team"])].copy()
    unmatched["phase_by_date"] = unmatched["kalshi_game_date"].map(season_phase)
    lines += ["## Kalshi events not matched to a regular-season game", ""]
    lines += ["Counts by phase (from game date): "
              + ", ".join(f"{k} {v}" for k, v in
                          unmatched["phase_by_date"].value_counts().items()), ""]
    reg_unmatched = unmatched[unmatched["phase_by_date"] == "regular_season"]
    lines += ["Unmatched events dated inside the regular season:", ""]
    lines += table(reg_unmatched, ["event_ticker", "status", "kalshi_away", "kalshi_home"])
    lines += ["All other unmatched events are preseason, play-in or playoff games, which are "
              "outside the regular-season sample.", ""]

    lines += ["## Team code checks", ""]
    lines += ([f"- {p}" for p in name_problems] if name_problems
              else ["All Kalshi team names match the aliases in team_code_map.csv."])
    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download the Kalshi market list")
    args = parser.parse_args()

    nba = pd.read_parquet(NBA_PATH)
    ev = kalshi_events(load_markets(args.refresh))

    pd.DataFrame([{"kalshi_code": k, "nba_tricode": v[0], "kalshi_names": "|".join(v[1])}
                  for k, v in TEAM_MAP.items()]).to_csv(TEAM_MAP_PATH, index=False)
    unknown_nba = set(TEAM_MAP[k][0] for k in TEAM_MAP) - set(nba["home_team"])
    if unknown_nba:
        raise ValueError(f"TEAM_MAP values not in NBA schedule: {unknown_nba}")

    game_map, ev = match(nba, ev)
    game_map.to_parquet(GAME_MAP_PATH, index=False)
    print(write_report(game_map, ev, check_team_names(ev)))
    print(f"Saved {GAME_MAP_PATH.relative_to(REPO_ROOT)}, {TEAM_MAP_PATH.relative_to(REPO_ROOT)}, "
          f"{REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

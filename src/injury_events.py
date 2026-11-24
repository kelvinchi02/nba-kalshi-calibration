"""Stage 10: injury report events (ANALYSIS_PLAN.md section 10.4, part D2).

Two layers.

1. Run as a script, this reads every parsed injury report row (data/raw/injury_reports/,
   written by envs/injuries/fetch_injury_reports.py) and writes
   data/processed/injury_status_changes_2025_26.csv: one row each time a player's status
   for a game differs between two consecutive parsed snapshots.

2. As a library for notebooks/07_information_arrival.ipynb, it turns those changes into
   events, with the definitions fixed in the plan:

   status change to Out   the player is Out in a snapshot and was not Out in the snapshot
                          before it: listed with another status, or not listed while his
                          team had already submitted its report for that game. Only the
                          first snapshot in which a player is Out for a game can count.
   key player             top 3 on his team in minutes per game played for that team, over
                          the team's games before that game, with at least 5 such games
   event                  one team in one game at one snapshot with at least one key player
                          changing to Out, released at least 30 minutes before tipoff. If
                          both teams of a game have one in the same snapshot, both are
                          dropped.
   release time           the file's posting time (the server's Last-Modified time) when it
                          lies from 30 minutes before to 60 minutes after the scheduled
                          report time, otherwise the scheduled time. The scheduled time is
                          the report time printed in the file's header, or the time in the
                          file name when the header cannot be read.

Usage:
    python src/injury_events.py                     # write the status changes
    python src/injury_events.py --sample out.csv    # draw 20 events for the hand check
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from csv_io import read_csv, write_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
ROWS_PATH = REPO_ROOT / "data" / "raw" / "injury_reports" / "injury_report_rows_2025_26.csv.gz"
SNAPSHOTS_PATH = PROCESSED / "injury_report_snapshots_2025_26.csv"
CHANGES_PATH = PROCESSED / "injury_status_changes_2025_26.csv"
MINUTES_PATH = PROCESSED / "player_minutes_2025_26.csv"
CHECK_PATH = PROCESSED / "injury_event_check_2025_26.csv"
GAME_MAP_PATH = PROCESSED / "game_map.csv"

EASTERN = ZoneInfo("America/New_York")
OUT = "Out"
NOT_SUBMITTED = "NOT YET SUBMITTED"
KEY_RANK = 3                    # top 3 in minutes per game
MIN_PRIOR_GAMES = 5             # games played for the team before the game
MIN_LEAD_MINUTES = 30           # release at least this long before tipoff
POSTED_EARLY_MIN, POSTED_LATE_MIN = 30, 60   # when the server time counts as the release
CHECK_EVENTS = 20               # events checked by hand against the report files
CHECK_SEED = 4981
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# Names the injury report uses that differ from the NBA's team list.
TEAM_ALIASES = {"LA Clippers": "LAC", "Los Angeles Clippers": "LAC"}


# ================================================================ layer 1: status changes

def snapshot_state(rows: pd.DataFrame) -> dict:
    """One snapshot as {(game_date, matchup): {team: (submitted, {player: status})}}.

    A team whose only row says NOT YET SUBMITTED has submitted = False.
    """
    state: dict = {}
    for r in rows.itertuples(index=False):
        teams = state.setdefault((r.game_date, r.matchup), {})
        submitted, players = teams.get(r.team, (True, {}))
        if r.reason.strip().casefold() == NOT_SUBMITTED.casefold():
            submitted = False
        elif r.player_name:
            players[r.player_name] = r.status
        teams[r.team] = (submitted, players)
    return state


def compare_snapshots(prev: dict, cur: dict) -> list[dict]:
    """Every player whose status for a game differs between two snapshots.

    Only games listed in both snapshots, and teams that appear in the later one, are
    compared. A player missing from a snapshot has the status "".
    """
    out = []
    for game, teams in cur.items():
        if game not in prev:
            continue
        for team, (submitted_now, players_now) in teams.items():
            listed_prev = team in prev[game]
            submitted_prev, players_prev = prev[game].get(team, (False, {}))
            for player in sorted(set(players_now) | set(players_prev)):
                before, after = players_prev.get(player, ""), players_now.get(player, "")
                if before != after:
                    out.append({"game_date": game[0], "matchup": game[1], "team_name": team,
                                "player_name": player, "prev_status": before, "status": after,
                                "team_submitted_prev": bool(listed_prev and submitted_prev),
                                "team_submitted_now": bool(submitted_now)})
    return out


def first_out_snapshot(rows: pd.DataFrame, order: dict[str, int]) -> pd.DataFrame:
    """For each game, team and player: the first snapshot in which his status is Out."""
    out = rows[rows["status"] == OUT].copy()
    out["snapshot_order"] = out["report_file"].map(order)
    first = out.sort_values("snapshot_order").drop_duplicates(
        ["game_date", "matchup", "team", "player_name"])
    return first[["game_date", "matchup", "team", "player_name", "report_file"]].rename(
        columns={"team": "team_name", "report_file": "first_out_report_file"})


def build_status_changes(rows: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """All status changes between consecutive parsed snapshots, in scheduled order.

    unparsed_between counts the snapshots that exist on the server between the two but
    could not be parsed. first_out is 1 when the later snapshot is the first one in which
    the player is Out for that game.
    """
    snaps = snapshots.sort_values("scheduled_time_et").reset_index(drop=True)
    parsed = snaps[snaps["parsed"] == 1]
    order = {name: i for i, name in enumerate(snaps["report_file"])}
    unparsed = ((snaps["on_server"] == 1) & (snaps["parsed"] != 1)).cumsum().to_numpy()
    by_file = {name: g for name, g in rows.groupby("report_file", sort=False)}
    empty = rows.iloc[0:0]

    records, prev_name, prev_state = [], None, None
    for name in parsed["report_file"]:
        state = snapshot_state(by_file.get(name, empty))
        if prev_state is not None:
            gap = int(unparsed[order[name]] - unparsed[order[prev_name]])
            for rec in compare_snapshots(prev_state, state):
                records.append({"report_file": name, "prev_report_file": prev_name,
                                "unparsed_between": gap, **rec})
        prev_name, prev_state = name, state
    cols = ["report_file", "prev_report_file", "unparsed_between", "game_date", "matchup",
            "team_name", "player_name", "prev_status", "status", "team_submitted_prev",
            "team_submitted_now"]
    changes = pd.DataFrame(records, columns=cols)
    first = first_out_snapshot(rows, order)
    changes = changes.merge(first, on=["game_date", "matchup", "team_name", "player_name"],
                            how="left")
    changes["first_out"] = (changes["first_out_report_file"] == changes["report_file"]) \
        & (changes["status"] == OUT)
    changes = changes.drop(columns="first_out_report_file")
    for col in ("team_submitted_prev", "team_submitted_now", "first_out"):
        changes[col] = changes[col].astype(int)
    return changes


def read_rows(path: Path = ROWS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def read_snapshots(path: Path = SNAPSHOTS_PATH) -> pd.DataFrame:
    """The snapshot table with numeric flags (empty cells become missing)."""
    s = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    for col in ("on_server", "parsed", "n_rows", "n_bytes", "http_status"):
        s[col] = pd.to_numeric(s[col].where(s[col] != ""), errors="coerce").astype("Int64")
    return s


def read_changes(path: Path = CHANGES_PATH) -> pd.DataFrame:
    c = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    for col in ("unparsed_between", "team_submitted_prev", "team_submitted_now", "first_out"):
        c[col] = c[col].astype(int)
    return c


# ================================================================ layer 2: events

def release_times(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Per snapshot: the scheduled time, the posting time and the release time used (UTC).

    The posting time is used when it lies from 30 minutes before to 60 minutes after the
    scheduled time. Otherwise the scheduled time is the release time.
    """
    out = snapshots[["report_file"]].copy()
    # Hourly file names carry no minute, so the time printed in the file comes first.
    header = snapshots.get("header_time_et", pd.Series("", index=snapshots.index))
    scheduled = pd.to_datetime(header.where(header != "", snapshots["scheduled_time_et"]))
    out["scheduled_utc"] = scheduled.dt.tz_localize(EASTERN, ambiguous="NaT",
                                                    nonexistent="NaT").dt.tz_convert("UTC")
    posted = pd.to_datetime(snapshots["posted_utc"].where(snapshots["posted_utc"] != ""),
                            utc=True)
    lag = (posted - out["scheduled_utc"]).dt.total_seconds() / 60
    usable = posted.notna() & (lag >= -POSTED_EARLY_MIN) & (lag <= POSTED_LATE_MIN)
    out["posted_utc"] = posted
    out["posted_lag_min"] = lag
    out["release_utc"] = posted.where(usable, out["scheduled_utc"])
    out["release_source"] = np.where(usable, "posted", "scheduled")
    return out


def name_key(name: str) -> str:
    """A player name reduced to its letters: no case, accents, punctuation or suffix.

    Works for "Last, First" (injury report) and "First Last" (game log) alike:
    "Oubre Jr., Kelly" and "Kelly Oubre Jr." both give "kellyoubre".
    """
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    words = re.sub(r"[^a-z\s]", "", text.lower().replace("-", " ")).split()
    return "".join(w for w in words if w not in SUFFIXES)


def team_codes(team_names: pd.Series, matchup: pd.Series, nba_teams: dict[str, str]) -> pd.Series:
    """NBA tricode of each team name. `nba_teams` maps full names to tricodes.

    The code must be one of the two teams of the matchup, or the result is missing.
    """
    names = {**nba_teams, **TEAM_ALIASES}
    code = team_names.map(names)
    sides = matchup.str.split("@", n=1, expand=True)
    ok = (code == sides[0]) | (code == sides[1])
    return code.where(ok)


def nba_team_names() -> dict[str, str]:
    """Full team name -> tricode, from nba_api's static team list."""
    from nba_api.stats.static import teams
    return {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}


def minutes_ranks(minutes: pd.DataFrame, team: str, before_date) -> pd.DataFrame:
    """The team's players with at least 5 games played for it before `before_date`,
    ranked by minutes per game played (1 = most). Ties go to the lower player_id."""
    prior = minutes[(minutes["team"] == team) & (minutes["game_date"] < before_date)]
    table = prior.groupby("player_id").agg(
        player_name=("player_name", "first"), games=("minutes", "size"),
        minutes_per_game=("minutes", "mean")).reset_index()
    table = table[table["games"] >= MIN_PRIOR_GAMES]
    table = table.sort_values(["minutes_per_game", "player_id"], ascending=[False, True],
                              ignore_index=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def candidate_outs(changes: pd.DataFrame, game_map: pd.DataFrame,
                   nba_teams: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Status changes to Out for main-sample games, with the game and team attached.

    Returns the table and the counts at each step.
    """
    counts = {"status changes, all kinds": len(changes)}
    c = changes[changes["status"] == OUT]
    counts["changes to Out between consecutive snapshots"] = len(c)
    c = c[c["first_out"] == 1]
    counts["of those, the player's first Out for the game"] = len(c)
    c = c[c["team_submitted_prev"] == 1]
    counts["of those, team had submitted in the earlier snapshot"] = len(c)
    c = c[c["unparsed_between"] == 0].copy()
    counts["of those, no unparsed snapshot in between"] = len(c)

    c["team"] = team_codes(c["team_name"], c["matchup"], nba_teams)
    c["game_date"] = pd.to_datetime(c["game_date"], format="%m/%d/%Y").dt.date
    sides = c["matchup"].str.split("@", n=1, expand=True)
    c["away_team"], c["home_team"] = sides[0], sides[1]
    games = game_map.loc[game_map["main_sample"], ["game_id", "game_date", "away_team",
                                                   "home_team", "tipoff_utc"]]
    c = c.merge(games, on=["game_date", "away_team", "home_team"], how="left")
    counts["of those, team name not recognized"] = int(c["team"].isna().sum())
    c = c[c["team"].notna() & c["game_id"].notna()].reset_index(drop=True)
    counts["of those, for a main-sample game"] = len(c)
    return c, counts


def mark_key_players(c: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Add player_id, minutes_rank, minutes_per_game and key_player to the candidates.

    name_matched says whether the report name was found among the players who appear in
    the team's game logs at any time in the season.
    """
    keys = minutes.assign(key=minutes["player_name"].map(name_key))
    roster = {team: dict(zip(g["key"], g["player_id"])) for team, g in keys.groupby("team")}
    rank_cache: dict = {}
    rows = []
    for r in c.itertuples(index=False):
        player_id = roster.get(r.team, {}).get(name_key(r.player_name))
        if (r.team, r.game_date) not in rank_cache:
            rank_cache[(r.team, r.game_date)] = minutes_ranks(minutes, r.team, r.game_date)
        ranks = rank_cache[(r.team, r.game_date)]
        hit = ranks[ranks["player_id"] == player_id] if player_id is not None else ranks.iloc[0:0]
        rows.append({"player_id": player_id if player_id is not None else pd.NA,
                     "name_matched": player_id is not None,
                     "minutes_rank": int(hit["rank"].iloc[0]) if len(hit) else pd.NA,
                     "minutes_per_game": float(hit["minutes_per_game"].iloc[0]) if len(hit)
                     else np.nan,
                     "prior_games": int(hit["games"].iloc[0]) if len(hit) else pd.NA})
    out = pd.concat([c.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["player_id"] = out["player_id"].astype("Int64")
    out["minutes_rank"] = out["minutes_rank"].astype("Int64")
    out["prior_games"] = out["prior_games"].astype("Int64")
    out["key_player"] = out["minutes_rank"].notna() & (out["minutes_rank"] <= KEY_RANK)
    return out


def build_events(candidates: pd.DataFrame, releases: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One event per team, game and snapshot with at least one key player changing to Out.

    Returns the events and the counts at each step. minutes_before_tipoff is the time from
    the release to the official tipoff.
    """
    counts = {"changes to Out with the name matched to a player": int(
        candidates["name_matched"].sum()),
        "changes to Out by a key player": int(candidates["key_player"].sum())}
    key = candidates[candidates["key_player"]]
    events = key.groupby(["game_id", "team", "report_file"], sort=False).agg(
        game_date=("game_date", "first"), home_team=("home_team", "first"),
        away_team=("away_team", "first"), tipoff_utc=("tipoff_utc", "first"),
        prev_report_file=("prev_report_file", "first"),
        n_key_players=("player_name", "size"),
        players=("player_name", lambda s: "; ".join(s)),
        prev_statuses=("prev_status", lambda s: "; ".join(v if v else "not listed" for v in s)),
        best_minutes_rank=("minutes_rank", "min")).reset_index()
    counts["events (team, game, snapshot)"] = len(events)
    both = events.groupby(["game_id", "report_file"])["team"].transform("nunique") > 1
    counts["dropped: both teams in the same snapshot"] = int(both.sum())
    events = events[~both]
    events = events.merge(releases[["report_file", "scheduled_utc", "posted_utc",
                                    "release_utc", "release_source"]], on="report_file",
                          how="left", validate="many_to_one")
    events["minutes_before_tipoff"] = (
        (events["tipoff_utc"] - events["release_utc"]).dt.total_seconds() / 60)
    late = ~(events["minutes_before_tipoff"] >= MIN_LEAD_MINUTES)
    counts[f"dropped: released less than {MIN_LEAD_MINUTES} minutes before tipoff"] = int(
        late.sum())
    events = events[~late].copy()
    events["side"] = np.where(events["team"] == events["home_team"], "home", "away")
    counts["events released in time"] = len(events)
    events = events.sort_values(["release_utc", "game_id", "team"], ignore_index=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    return events, counts


def load_events() -> dict:
    """Events from the processed files, with the counts and the tables behind them."""
    snapshots = read_snapshots()
    changes = read_changes()
    game_map = read_csv(GAME_MAP_PATH)
    minutes = read_csv(MINUTES_PATH)
    candidates, counts = candidate_outs(changes, game_map, nba_team_names())
    candidates = mark_key_players(candidates, minutes)
    releases = release_times(snapshots)
    events, more = build_events(candidates, releases)
    return {"events": events, "counts": {**counts, **more}, "candidates": candidates,
            "releases": releases, "snapshots": snapshots}


def sample_for_check(loaded: dict, n: int = CHECK_EVENTS, seed: int = CHECK_SEED) -> pd.DataFrame:
    """A random draw of events for the check against the original report files.

    One row per key player of each drawn event, with what the event says about him: the
    two snapshots, the game, his status before, and the scheduled report time.
    """
    events = loaded["events"]
    drawn = events.sample(n=min(n, len(events)), random_state=seed).sort_values("event_id")
    key = loaded["candidates"][loaded["candidates"]["key_player"]]
    rows = drawn[["event_id", "game_id", "team", "report_file"]].merge(
        key[["game_id", "team", "report_file", "prev_report_file", "matchup", "team_name",
             "player_name", "prev_status"]], on=["game_id", "team", "report_file"])
    game_date = drawn.set_index("event_id")["game_date"]
    rows["game_date"] = [game_date[e].strftime("%m/%d/%Y") for e in rows["event_id"]]
    scheduled = loaded["snapshots"].set_index("report_file")["scheduled_time_et"]
    rows["scheduled_time_et"] = rows["report_file"].map(scheduled)
    return rows[["event_id", "report_file", "prev_report_file", "game_date", "matchup",
                 "team_name", "player_name", "prev_status", "scheduled_time_et"]]


def reliability(snapshots: pd.DataFrame, rows_games: pd.DataFrame | None,
                game_map: pd.DataFrame, check: pd.DataFrame | None = None) -> pd.DataFrame:
    """The checks of the plan's reliability rule: two coverage shares that need at least
    95%, and the hand check, in which every checked event must agree with the report files.

    rows_games: one row per (game_date, matchup, team_name) seen in any parsed snapshot,
    or None to leave the game coverage out. check: the output of envs/injuries/
    check_events.py (one row per player of each checked event), or None to leave it out.
    """
    on_server = int((snapshots["on_server"] == 1).sum())
    parsed = int((snapshots["parsed"] == 1).sum())
    rows = [{"check": "snapshots on the server that were fetched and parsed",
             "count": parsed, "out_of": on_server, "share": parsed / on_server}]
    if rows_games is not None:
        seen = rows_games.copy()
        seen["game_date"] = pd.to_datetime(seen["game_date"], format="%m/%d/%Y").dt.date
        teams = seen.groupby(["game_date", "matchup"])["team_name"].nunique()
        both = set(teams[teams >= 2].index)
        main = game_map[game_map["main_sample"]]
        keys = list(zip(main["game_date"], main["away_team"] + "@" + main["home_team"]))
        covered = sum(k in both for k in keys)
        rows.append({"check": "main-sample games in a parsed snapshot with both teams",
                     "count": covered, "out_of": len(keys), "share": covered / len(keys)})
    for row in rows:
        row["required_share"] = 0.95
    if check is not None:
        agrees = check["agrees"].astype(int).groupby(check["event_id"]).min()
        rows.append({"check": "events of the hand check that agree with the report files",
                     "count": int(agrees.sum()), "out_of": len(agrees),
                     "share": float(agrees.mean()), "required_share": 1.0})
    out = pd.DataFrame(rows)
    out["passes"] = (out["share"] >= out["required_share"]).astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", help="write the events drawn for the hand check here")
    args = parser.parse_args()
    if args.sample:
        sample = sample_for_check(load_events())
        sample.to_csv(args.sample, index=False, encoding="utf-8", lineterminator="\n")
        print(f"Saved {sample['event_id'].nunique()} events ({len(sample)} player rows) to "
              f"{args.sample}")
        return
    rows = read_rows()
    snapshots = read_snapshots()
    changes = build_status_changes(rows, snapshots)
    changes.to_csv(CHANGES_PATH, index=False, encoding="utf-8", lineterminator="\n")
    games = rows[["game_date", "matchup", "team"]].drop_duplicates().rename(
        columns={"team": "team_name"})
    games.to_csv(PROCESSED / "injury_report_games_2025_26.csv", index=False, encoding="utf-8",
                 lineterminator="\n")
    print(f"Saved {len(changes):,} status changes to {CHANGES_PATH.relative_to(REPO_ROOT)}")
    print(f"Changes to Out: {(changes['status'] == OUT).sum():,}, of which first Out for "
          f"the game: {changes['first_out'].sum():,}")
    print(reliability(snapshots, games, read_csv(GAME_MAP_PATH)).to_string(index=False))


if __name__ == "__main__":
    main()

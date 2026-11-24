"""Stage 10.4 D2 event construction on small hand-made reports (no real data)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import injury_events as ie

NBA = {"Alpha City Aces": "AAA", "Beta Town Bears": "BBB", "Gamma Bay Gulls": "CCC"}


def rows(report_file: str, entries: list[tuple]) -> pd.DataFrame:
    """entries: (matchup, team, player, status, reason). Game date fixed."""
    return pd.DataFrame([{"report_file": report_file, "game_date": "01/10/2026",
                          "game_time": "07:00 (ET)", "matchup": m, "team": t,
                          "player_name": p, "status": s, "reason": r}
                         for m, t, p, s, r in entries])


def snaps(files: list[tuple]) -> pd.DataFrame:
    """files: (report_file, scheduled ET, posted UTC, on_server, parsed)."""
    return pd.DataFrame([{"report_file": f, "scheduled_time_et": s, "posted_utc": p,
                          "on_server": o, "parsed": q} for f, s, p, o, q in files])


G = "BBB@AAA"
R1 = rows("r1", [(G, "Alpha City Aces", "Star, Sam", "Questionable", "Injury/Illness - Knee"),
                 (G, "Alpha City Aces", "Bench, Bo", "Out", "G League - Two-Way"),
                 (G, "Beta Town Bears", "", "", "NOT YET SUBMITTED")])
R2 = rows("r2", [(G, "Alpha City Aces", "Star, Sam", "Out", "Injury/Illness - Knee"),
                 (G, "Alpha City Aces", "Bench, Bo", "Out", "G League - Two-Way"),
                 (G, "Alpha City Aces", "Late, Lou", "Out", "Injury/Illness - Illness"),
                 (G, "Beta Town Bears", "Wing, Will", "Out", "Injury/Illness - Ankle")])
R3 = rows("r3", [(G, "Alpha City Aces", "Star, Sam", "Out", "Injury/Illness - Knee"),
                 (G, "Alpha City Aces", "Bench, Bo", "Out", "G League - Two-Way"),
                 (G, "Beta Town Bears", "Wing, Will", "Out", "Injury/Illness - Ankle"),
                 (G, "Beta Town Bears", "Guard, Gus", "Doubtful", "Injury/Illness - Back")])
SNAPS = snaps([("r1", "2026-01-10T13:00:00", "2026-01-10T18:00:08Z", 1, 1),
               ("r2", "2026-01-10T13:15:00", "2026-01-10T18:16:30Z", 1, 1),
               ("r3", "2026-01-10T13:30:00", "", 1, 1)])


def test_snapshot_state_reads_submission_and_statuses():
    state = ie.snapshot_state(R1)[("01/10/2026", G)]
    assert state["Alpha City Aces"] == (True, {"Star, Sam": "Questionable", "Bench, Bo": "Out"})
    assert state["Beta Town Bears"] == (False, {})


def changes_table() -> pd.DataFrame:
    return ie.build_status_changes(pd.concat([R1, R2, R3], ignore_index=True), SNAPS)


def test_status_changes_between_consecutive_snapshots():
    c = changes_table()
    got = {(r.report_file, r.player_name): r for r in c.itertuples(index=False)}
    assert set(got) == {("r2", "Star, Sam"), ("r2", "Late, Lou"), ("r2", "Wing, Will"),
                        ("r3", "Late, Lou"), ("r3", "Guard, Gus")}
    star = got[("r2", "Star, Sam")]
    assert (star.prev_status, star.status, star.team_submitted_prev, star.first_out) == (
        "Questionable", "Out", 1, 1)
    late = got[("r2", "Late, Lou")]              # not listed before, team had submitted
    assert (late.prev_status, late.status, late.team_submitted_prev, late.first_out) == (
        "", "Out", 1, 1)
    wing = got[("r2", "Wing, Will")]             # the team's first submitted report
    assert (wing.status, wing.team_submitted_prev, wing.first_out) == ("Out", 0, 1)
    gone = got[("r3", "Late, Lou")]              # dropped from the report
    assert (gone.prev_status, gone.status, gone.first_out) == ("Out", "", 0)
    assert (c["prev_report_file"] == c["report_file"].map({"r2": "r1", "r3": "r2"})).all()
    assert (c["unparsed_between"] == 0).all()


def test_a_player_who_was_out_earlier_cannot_be_a_first_out_again():
    # Out in r1, missing in r2 (as when the parser drops a row), Out again in r3.
    a = rows("r1", [(G, "Alpha City Aces", "Star, Sam", "Out", "x")])
    b = rows("r2", [(G, "Alpha City Aces", "Other, Ollie", "Probable", "x")])
    c3 = rows("r3", [(G, "Alpha City Aces", "Star, Sam", "Out", "x")])
    c = ie.build_status_changes(pd.concat([a, b, c3], ignore_index=True), SNAPS)
    back = c[(c["report_file"] == "r3") & (c["player_name"] == "Star, Sam")].iloc[0]
    assert (back["prev_status"], back["status"], back["first_out"]) == ("", "Out", 0)


def test_unparsed_snapshot_between_two_parsed_ones_is_counted():
    s = snaps([("r1", "2026-01-10T13:00:00", "", 1, 1), ("rx", "2026-01-10T13:15:00", "", 1, 0),
               ("ry", "2026-01-10T13:20:00", "", 0, pd.NA), ("r2", "2026-01-10T13:30:00", "", 1, 1)])
    c = ie.build_status_changes(pd.concat([R1, R2], ignore_index=True), s)
    assert (c["prev_report_file"] == "r1").all() and (c["unparsed_between"] == 1).all()


def test_games_not_in_the_earlier_snapshot_create_no_changes():
    new_game = rows("r2", [("CCC@AAA", "Gamma Bay Gulls", "New, Ned", "Out", "x")])
    c = ie.build_status_changes(pd.concat([R1, new_game], ignore_index=True), SNAPS)
    assert "New, Ned" not in set(c["player_name"])


def test_release_time_uses_the_posting_time_only_when_it_is_near_the_schedule():
    s = snaps([("a", "2026-01-10T13:00:00", "2026-01-10T18:00:08Z", 1, 1),    # 8 s late
               ("b", "2026-01-10T13:15:00", "2026-01-10T19:15:01Z", 1, 1),    # exactly +60 min
               ("c", "2026-01-10T13:30:00", "2026-03-02T10:00:00Z", 1, 1),    # re-upload
               ("d", "2026-01-10T13:45:00", "", 1, 1),                        # no header
               ("e", "2026-07-10T13:00:00", "2026-07-10T16:59:00Z", 1, 1)])   # summer time
    r = ie.release_times(s).set_index("report_file")
    assert r.loc["a", "release_source"] == "posted"
    assert r.loc["a", "release_utc"] == pd.Timestamp("2026-01-10T18:00:08Z")
    assert r.loc["a", "scheduled_utc"] == pd.Timestamp("2026-01-10T18:00:00Z")
    assert r.loc["b", "release_source"] == "scheduled"            # 60 min 1 s is outside
    assert r.loc["c", "release_source"] == "scheduled"
    assert r.loc["c", "release_utc"] == pd.Timestamp("2026-01-10T18:30:00Z")
    assert r.loc["d", "release_source"] == "scheduled"
    assert r.loc["e", "scheduled_utc"] == pd.Timestamp("2026-07-10T17:00:00Z")   # EDT
    assert r.loc["e", "release_source"] == "posted"               # 1 minute early is inside


@pytest.mark.parametrize("a, b", [
    ("Oubre Jr., Kelly", "Kelly Oubre Jr."),
    ("Gilgeous-Alexander, Shai", "Shai Gilgeous-Alexander"),
    ("Dončić, Luka", "Luka Doncic"),
    ("O'Neale, Royce", "Royce O'Neale"),
    ("Jackson Jr., Jaren", "Jaren Jackson Jr."),
    ("Williams III, Robert", "Robert Williams III"),
    ("Nance Jr.,Larry", "Larry Nance Jr."),
])
def test_name_key_matches_report_and_game_log_spellings(a, b):
    assert ie.name_key(a) == ie.name_key(b)


def test_name_key_keeps_different_players_apart():
    assert ie.name_key("Williams, Jalen") != ie.name_key("Williams, Jaylin")
    assert ie.name_key("Green, Jalen") != ie.name_key("Green, Jeff")


def test_team_codes_need_a_known_name_on_one_side_of_the_matchup():
    names = pd.Series(["Alpha City Aces", "Beta Town Bears", "Gamma Bay Gulls", "Nobody",
                       "LA Clippers"])
    matchup = pd.Series([G, G, G, G, "LAC@AAA"])
    out = ie.team_codes(names, matchup, NBA)
    assert out.tolist()[:2] == ["AAA", "BBB"] and out.isna().tolist()[2:4] == [True, True]
    assert out.iloc[4] == "LAC"


def minutes_table() -> pd.DataFrame:
    """AAA: six games before January 10. Star 36, Second 30, Third 28, Fourth 20 minutes,
    Rookie 40 minutes in only two games, Late 30 minutes but only after the game."""
    recs = []
    for i in range(6):
        d = date(2026, 1, 1 + i)
        gid = f"g{i}"
        for pid, name, mins in ((1, "Sam Star", 36), (2, "Sid Second", 30), (3, "Tom Third", 28),
                                (4, "Finn Fourth", 20)):
            recs.append((gid, d, "AAA", pid, name, mins))
        if i >= 4:
            recs.append((gid, d, "AAA", 5, "Ray Rookie", 40))
    for i in range(6):
        recs.append((f"h{i}", date(2026, 1, 11 + i), "AAA", 6, "Lou Late", 30))
        recs.append((f"b{i}", date(2026, 1, 1 + i), "BBB", 7, "Will Wing", 33))
    return pd.DataFrame(recs, columns=["game_id", "game_date", "team", "player_id",
                                       "player_name", "minutes"])


def test_minutes_ranks_use_only_earlier_games_and_need_five_of_them():
    ranks = ie.minutes_ranks(minutes_table(), "AAA", date(2026, 1, 10))
    assert ranks["player_name"].tolist() == ["Sam Star", "Sid Second", "Tom Third", "Finn Fourth"]
    assert ranks["rank"].tolist() == [1, 2, 3, 4]
    assert ranks["minutes_per_game"].tolist() == [36, 30, 28, 20]
    # Ray Rookie has 2 games, Lou Late's games come later: neither is ranked.
    assert ie.minutes_ranks(minutes_table(), "AAA", date(2026, 1, 5)).empty   # 4 games only
    assert len(ie.minutes_ranks(minutes_table(), "AAA", date(2026, 1, 6))) == 4


def game_map() -> pd.DataFrame:
    return pd.DataFrame({"game_id": ["0022500500"], "game_date": [date(2026, 1, 10)],
                         "away_team": ["BBB"], "home_team": ["AAA"], "main_sample": [True],
                         "tipoff_utc": [pd.Timestamp("2026-01-11T00:00:00Z")]})


def test_candidates_keep_first_outs_from_teams_that_had_submitted():
    c, counts = ie.candidate_outs(changes_table(), game_map(), NBA)
    assert set(c["player_name"]) == {"Star, Sam", "Late, Lou"}      # Wing: first submission
    assert (c["team"] == "AAA").all() and (c["game_id"] == "0022500500").all()
    assert counts["changes to Out between consecutive snapshots"] == 3
    assert counts["of those, team had submitted in the earlier snapshot"] == 2
    assert counts["of those, for a main-sample game"] == 2
    none, _ = ie.candidate_outs(changes_table(), game_map().assign(main_sample=False), NBA)
    assert none.empty


def test_key_players_are_the_top_three_by_minutes():
    c, _ = ie.candidate_outs(changes_table(), game_map(), NBA)
    marked = ie.mark_key_players(c, minutes_table()).set_index("player_name")
    assert marked.loc["Star, Sam", "key_player"] and marked.loc["Star, Sam", "minutes_rank"] == 1
    # Lou Late is in the game logs (later games) but has no earlier game: not ranked.
    assert marked.loc["Late, Lou", "name_matched"] and not marked.loc["Late, Lou", "key_player"]
    fourth = c.iloc[[0]].assign(player_name="Fourth, Finn")
    assert not ie.mark_key_players(fourth, minutes_table())["key_player"].iloc[0]
    unknown = c.iloc[[0]].assign(player_name="Ghost, Gary")
    row = ie.mark_key_players(unknown, minutes_table()).iloc[0]
    assert not row["name_matched"] and not row["key_player"] and pd.isna(row["minutes_rank"])


def events_from(changes: pd.DataFrame, snapshots: pd.DataFrame = SNAPS):
    c, _ = ie.candidate_outs(changes, game_map(), NBA)
    marked = ie.mark_key_players(c, minutes_table())
    return ie.build_events(marked, ie.release_times(snapshots))


def test_event_uses_the_posting_time_and_the_lead_to_tipoff():
    events, counts = events_from(changes_table())
    assert len(events) == 1
    e = events.iloc[0]
    assert (e["team"], e["side"], e["players"], e["prev_statuses"]) == (
        "AAA", "home", "Star, Sam", "Questionable")
    assert e["release_source"] == "posted"
    assert e["release_utc"] == pd.Timestamp("2026-01-10T18:16:30Z")
    assert e["minutes_before_tipoff"] == pytest.approx(343.5)
    assert counts["events released in time"] == 1 and e["event_id"] == 1


def test_event_released_less_than_30_minutes_before_tipoff_is_dropped():
    late = SNAPS.copy()
    late.loc[late["report_file"] == "r2", ["scheduled_time_et", "posted_utc"]] = [
        "2026-01-10T18:45:00", "2026-01-10T23:45:10Z"]          # 14 min 50 s before tipoff
    late.loc[late["report_file"] == "r3", "scheduled_time_et"] = "2026-01-10T19:00:00"
    events, counts = events_from(changes_table(), late)
    assert events.empty
    assert counts["dropped: released less than 30 minutes before tipoff"] == 1


def test_both_teams_in_the_same_snapshot_are_dropped():
    r1 = rows("r1", [(G, "Alpha City Aces", "Star, Sam", "Questionable", "x"),
                     (G, "Beta Town Bears", "Wing, Will", "Probable", "x")])
    r2 = rows("r2", [(G, "Alpha City Aces", "Star, Sam", "Out", "x"),
                     (G, "Beta Town Bears", "Wing, Will", "Out", "x")])
    changes = ie.build_status_changes(pd.concat([r1, r2], ignore_index=True), SNAPS)
    events, counts = events_from(changes)
    assert events.empty and counts["dropped: both teams in the same snapshot"] == 2


def test_two_key_players_of_one_team_make_one_event():
    r1 = rows("r1", [(G, "Alpha City Aces", "Star, Sam", "Questionable", "x"),
                     (G, "Alpha City Aces", "Second, Sid", "Doubtful", "x")])
    r2 = rows("r2", [(G, "Alpha City Aces", "Star, Sam", "Out", "x"),
                     (G, "Alpha City Aces", "Second, Sid", "Out", "x")])
    changes = ie.build_status_changes(pd.concat([r1, r2], ignore_index=True), SNAPS)
    events, _ = events_from(changes)
    assert len(events) == 1 and events.iloc[0]["n_key_players"] == 2
    assert events.iloc[0]["best_minutes_rank"] == 1


def test_reliability_shares():
    s = snaps([("a", "2026-01-10T13:00:00", "", 1, 1), ("b", "2026-01-10T13:15:00", "", 1, 0),
               ("c", "2026-01-10T13:30:00", "", 0, pd.NA)])
    s["parsed"] = s["parsed"].astype("Int64")
    games = pd.DataFrame({"game_date": ["01/10/2026", "01/10/2026"], "matchup": [G, G],
                          "team_name": ["Alpha City Aces", "Beta Town Bears"]})
    out = ie.reliability(s, games, game_map())
    assert out["share"].tolist() == [0.5, 1.0]
    assert out["passes"].tolist() == [0, 1]
    one_team = ie.reliability(s, games.iloc[[0]], game_map())
    assert one_team["share"].iloc[1] == 0.0
    # The hand check: an event agrees only if every one of its players agrees.
    check = pd.DataFrame({"event_id": ["1", "1", "2"], "agrees": ["1", "0", "1"]})
    with_check = ie.reliability(s, games, game_map(), check).iloc[2]
    assert (with_check["count"], with_check["out_of"], with_check["passes"]) == (1, 2, 0)
    assert with_check["required_share"] == 1.0
    all_good = ie.reliability(s, games, game_map(), check.assign(agrees="1")).iloc[2]
    assert all_good["passes"] == 1


def test_sample_for_check_lists_the_key_players_of_the_drawn_events():
    c, _ = ie.candidate_outs(changes_table(), game_map(), NBA)
    marked = ie.mark_key_players(c, minutes_table())
    events, _ = ie.build_events(marked, ie.release_times(SNAPS))
    loaded = {"events": events, "candidates": marked,
              "snapshots": SNAPS.assign(on_server=1, parsed=1)}
    sample = ie.sample_for_check(loaded)
    assert len(sample) == 1                       # one event, one key player
    row = sample.iloc[0]
    assert (row["player_name"], row["prev_status"]) == ("Star, Sam", "Questionable")
    assert (row["report_file"], row["prev_report_file"]) == ("r2", "r1")
    assert row["game_date"] == "01/10/2026" and row["matchup"] == G
    assert row["scheduled_time_et"] == "2026-01-10T13:15:00"
    # the same seed draws the same events
    assert ie.sample_for_check(loaded)["event_id"].tolist() == sample["event_id"].tolist()


def test_release_time_takes_the_scheduled_time_from_the_file_header_when_there_is_one():
    # An hourly file stamped at a quarter to the hour: the name alone would say half past.
    s = snaps([("a", "2025-12-19T16:30:00", "", 1, 1), ("b", "2025-12-19T17:30:00", "", 1, 1)])
    s["header_time_et"] = ["2025-12-19T16:45:00", ""]
    r = ie.release_times(s).set_index("report_file")
    assert r.loc["a", "scheduled_utc"] == pd.Timestamp("2025-12-19T21:45:00Z")
    assert r.loc["b", "scheduled_utc"] == pd.Timestamp("2025-12-19T22:30:00Z")

"""Stage 10.4 machinery on small hand-made paths and events (no real data)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import information_arrival as ia
from stats_tools import logit

TIP = pd.Timestamp("2026-01-11T00:00:00Z")          # 7:00 pm US Eastern on January 10


def path(game_id: str, prices: dict, tipoff=TIP, game_date=date(2026, 1, 10)) -> pd.DataFrame:
    """A game's path from {minutes_to_tipoff: p_home_mid}."""
    return pd.DataFrame({"game_id": game_id, "game_date": game_date, "tipoff_utc": tipoff,
                         "minutes_to_tipoff": list(prices), "p_home_mid": list(prices.values())})


def test_step_changes_need_both_neighbours_and_do_not_bridge_gaps():
    p = path("g", {1800: 0.50, 1795: 0.50, 1790: 0.60, 1785: np.nan, 1780: 0.60, 1775: 0.55})
    c = ia.step_changes(p).set_index("minutes_to_tipoff")["abs_change"]
    assert 1800 not in c.index                               # no earlier step
    assert c[1795] == 0.0
    assert c[1790] == pytest.approx(logit(0.6) - logit(0.5))
    assert np.isnan(c[1785]) and np.isnan(c[1780])           # the gap is not bridged
    assert c[1775] == pytest.approx(abs(logit(0.55) - logit(0.6)))


def test_step_changes_do_not_run_across_games_or_missing_rows():
    two = pd.concat([path("a", {10: 0.5, 5: 0.6, 0: 0.6}), path("b", {10: 0.2, 0: 0.3})])
    c = ia.step_changes(two).set_index(["game_id", "minutes_to_tipoff"])["abs_change"]
    assert c[("a", 5)] == pytest.approx(logit(0.6) - logit(0.5)) and c[("a", 0)] == 0
    assert np.isnan(c[("b", 0)])                              # the step at 5 is missing


def changes_frame() -> pd.DataFrame:
    """Two games with a change at every listed step (values chosen by hand)."""
    rows = []
    for gid, tip, values in (
            ("a", TIP, {1500: 0.1, 1000: 0.2, 100: 0.3, 45: 0.4, 10: 1.0}),
            ("b", TIP + pd.Timedelta(hours=1), {1500: 0.3, 1000: 0.4, 100: np.nan, 45: 0.6,
                                               10: 2.0})):
        for m, v in values.items():
            rows.append({"game_id": gid, "game_date": date(2026, 1, 10), "tipoff_utc": tip,
                         "minutes_to_tipoff": m, "p_home_mid": 0.5, "abs_change": v})
    return pd.DataFrame(rows)


def test_by_minutes_averages_over_games_with_a_change():
    t = ia.by_minutes(changes_frame()).set_index("minutes_to_tipoff")
    assert t.loc[10, "mean_abs_change"] == pytest.approx(1.5) and t.loc[10, "game_steps"] == 2
    assert t.loc[100, "mean_abs_change"] == pytest.approx(0.3) and t.loc[100, "game_steps"] == 1
    assert t.loc[100, "games_possible"] == 2
    assert t.loc[1500, "range"] == "30h to 24h" and t.loc[1000, "range"] == "24h to tipoff"
    assert t.index[0] == 1500                                 # earliest first


def test_window_shares_use_the_24h_total_and_keep_the_early_window_apart():
    t = ia.window_shares(changes_frame()).set_index("window")
    total = (0.2 + 0.4) + 0.3 + (0.4 + 0.6) + (1.0 + 2.0)
    assert t.loc["24h to 6h", "share_of_24h_total"] == pytest.approx(0.6 / total)
    assert t.loc["6h to 1h", "share_of_24h_total"] == pytest.approx(0.3 / total)
    assert t.loc["1h to 30 min", "share_of_24h_total"] == pytest.approx(1.0 / total)
    assert t.loc["last 30 min", "share_of_24h_total"] == pytest.approx(3.0 / total)
    assert t["share_of_24h_total"].sum() == pytest.approx(1.0)
    early = t.loc["30h to 24h"]
    assert np.isnan(early["share_of_24h_total"]) and early["in_24h_total"] == 0
    assert early["sum_abs_change"] == pytest.approx(0.4) and early["game_steps"] == 2
    assert t.loc["6h to 1h", "game_steps"] == 1 and t.loc["6h to 1h", "game_steps_possible"] == 2
    assert t.loc["last 30 min", "share_per_hour"] == pytest.approx(3.0 / total / 0.5)


def test_window_edges_are_half_open():
    rows = [{"game_id": "a", "game_date": date(2026, 1, 10), "tipoff_utc": TIP,
             "minutes_to_tipoff": m, "p_home_mid": 0.5, "abs_change": 1.0}
            for m in (1440, 1435, 360, 355, 60, 55, 30, 25, 0)]
    t = ia.window_shares(pd.DataFrame(rows)).set_index("window")["game_steps"]
    assert t.to_dict() == {"24h to 6h": 2, "6h to 1h": 2, "1h to 30 min": 2, "last 30 min": 2,
                           "30h to 24h": 1}


def test_by_clock_uses_eastern_time_and_only_the_game_day():
    # Game a tips at 7:00 pm ET. 45 minutes before is 6:15 pm; 1,500 minutes before is the
    # day before and is left out. Game b tips at 8:00 pm ET.
    t = ia.by_clock(changes_frame()).set_index("clock_time_et")
    assert "18:15" in t.index and t.loc["18:15", "game_steps"] == 1
    assert t.loc["18:50", "mean_abs_change"] == pytest.approx(1.0)       # a, 10 min before
    assert t.loc["19:50", "mean_abs_change"] == pytest.approx(2.0)       # b, 10 min before
    assert t["game_steps"].sum() == 7                                    # 9 changes - 2 early
    assert (t["plotted"] == 0).all()                                     # fewer than 50 each


def test_grid_steps_cover_at_least_the_window():
    assert ia.step_at_or_before(125.0) == 125 and ia.step_at_or_after(125.0) == 125
    assert ia.step_at_or_before(123.5) == 125        # 123.5 min before tipoff: step at 125
    assert ia.step_at_or_after(123.5) == 120
    # Released 343.5 minutes before tipoff (16:30 after the hour mark).
    s = ia.window_steps(343.5)
    assert s == {"before": 350, "after": 310, "anticipation": 405}
    # Released exactly on the grid: the stated window.
    assert ia.window_steps(300.0) == {"before": 305, "after": 270, "anticipation": 360}
    assert ia.window_steps(300.0, shift=180) == {"before": 485, "after": 450,
                                                 "anticipation": 540}


def study_paths() -> pd.DataFrame:
    """Game g1: home price 0.60 until 300 minutes before tipoff, 0.50 after. Game g2: flat."""
    minutes = np.arange(1800, -1, -5)
    g1 = {int(m): (0.60 if m > 300 else 0.50) for m in minutes}
    g1[320] = 0.62
    g2 = {int(m): 0.40 for m in minutes}
    g2[900] = np.nan
    return pd.concat([path("g1", g1), path("g2", g2, game_date=date(2026, 1, 12))],
                     ignore_index=True)


def study_events() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": [1, 2, 3, 4],
        "game_id": ["g1", "g1", "g2", "g2"],
        "game_date": [date(2026, 1, 10)] * 2 + [date(2026, 1, 12)] * 2,
        "side": ["home", "away", "home", "home"],
        "minutes_before_tipoff": [300.0, 300.0, 720.0, 1750.0],
        "release_source": ["posted", "posted", "scheduled", "posted"]})


def test_event_changes_by_hand():
    c = ia.event_changes(study_events(), study_paths()).set_index("event_id")
    # Event 1, home team: 0.60 at step 305, 0.50 at step 270.
    assert c.loc[1, "change"] == pytest.approx(-0.10)
    assert c.loc[1, "placebo_change"] == pytest.approx(0.0)              # 485 -> 450
    assert c.loc[1, "anticipation_change"] == pytest.approx(0.0)         # 360 -> 305
    assert c.loc[1, "change_minus_placebo"] == pytest.approx(-0.10)
    # Event 2, away team in the same game: the mirror image.
    assert c.loc[2, "price_before"] == pytest.approx(0.40)
    assert c.loc[2, "change"] == pytest.approx(+0.10)
    # Event 3: flat path, but the placebo window starts on a step with no midpoint
    # (720 + 180 + 5 = 905 is fine, 900 would be the anticipation... check the steps).
    assert c.loc[3, "change"] == pytest.approx(0.0)
    assert c.loc[3, "placebo_step_before"] == 905 and c.loc[3, "placebo_step_after"] == 870
    # Event 4: released 1,750 minutes before tipoff. Its placebo lies before the path.
    assert c.loc[4, "change"] == pytest.approx(0.0)
    assert np.isnan(c.loc[4, "placebo_change"]) and np.isnan(c.loc[4, "anticipation_change"])


def test_anticipation_window_and_missing_midpoints():
    events = pd.DataFrame({"event_id": [1, 2], "game_id": ["g1", "g2"],
                           "game_date": [date(2026, 1, 10), date(2026, 1, 12)],
                           "side": ["home", "home"],
                           "minutes_before_tipoff": [260.0, 895.0],
                           "release_source": ["posted", "posted"]})
    c = ia.event_changes(events, study_paths()).set_index("event_id")
    # Event 1: before = step 265 (0.50), anticipation = step 320 (0.62).
    assert c.loc[1, "anticipation_change"] == pytest.approx(0.50 - 0.62)
    # Event 2: its price before is the step at 900, which has no midpoint.
    assert np.isnan(c.loc[2, "change"]) and np.isnan(c.loc[2, "anticipation_change"])
    samples = ia.study_samples(c.reset_index())
    assert samples["main"]["event_id"].tolist() == [1]
    assert samples["anticipation"]["event_id"].tolist() == [1]


def test_event_tests_report_means_and_keep_the_plan_roles():
    c = ia.event_changes(study_events(), study_paths())
    results, tests = ia.event_tests(c, n_boot=50, seed=1)
    r = results.set_index("test_id")
    assert r.loc["injury_event_mean", "n_events"] == 4
    assert r.loc["injury_event_mean", "mean"] == pytest.approx(0.0)       # -0.1, +0.1, 0, 0
    assert r.loc["injury_event_placebo_mean", "n_events"] == 3            # event 4 has none
    assert r.loc["injury_anticipation_mean", "n_events"] == 3
    assert r.loc["injury_event_mean", "n_dates"] == 2
    t = tests.set_index("test_id")
    assert t.loc["injury_event_mean", "role"] == "primary"
    assert (t.drop(index="injury_event_mean")["role"] == "secondary").all()
    assert (t["unit"] == "events").all() and (t["p_method"] == "block bootstrap").all()
    assert len(t) == 6


def test_event_counts_append_the_window_counts():
    c = ia.event_changes(study_events(), study_paths())
    table = ia.event_counts({"events released in time": 4}, c).set_index("step")["count"]
    assert table["events released in time"] == 4
    assert table["events with both midpoints of the main window"] == 4
    assert table["of those, with the placebo window"] == 3
    assert table["main-window events released at the posting time"] == 3
    assert table["main-window events released at the scheduled time"] == 1


def test_event_paths_line_up_on_the_step_before_and_stop_at_tipoff():
    events = pd.DataFrame({"event_id": [1, 2], "game_id": ["g1", "g1"],
                           "game_date": [date(2026, 1, 10)] * 2, "side": ["home", "home"],
                           "minutes_before_tipoff": [300.0, 40.0],
                           "release_source": ["posted", "posted"]})
    c = ia.event_changes(events, study_paths())
    rows = ia.event_paths(c, study_paths())
    one = rows[rows["event_id"] == 1].set_index("offset_min")["relative_price"]
    assert one[-5] == 0.0                                     # the reference step
    assert one[-20] == pytest.approx(0.02)                    # step 320 holds 0.62
    assert one[0] == pytest.approx(-0.10) and one[120] == pytest.approx(-0.10)
    assert len(one) == len(ia.PATH_OFFSETS)
    two = rows[rows["event_id"] == 2]
    assert two["offset_min"].max() == 40                      # tipoff is 40 minutes after
    avg = ia.average_path(rows, n_boot=20, seed=1).set_index("offset_min")
    assert avg.loc[0, "n_events"] == 2 and avg.loc[120, "n_events"] == 1
    assert avg.loc[120, "mean_relative_price"] == pytest.approx(-0.10)
    assert avg.loc[-5, "mean_relative_price"] == 0.0
    assert avg.loc[0, "boot_lo"] <= avg.loc[0, "mean_relative_price"] <= avg.loc[0, "boot_hi"]


# ---------------------------------------------------------------- exploratory (plan X1)

@pytest.mark.parametrize("text, group", [
    ("Questionable", "Questionable"), ("Doubtful", "Doubtful"), ("not listed", "not listed"),
    ("Questionable; Questionable", "Questionable"), ("Doubtful; Doubtful", "Doubtful"),
    ("Questionable; Doubtful", "other"), ("Probable", "other"), ("Available", "other"),
    ("not listed; Questionable", "other"),
])
def test_previous_status_group(text, group):
    assert ia.previous_status_group(text) == group


def status_changes() -> pd.DataFrame:
    rows = [("Questionable", -0.02, -0.01), ("Questionable", -0.04, 0.00),
            ("Questionable; Questionable", -0.06, -0.02), ("Doubtful", 0.00, -0.01),
            ("Doubtful", -0.01, np.nan), ("not listed", -0.08, 0.00), ("Probable", -0.03, 0.0),
            ("Questionable; Doubtful", -0.05, 0.0), ("Questionable", np.nan, -0.5)]
    return pd.DataFrame({
        "event_id": range(1, len(rows) + 1),
        "game_date": [date(2026, 1, 1) + timedelta(days=i % 4) for i in range(len(rows))],
        "prev_statuses": [r[0] for r in rows], "change": [r[1] for r in rows],
        "anticipation_change": [r[2] for r in rows]})


def test_status_split_means_counts_and_roles():
    results, tests = ia.status_split(status_changes(), n_boot=40, seed=1)
    r = results.set_index(["status_group", "window"])
    # The event without a main-window change is in no group at all (not a main event).
    assert r.loc[("Questionable", "main"), "n_events"] == 3
    assert r.loc[("Questionable", "main"), "mean"] == pytest.approx(-0.04)
    assert r.loc[("Questionable", "anticipation"), "mean"] == pytest.approx(-0.01)
    assert r.loc[("Doubtful", "main"), "n_events"] == 2
    assert r.loc[("Doubtful", "anticipation"), "n_events"] == 1      # one has no midpoint
    assert r.loc[("not listed", "main"), "mean"] == pytest.approx(-0.08)
    assert r.loc[("other", "main"), "n_events"] == 2
    assert np.isnan(r.loc[("other", "main"), "boot_lo"])             # counted, not tested
    q = r.loc[("Questionable", "main")]
    assert q["boot_lo"] <= q["mean"] <= q["boot_hi"]
    assert len(results) == 8
    assert len(tests) == 6 and (tests["role"] == "exploratory").all()
    assert set(tests["test_id"]) == {
        f"exploratory_injury_{g}_{w}" for g in ("questionable", "doubtful", "not_listed")
        for w in ("main", "anticipation")}
    t = tests.set_index("test_id").loc["exploratory_injury_questionable_main"]
    assert t["n"] == 3 and t["unit"] == "events" and t["estimate"] == pytest.approx(-0.04)

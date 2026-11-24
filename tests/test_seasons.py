"""Season parameter of the data pipeline (plan 10.6): names, sources and the partial season."""
import gzip
import json
from datetime import date

import pandas as pd
import pytest

import build_snapshots
import match_games
import nba_schedule
import pull_kalshi
import seasons
from kalshi_client import KalshiClient


def test_default_season_keeps_the_file_names_it_always_had():
    s = seasons.get_season()
    assert s.label == "2025-26" and s.source == "historical" and s.complete
    assert s.nba_games_path.name == "nba_games_2025_26.csv"
    assert s.game_map_path.name == "game_map.csv"
    assert s.snapshots_path.name == "snapshots.csv"
    assert s.match_report_path.name == "match_report.md"
    assert s.markets_cache_path.name == "kalshi_markets_KXNBAGAME.json"
    assert nba_schedule.OUT_PATH == s.nba_games_path
    assert build_snapshots.OUT_PATH == s.snapshots_path
    assert pull_kalshi.GAME_MAP_PATH == s.game_map_path


def test_another_season_carries_the_season_in_its_file_names():
    s = seasons.get_season("2026-27")
    assert s.source == "live" and not s.complete and s.kalshi_end is None
    assert s.nba_games_path.name == "nba_games_2026_27.csv"
    assert s.game_map_path.name == "game_map_2026_27.csv"
    assert s.snapshots_path.name == "snapshots_2026_27.csv"
    assert s.match_report_path.name == "match_report_2026_27.md"
    assert s.markets_cache_path.name == "kalshi_markets_KXNBAGAME_2026_27.json"
    with pytest.raises(SystemExit):
        seasons.get_season("1999-00")


def games() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["1", "2", "3", "4"],
        "game_date": [date(2026, 11, 20), date(2026, 11, 21), date(2026, 11, 21),
                      date(2026, 11, 22)],
        "status": ["Final", "Final/OT", "PPD", "Final"]})


def test_played_through_keeps_final_games_up_to_the_cutoff():
    final, not_final = nba_schedule.played_through(games(), date(2026, 11, 21))
    assert final["game_id"].tolist() == ["1", "2"]
    assert not_final["game_id"].tolist() == ["3"]            # reported, not kept
    final, not_final = nba_schedule.played_through(games(), date(2026, 11, 19))
    assert final.empty and not_final.empty


def schedule_frame(n_teams: int = 30) -> pd.DataFrame:
    teams = [f"T{i:02d}" for i in range(n_teams)]
    rows = [{"game_id": f"{i:03d}", "game_date": date(2026, 11, 1),
             "tipoff_utc": pd.Timestamp("2026-11-02T00:00:00Z"), "home_team": teams[i],
             "away_team": teams[(i + 1) % n_teams], "home_score": 100, "away_score": 90,
             "status": "Final"} for i in range(n_teams)]
    return pd.DataFrame(rows)


def test_check_skips_the_full_season_counts_for_a_season_in_progress():
    g = schedule_frame()
    assert nba_schedule.check(g, expected_games=None) == []
    full = nba_schedule.check(g, expected_games=1230)
    assert any("row count" in p for p in full) and any("30 x 82" in p for p in full)
    few = nba_schedule.check(schedule_frame(28), expected_games=None)
    assert few == ["28 teams have played, expected 30"]


def test_expiration_time_falls_back_to_the_renamed_field():
    old = {"expiration_time": "2025-11-04T23:30:00Z", "latest_expiration_time": "2099-01-01T00:00:00Z"}
    new = {"latest_expiration_time": "2025-11-04T23:30:00Z"}
    live = {"expiration_time": None, "latest_expiration_time": "2025-11-04T23:30:00Z"}
    for market in (old, new, live):
        assert match_games.expiration_time(market) == pd.Timestamp("2025-11-04T23:30:00Z")
    assert pd.isna(match_games.expiration_time({"expiration_time": None}))


def market(event: str, team: str, **extra) -> dict:
    return {"ticker": f"{event}-{team}", "event_ticker": event, "yes_sub_title": team,
            "result": "yes", "latest_expiration_time": "2026-11-15T00:30:00Z",
            "open_time": "2026-10-29T00:00:00Z", "volume_fp": "10.00", **extra}


def test_kalshi_events_keeps_only_the_seasons_dates():
    markets = []
    for event, away, home in (("KXNBAGAME-26NOV01BOSNYK", "BOS", "NYK"),
                              ("KXNBAGAME-26SEP30BOSNYK", "BOS", "NYK"),
                              ("KXNBAGAME-26NOV05LALGSW", "LAL", "GSW")):
        markets += [market(event, away), market(event, home)]
    ev = match_games.kalshi_events(markets, date(2026, 10, 1), date(2026, 11, 2))
    assert ev["event_ticker"].tolist() == ["KXNBAGAME-26NOV01BOSNYK"]
    row = ev.iloc[0]
    assert (row["kalshi_away"], row["kalshi_home"]) == ("BOS", "NYK")
    assert row["kalshi_tipoff_estimate"] == pd.Timestamp("2026-11-01T00:30:00Z")
    assert row["kalshi_volume"] == 20.0


def test_season_phase_defaults_are_2025_26_and_a_season_in_progress_uses_its_games():
    assert match_games.season_phase(date(2025, 10, 20)) == "preseason"
    assert match_games.season_phase(date(2026, 4, 12)) == "regular_season"
    assert match_games.season_phase(date(2026, 4, 15)) == "play_in"
    assert match_games.season_phase(date(2026, 4, 18)) == "playoffs"
    nba = pd.DataFrame({"game_date": [date(2026, 10, 20), date(2026, 11, 22)]})
    phases = match_games.phase_dates(seasons.get_season("2026-27"), nba)
    assert phases == (date(2026, 10, 20), date(2026, 11, 22), None)
    assert match_games.season_phase(date(2026, 10, 10), *phases) == "preseason"
    assert match_games.season_phase(date(2026, 11, 1), *phases) == "regular_season"
    assert match_games.phase_dates(seasons.get_season(), nba) == (
        date(2025, 10, 21), date(2026, 4, 12), date(2026, 4, 18))


class Recorder(KalshiClient):
    """A client that records the requests instead of sending them."""

    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        return {"markets": [], "trades": [], "candlesticks": [{"end_period_ts": 1}]}


def test_client_uses_the_endpoints_of_the_source():
    c = Recorder()
    c.markets("KXNBAGAME")
    c.markets("KXNBAGAME", "live")
    c.trades("KXNBAGAME-X-AAA", 1, 2)
    c.trades("KXNBAGAME-X-AAA", 1, 2, "live")
    c.candlesticks("KXNBAGAME-X-AAA", 1, 2)
    assert c.candlesticks("KXNBAGAME-X-AAA", 1, 2, source="live") == [{"end_period_ts": 1}]
    paths = [p for p, _ in c.calls]
    assert paths == ["/historical/markets", "/markets", "/historical/trades", "/markets/trades",
                     "/historical/markets/KXNBAGAME-X-AAA/candlesticks",
                     "/series/KXNBAGAME/markets/KXNBAGAME-X-AAA/candlesticks"]
    assert c.calls[1][1]["status"] == "settled" and "status" not in c.calls[0][1]
    assert c.calls[3][1] == {"ticker": "KXNBAGAME-X-AAA", "min_ts": 1, "max_ts": 2, "limit": 1000}
    c.historical_trades("T", 5, 6)
    assert c.calls[-1][0] == "/historical/trades"
    with pytest.raises(ValueError):
        c.markets("KXNBAGAME", "archive")


def test_check_source_stops_when_the_data_is_on_the_other_side_of_the_cutoff():
    cutoff = pd.Timestamp("2026-09-24T00:00:00Z")
    old_job = [{"ticker": "A", "tipoff_utc": "2026-04-12T22:00:00+00:00"}]
    new_job = [{"ticker": "B", "tipoff_utc": "2026-11-20T00:00:00+00:00"}]
    meta = {"A": {"open_time": "2026-04-10T00:00:00Z"}, "B": {"open_time": "2026-11-17T00:00:00Z"}}
    pull_kalshi.check_source(seasons.get_season(), old_job, meta, cutoff)
    pull_kalshi.check_source(seasons.get_season("2026-27"), new_job, meta, cutoff)
    with pytest.raises(SystemExit, match="live endpoints"):
        pull_kalshi.check_source(seasons.get_season(), new_job, meta, cutoff)
    with pytest.raises(SystemExit, match="historical endpoints"):
        pull_kalshi.check_source(seasons.get_season("2026-27"), old_job, meta, cutoff)


def test_load_market_reads_both_candle_formats(tmp_path, monkeypatch):
    trade = {"trade_id": "t1", "created_time": "2026-11-20T00:00:00Z",
             "yes_price_dollars": "0.6000", "count_fp": "2.50"}
    historical = {"end_period_ts": 100, "yes_bid": {"close": "0.6000"},
                  "yes_ask": {"close": "0.6200"}}
    live = {"end_period_ts": 160, "yes_bid": {"close_dollars": "0.6100"},
            "yes_ask": {"close_dollars": "0.6300"}}
    empty = {"end_period_ts": 220, "yes_bid": {}, "yes_ask": None}
    with gzip.open(tmp_path / "TICK.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"trades": [trade], "candlesticks": [historical, live, empty]}, f)
    monkeypatch.setattr(build_snapshots, "CACHE_DIR", tmp_path)
    trades, candles = build_snapshots.load_market("TICK")
    assert trades["count"].tolist() == [2.5]
    assert candles["bid"].tolist()[:2] == [0.60, 0.61]
    assert candles["ask"].tolist()[:2] == [0.62, 0.63]
    assert candles[["bid", "ask"]].iloc[2].isna().all()
    assert build_snapshots.load_market("MISSING") is None

"""Stage 2 feasibility probe for Kalshi NBA single-game markets (series KXNBAGAME).

Public endpoints only (no authentication). Settled 2025-26 markets live on the
/historical/* endpoints, so this probe reads from there.

What it reports:
  1. All 2025-26 KXNBAGAME events, tagged regular season / play-in / playoffs,
     with counts, date range and per-game volume distribution.
  2. How long before tipoff each market opened (can the 30h window be covered at all?).
  3. For 3 sample regular-season games (low / median / high volume): historical trades
     and 1-minute candles over the 30 hours before tipoff, with earliest timestamps.
  4. The structure of one event (markets per event, ticker format, key fields).

Tipoff here is a PROXY taken from Kalshi metadata (expiration_time - 14 days, cross-checked
against expected_expiration_time - 3 hours). The real scheduled tipoff comes from the NBA
schedule in Stage 3.

Usage:
    python scripts/probe_kalshi.py
"""
from __future__ import annotations

import json
import re
import statistics
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNBAGAME"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "interim"

SEASON_START = date(2025, 10, 1)
SEASON_END = date(2026, 6, 30)
# 2025-26 NBA calendar boundaries (by US game date). Provisional: Stage 3/5 will replace
# this with the NBA game ID prefix (001 preseason, 002 regular season, 005 play-in, 004 playoffs).
REG_SEASON_FIRST_DAY = date(2025, 10, 21)
REG_SEASON_LAST_DAY = date(2026, 4, 12)
PLAYOFFS_FIRST_DAY = date(2026, 4, 18)

WINDOW_HOURS = 30
SESSION = requests.Session()


# ---------------------------------------------------------------- HTTP helpers

def get(path: str, params: dict | None = None) -> dict:
    """GET with a small delay and simple retry. Public endpoints, no auth headers."""
    for attempt in range(5):
        time.sleep(0.1)  # stay well under the public rate limit
        resp = SESSION.get(BASE_URL + path, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def get_all(path: str, key: str, params: dict) -> list[dict]:
    """Follow the cursor until exhausted."""
    items: list[dict] = []
    params = dict(params)
    while True:
        data = get(path, params)
        items.extend(data.get(key) or [])
        cursor = data.get("cursor")
        if not cursor:
            return items
        params["cursor"] = cursor


# ---------------------------------------------------------------- parsing helpers

def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fmt(d: datetime | None) -> str:
    return d.strftime("%Y-%m-%d %H:%M:%SZ") if d else "none"


def ticker_game_date(event_ticker: str) -> date:
    """KXNBAGAME-26JAN15MILSAS -> 2026-01-15 (US local game date)."""
    m = re.match(rf"{SERIES}-(\d\d)([A-Z]{{3}})(\d\d)", event_ticker)
    return datetime.strptime("".join(m.groups()), "%y%b%d").date()


def ticker_teams(event_ticker: str) -> tuple[str, str]:
    """Last 6 letters are AWAY + HOME 3-letter codes (to be verified in Stage 5)."""
    codes = event_ticker.split("-")[1][7:]
    return codes[:3], codes[3:]


PHASES = ["preseason", "regular_season", "play_in", "playoffs"]


def phase(game_date: date) -> str:
    if game_date < REG_SEASON_FIRST_DAY:
        return "preseason"
    if game_date <= REG_SEASON_LAST_DAY:
        return "regular_season"
    if game_date < PLAYOFFS_FIRST_DAY:
        return "play_in"
    return "playoffs"


def tipoff_proxy(market: dict) -> datetime:
    return ts(market["expiration_time"]) - timedelta(days=14)


def quantiles(xs: list[float]) -> str:
    xs = sorted(xs)
    if not xs:
        return "n=0"
    q = statistics.quantiles(xs, n=4) if len(xs) > 1 else [xs[0]] * 3
    return (f"n={len(xs)}  min={xs[0]:,.0f}  p25={q[0]:,.0f}  median={q[1]:,.0f}  "
            f"p75={q[2]:,.0f}  max={xs[-1]:,.0f}")


# ---------------------------------------------------------------- 1. list events

def list_season_events() -> tuple[dict, dict]:
    cutoff = get("/historical/cutoff")
    print("Historical cutoff:", json.dumps(cutoff))

    markets = get_all("/historical/markets", "markets", {"series_ticker": SERIES, "limit": 1000})
    assert all(m["ticker"].startswith(SERIES + "-") for m in markets), "series filter not applied"
    print(f"/historical/markets returned {len(markets)} {SERIES} markets (all seasons)")

    # Completeness check: any 2025-26 game still on the live endpoint?
    live = get_all("/markets", "markets", {"series_ticker": SERIES, "limit": 1000})
    live_in_season = [m for m in live
                      if SEASON_START <= ticker_game_date(m["event_ticker"]) <= SEASON_END]
    print(f"/markets (live) returned {len(live)} {SERIES} markets; "
          f"{len(live_in_season)} of them are 2025-26 games")

    events: dict[str, dict] = {}
    for m in markets + live_in_season:
        et = m["event_ticker"]
        gd = ticker_game_date(et)
        if not SEASON_START <= gd <= SEASON_END:
            continue
        ev = events.setdefault(et, {"event_ticker": et, "game_date": gd, "phase": phase(gd),
                                    "markets": {}})
        ev["markets"][m["ticker"]] = m
    return events, cutoff


def summarize_events(events: dict) -> None:
    print("\n=== 1. 2025-26 KXNBAGAME events ===")
    by_phase: dict[str, list[dict]] = {}
    for ev in events.values():
        by_phase.setdefault(ev["phase"], []).append(ev)

    print(f"Total events: {len(events)}")
    for ph in PHASES:
        evs = by_phase.get(ph, [])
        if not evs:
            print(f"  {ph:15s}: 0")
            continue
        dates = [e["game_date"] for e in evs]
        print(f"  {ph:15s}: {len(evs):5d} events, {min(dates)} to {max(dates)}")

    n_markets = {}
    for ev in events.values():
        n_markets[len(ev["markets"])] = n_markets.get(len(ev["markets"]), 0) + 1
    print(f"Markets per event: {n_markets}")

    # Volume per game = sum of contracts traded in both team markets (whole market life).
    print("\nTotal volume per game (contracts, both markets, whole market life):")
    for ph in PHASES:
        vols = [sum(float(m["volume_fp"]) for m in e["markets"].values())
                for e in by_phase.get(ph, [])]
        print(f"  {ph:15s}: {quantiles(vols)}")

    # Settlement results: anything other than a clean yes/no is a problem game.
    odd = sorted({(e["event_ticker"], m["result"], m.get("settlement_value_dollars"))
                  for e in events.values() for m in e["markets"].values()
                  if m["result"] not in ("yes", "no")})
    if odd:
        print("\nMarkets not settled yes/no (likely postponed/cancelled games):")
        for et, res, val in odd:
            print(f"  {et}  result={res}  settlement_value={val}")

    # Regular season: games per team, and games that were actually played (yes/no result).
    reg = by_phase.get("regular_season", [])
    per_team: dict[str, int] = {}
    for e in reg:
        for team in ticker_teams(e["event_ticker"]):
            per_team[team] = per_team.get(team, 0) + 1
    counts: dict[int, list[str]] = {}
    for team, n in per_team.items():
        counts.setdefault(n, []).append(team)
    played = [e for e in reg if all(m["result"] in ("yes", "no") for m in e["markets"].values())]
    summary = "; ".join(f"{n} -> {sorted(teams)}" if len(teams) < 10 else f"{n} -> {len(teams)} teams"
                        for n, teams in sorted(counts.items()))
    print(f"\nRegular season: {len(per_team)} teams; events per team: {summary}")
    print(f"Regular season events with a yes/no result (game played): {len(played)} "
          f"(NBA schedule has 1,230)")

    # Tipoff proxy consistency checks.
    bad_proxy, bad_date = [], []
    for e in events.values():
        m = next(iter(e["markets"].values()))
        t1 = tipoff_proxy(m)
        t2 = ts(m["expected_expiration_time"]) - timedelta(hours=3)
        if t1 != t2:
            bad_proxy.append(e["event_ticker"])
        # UTC-6h maps every NBA tipoff (noon to 10:30pm ET, plus Europe/Mexico games)
        # onto its US game date without needing a timezone database.
        if (t1 - timedelta(hours=6)).date() != e["game_date"]:
            bad_date.append((e["event_ticker"], fmt(t1)))
    print(f"\nTipoff proxy (expiration_time - 14d) disagrees with expected_expiration_time - 3h: "
          f"{len(bad_proxy)} events {bad_proxy[:5]}")
    print(f"Tipoff proxy date disagrees with ticker game date: {len(bad_date)} events {bad_date[:5]}")


# ---------------------------------------------------------------- 2. open time vs tipoff

def summarize_open_lead(events: dict) -> None:
    print(f"\n=== 2. Market open time relative to tipoff proxy (regular season) ===")
    leads = []
    short = []
    for e in events.values():
        if e["phase"] != "regular_season":
            continue
        for m in e["markets"].values():
            lead_h = (tipoff_proxy(m) - ts(m["open_time"])).total_seconds() / 3600
            leads.append(lead_h)
            if lead_h < WINDOW_HOURS:
                short.append((m["ticker"], round(lead_h, 1)))
    print(f"Hours from market open to tipoff: {quantiles(leads)}")
    print(f"Markets opened less than {WINDOW_HOURS}h before tipoff: {len(short)} of {len(leads)}")
    for t, h in sorted(short, key=lambda x: x[1])[:10]:
        print(f"  {t}: opened {h}h before tipoff")


# ---------------------------------------------------------------- 3. sample games

def pick_samples(events: dict) -> list[dict]:
    """Low (p10), median and high (p90) volume regular-season games with clean yes/no results."""
    reg = [e for e in events.values()
           if e["phase"] == "regular_season"
           and all(m["result"] in ("yes", "no") for m in e["markets"].values())]
    reg.sort(key=lambda e: sum(float(m["volume_fp"]) for m in e["markets"].values()))
    n = len(reg)
    return [reg[int(n * 0.10)], reg[n // 2], reg[int(n * 0.90)]]


def probe_market(m: dict) -> dict:
    tip = tipoff_proxy(m)
    start = tip - timedelta(hours=WINDOW_HOURS)
    t0, t1 = int(start.timestamp()), int(tip.timestamp())

    # Trades inside the window, strictly before tipoff (max_ts is treated as inclusive,
    # so filter afterwards).
    trades = get_all("/historical/trades", "trades",
                     {"ticker": m["ticker"], "min_ts": t0, "max_ts": t1, "limit": 1000})
    trades = [t for t in trades if ts(t["created_time"]) < tip]
    times = sorted(ts(t["created_time"]) for t in trades)

    # Trades before the window: proves activity predates the window, gives earliest trade.
    pre = get_all("/historical/trades", "trades",
                  {"ticker": m["ticker"], "max_ts": t0 - 1, "limit": 1000})
    pre_times = sorted(ts(t["created_time"]) for t in pre)

    # 1-minute candles over the window.
    candles = get(f"/historical/markets/{m['ticker']}/candlesticks",
                  {"start_ts": t0, "end_ts": t1, "period_interval": 1}).get("candlesticks") or []
    cts = [c["end_period_ts"] for c in candles]
    with_quote = [c for c in candles
                  if (c.get("yes_bid") or {}).get("close") is not None
                  and (c.get("yes_ask") or {}).get("close") is not None]
    with_trade = [c for c in candles if (c.get("price") or {}).get("close") is not None]
    gaps = [(b - a) / 60 for a, b in zip(cts, cts[1:])]
    last_quote = with_quote[-1] if with_quote else None

    gap_trades_h = max(((b - a).total_seconds() / 3600 for a, b in zip(times, times[1:])),
                       default=None)
    return {
        "ticker": m["ticker"],
        "team": m["yes_sub_title"],
        "tipoff_proxy": fmt(tip),
        "window_start": fmt(start),
        "open_time": m["open_time"],
        "hours_open_before_tipoff": round((tip - ts(m["open_time"])).total_seconds() / 3600, 1),
        "trades_in_window": len(trades),
        "contracts_in_window": round(sum(float(t["count_fp"]) for t in trades), 2),
        "first_trade_in_window": fmt(times[0] if times else None),
        "last_trade_in_window": fmt(times[-1] if times else None),
        "max_gap_between_trades_h": round(gap_trades_h, 2) if gap_trades_h is not None else None,
        "trades_before_window": len(pre),
        "earliest_trade_overall": fmt(pre_times[0] if pre_times else (times[0] if times else None)),
        "candles": len(candles),
        "candle_minutes_in_window": WINDOW_HOURS * 60 + 1,
        "first_candle": fmt(datetime.fromtimestamp(cts[0], timezone.utc)) if cts else "none",
        "last_candle": fmt(datetime.fromtimestamp(cts[-1], timezone.utc)) if cts else "none",
        "max_gap_between_candles_min": max(gaps, default=None),
        "candles_with_bid_and_ask": len(with_quote),
        "candles_with_trade": len(with_trade),
        "last_bid_ask": (f"{last_quote['yes_bid']['close']}/{last_quote['yes_ask']['close']} at "
                         f"{fmt(datetime.fromtimestamp(last_quote['end_period_ts'], timezone.utc))}"
                         if last_quote else "none"),
    }


def summarize_samples(events: dict) -> list[dict]:
    print(f"\n=== 3. Sample games: {WINDOW_HOURS}h pre-tipoff coverage (historical trades + 1m candles) ===")
    results = []
    for label, ev in zip(["low volume (p10)", "median volume", "high volume (p90)"],
                         pick_samples(events)):
        vol = sum(float(m["volume_fp"]) for m in ev["markets"].values())
        print(f"\n--- {label}: {ev['event_ticker']}  ({next(iter(ev['markets'].values()))['title']})"
              f"  total volume {vol:,.0f}")
        for m in ev["markets"].values():
            r = probe_market(m)
            results.append({"sample": label, "event_ticker": ev["event_ticker"], **r})
            for k, v in r.items():
                print(f"    {k:30s} {v}")
            print()
    return results


# ---------------------------------------------------------------- 4. event structure

def show_event_structure(events: dict) -> None:
    print("\n=== 4. Structure of one event ===")
    ev = sorted((e for e in events.values() if e["phase"] == "regular_season"),
                key=lambda e: e["event_ticker"])[0]
    meta = get(f"/events/{ev['event_ticker']}")["event"]
    away, home = ticker_teams(ev["event_ticker"])
    print(f"Event ticker: {ev['event_ticker']}  format {SERIES}-<YY><MON><DD><AWAY><HOME>")
    print(f"  parsed: date={ev['game_date']}  away={away}  home={home}")
    print(f"  /events title: {meta['title']!r}  sub_title: {meta['sub_title']!r}")
    print(f"  markets: {len(ev['markets'])}  (format <event_ticker>-<TEAM>, YES = that team wins)")
    for m in ev["markets"].values():
        print(f"    {m['ticker']:35s} yes_sub_title={m['yes_sub_title']!r}  result={m['result']}")
    m = next(iter(ev["markets"].values()))
    print("  time fields on a market:")
    for k in ["created_time", "open_time", "close_time", "expected_expiration_time",
              "expiration_time", "settlement_ts"]:
        print(f"    {k:26s} {m.get(k)}")
    print(f"  all market fields: {sorted(m)}")


# ---------------------------------------------------------------- main

def main() -> None:
    events, cutoff = list_season_events()
    summarize_events(events)
    summarize_open_lead(events)
    samples = summarize_samples(events)
    show_event_structure(events)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "probe_summary.json"
    out.write_text(json.dumps({
        "cutoff": cutoff,
        "events": [{"event_ticker": e["event_ticker"], "game_date": str(e["game_date"]),
                    "phase": e["phase"], "markets": list(e["markets"].values())}
                   for e in sorted(events.values(), key=lambda e: e["event_ticker"])],
        "samples": samples,
    }, indent=1, default=str), encoding="utf-8")
    print(f"\nSaved event list and sample results to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

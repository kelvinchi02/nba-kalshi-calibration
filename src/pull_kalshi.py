"""Stage 4: pull Kalshi trades and 1-minute candlesticks for every matched game market.

Two phases per market, both from the /historical/* endpoints (all 2025-26 data is older
than Kalshi's historical cutoff):

  window   trades and 1-minute candles over [official tipoff - 30h, official tipoff]
  pregame  trades from market open up to the window start, so Stage 6 can compute full
           pregame volume (market open to tipoff)

Tipoff is tipoff_utc from data/processed/game_map.csv (Stage 5), never Kalshi's
expiration-based estimate.

The raw cache keeps exactly what the API returned. max_ts and end_ts are inclusive, so the
window cache can hold a trade or candle stamped at tipoff, and the two phases can share
trades stamped in the second at the window start. Stage 6 enforces "strictly before" and
de-duplicates by trade_id. Markets that opened after the window start (late listings) need
no pregame request and get an empty pregame record.

Cache: one gzipped JSON file per market and phase, written atomically:
  data/raw/kalshi/markets/<market_ticker>.json.gz   window trades + candles + market metadata
  data/raw/kalshi/pregame/<market_ticker>.json.gz   trades from market open to window start
Re-running skips markets already cached (resumable).
Logs: logs/pull_kalshi.log (progress) and logs/pull_kalshi_failures.csv (failed markets).

Usage:
    python src/pull_kalshi.py                   # both phases, all matched games
    python src/pull_kalshi.py --phase pregame   # one phase only
    python src/pull_kalshi.py --limit 4         # first 4 markets, for testing
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from csv_io import read_csv
from kalshi_client import KalshiClient

SERIES = "KXNBAGAME"
WINDOW_HOURS = 30
REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_MAP_PATH = REPO_ROOT / "data" / "processed" / "game_map.csv"
MARKETS_CACHE = REPO_ROOT / "data" / "raw" / f"kalshi_markets_{SERIES}.json"
CACHE_DIRS = {
    "window": REPO_ROOT / "data" / "raw" / "kalshi" / "markets",
    "pregame": REPO_ROOT / "data" / "raw" / "kalshi" / "pregame",
}
LOG_DIR = REPO_ROOT / "logs"
FAILURES_PATH = LOG_DIR / "pull_kalshi_failures.csv"

log = logging.getLogger("pull_kalshi")


def build_jobs() -> list[dict]:
    """One job per team market of every matched game."""
    games = read_csv(GAME_MAP_PATH)
    games = games[games["matched"]]
    jobs = []
    for g in games.itertuples(index=False):
        tip = int(g.tipoff_utc.timestamp())
        for side, ticker in [("home", g.home_market), ("away", g.away_market)]:
            jobs.append({
                "ticker": ticker,
                "side": side,
                "team": g.home_team if side == "home" else g.away_team,
                "game_id": g.game_id,
                "event_ticker": g.event_ticker,
                "tipoff_utc": g.tipoff_utc.isoformat(),
                "window_start_ts": tip - WINDOW_HOURS * 3600,
                "window_end_ts": tip,
            })
    return jobs


# ---------------------------------------------------------------- cache

def cache_path(phase: str, ticker: str) -> Path:
    return CACHE_DIRS[phase] / f"{ticker}.json.gz"


def read_cache(phase: str, ticker: str) -> dict:
    with gzip.open(cache_path(phase, ticker), "rt", encoding="utf-8") as f:
        return json.load(f)


def is_cached(phase: str, ticker: str) -> bool:
    if not cache_path(phase, ticker).exists():
        return False
    try:
        return read_cache(phase, ticker).get("complete") is True
    except (json.JSONDecodeError, OSError, EOFError):
        return False  # partial or corrupt file: pull again


def write_cache(phase: str, ticker: str, record: dict) -> None:
    path = cache_path(phase, ticker)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(record, f)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache file


# ---------------------------------------------------------------- pulls

def pull_window(client: KalshiClient, job: dict, market_meta: dict) -> None:
    """Trades and 1-minute candles from tipoff - 30h to tipoff."""
    t0, t1 = job["window_start_ts"], job["window_end_ts"]
    trades = client.historical_trades(job["ticker"], min_ts=t0, max_ts=t1)
    candles = client.historical_candlesticks(job["ticker"], start_ts=t0, end_ts=t1,
                                             period_interval=1)
    write_cache("window", job["ticker"], {
        **job,
        "window_hours": WINDOW_HOURS,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "n_trades": len(trades),
        "n_candles": len(candles),
        "market": market_meta,
        "trades": trades,
        "candlesticks": candles,
        "complete": True,
    })


def pull_pregame(client: KalshiClient, job: dict, market_meta: dict) -> None:
    """Trades from market open up to the window start (both ends inclusive)."""
    open_ts = int(pd.Timestamp(market_meta["open_time"]).timestamp())
    end_ts = job["window_start_ts"]
    if open_ts <= end_ts:
        trades = client.historical_trades(job["ticker"], min_ts=open_ts, max_ts=end_ts)
    else:
        trades = []  # late listing: the market opened inside the 30h window
    write_cache("pregame", job["ticker"], {
        **job,
        "pregame_start_ts": open_ts,
        "pregame_end_ts": end_ts,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "n_trades": len(trades),
        "trades": trades,
        "complete": True,
    })


PULLERS = {"window": pull_window, "pregame": pull_pregame}


# ---------------------------------------------------------------- main

def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in [logging.FileHandler(LOG_DIR / "pull_kalshi.log", encoding="utf-8"),
                    logging.StreamHandler()]:
        handler.setFormatter(fmt)
        root.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["window", "pregame", "both"], default="both")
    parser.add_argument("--limit", type=int, help="only pull the first N markets (testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel workers sharing one rate limit (default 4)")
    args = parser.parse_args()
    setup_logging()

    client = KalshiClient(max_requests_per_sec=10)
    cutoff = pd.Timestamp(client.historical_cutoff()["trades_created_ts"])
    jobs = build_jobs()
    latest_end = max(pd.Timestamp(j["tipoff_utc"]) for j in jobs)
    if latest_end >= cutoff:
        raise SystemExit(f"Some windows end after the historical cutoff {cutoff}; "
                         "live endpoints would be needed and are not implemented.")

    meta = {m["ticker"]: m for m in json.loads(MARKETS_CACHE.read_text(encoding="utf-8"))}
    if args.limit:
        jobs = jobs[: args.limit]
    phases = ["window", "pregame"] if args.phase == "both" else [args.phase]

    failures = []
    for phase in phases:
        CACHE_DIRS[phase].mkdir(parents=True, exist_ok=True)
        todo = [j for j in jobs if not is_cached(phase, j["ticker"])]
        log.info("[%s] markets: %d total, %d already cached, %d to pull (signing=%s)",
                 phase, len(jobs), len(jobs) - len(todo), len(todo), client.signing)
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(PULLERS[phase], client, j, meta[j["ticker"]]): j
                       for j in todo}
            for fut in as_completed(futures):
                job = futures[fut]
                done += 1
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - record and continue
                    failures.append({"phase": phase, "ticker": job["ticker"],
                                     "game_id": job["game_id"], "error": repr(exc)[:500]})
                    log.error("FAILED [%s] %s: %r", phase, job["ticker"], exc)
                if done % 50 == 0 or done == len(todo):
                    log.info("[%s] progress %d/%d, failures %d", phase, done, len(todo),
                             len(failures))
        cached = sum(is_cached(phase, j["ticker"]) for j in jobs)
        log.info("[%s] finished: %d/%d markets cached", phase, cached, len(jobs))

    with FAILURES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "ticker", "game_id", "error"])
        writer.writeheader()
        writer.writerows(failures)
    log.info("%d failures this run (see %s)", len(failures), FAILURES_PATH.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()

"""Stage 4: pull Kalshi trades and 1-minute candlesticks for every matched game market.

Window per market: [official NBA tipoff - 30h, official NBA tipoff], using tipoff_utc from
data/processed/game_map.parquet (Stage 5), never Kalshi's expiration-based estimate.
All 2025-26 data is older than Kalshi's historical cutoff, so /historical/* endpoints are used.

The raw cache keeps exactly what the API returned for the window. max_ts and end_ts are
inclusive, so the cache can hold a trade or candle stamped at tipoff. Stage 6 enforces
"strictly before the snapshot".

Cache: one gzipped JSON file per market at data/raw/kalshi/markets/<market_ticker>.json.gz
(about 11x smaller than plain JSON), written atomically. Re-running skips markets already cached (resumable).
Logs: logs/pull_kalshi.log (progress) and logs/pull_kalshi_failures.csv (failed markets).

Usage:
    python src/pull_kalshi.py                # all matched games
    python src/pull_kalshi.py --limit 4      # first 4 markets, for testing
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

from kalshi_client import KalshiClient

SERIES = "KXNBAGAME"
WINDOW_HOURS = 30
REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_MAP_PATH = REPO_ROOT / "data" / "processed" / "game_map.parquet"
MARKETS_CACHE = REPO_ROOT / "data" / "raw" / f"kalshi_markets_{SERIES}.json"
CACHE_DIR = REPO_ROOT / "data" / "raw" / "kalshi" / "markets"
LOG_DIR = REPO_ROOT / "logs"
FAILURES_PATH = LOG_DIR / "pull_kalshi_failures.csv"

log = logging.getLogger("pull_kalshi")


def build_jobs() -> list[dict]:
    """One job per team market of every matched game."""
    games = pd.read_parquet(GAME_MAP_PATH)
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


def cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.json.gz"


def read_cache(ticker: str) -> dict:
    with gzip.open(cache_path(ticker), "rt", encoding="utf-8") as f:
        return json.load(f)


def is_cached(ticker: str) -> bool:
    path = cache_path(ticker)
    if not path.exists():
        return False
    try:
        return read_cache(ticker).get("complete") is True
    except (json.JSONDecodeError, OSError, EOFError):
        return False  # partial or corrupt file: pull again


def pull_market(client: KalshiClient, job: dict, market_meta: dict) -> dict:
    t0, t1 = job["window_start_ts"], job["window_end_ts"]
    trades = client.historical_trades(job["ticker"], min_ts=t0, max_ts=t1)
    candles = client.historical_candlesticks(job["ticker"], start_ts=t0, end_ts=t1,
                                             period_interval=1)
    record = {
        **job,
        "window_hours": WINDOW_HOURS,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "n_trades": len(trades),
        "n_candles": len(candles),
        "market": market_meta,
        "trades": trades,
        "candlesticks": candles,
        "complete": True,
    }
    path = cache_path(job["ticker"])
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(record, f)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache file
    return record


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

    meta = {m["ticker"]: m for m in json.loads(MARKETS_CACHE.read_text())}
    if args.limit:
        jobs = jobs[: args.limit]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not is_cached(j["ticker"])]
    log.info("markets: %d total, %d already cached, %d to pull (signing=%s)",
             len(jobs), len(jobs) - len(todo), len(todo), client.signing)

    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(pull_market, client, j, meta[j["ticker"]]): j for j in todo}
        for fut in as_completed(futures):
            job = futures[fut]
            done += 1
            try:
                rec = fut.result()
                log.debug("ok %s trades=%d candles=%d", job["ticker"], rec["n_trades"],
                          rec["n_candles"])
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append({"ticker": job["ticker"], "game_id": job["game_id"],
                                 "error": repr(exc)[:500]})
                log.error("FAILED %s: %r", job["ticker"], exc)
            if done % 50 == 0 or done == len(todo):
                log.info("progress %d/%d, failures %d", done, len(todo), len(failures))

    with FAILURES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "game_id", "error"])
        writer.writeheader()
        writer.writerows(failures)
    cached = sum(is_cached(j["ticker"]) for j in jobs)
    log.info("finished: %d/%d markets cached, %d failed this run (see %s)",
             cached, len(jobs), len(failures), FAILURES_PATH.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()

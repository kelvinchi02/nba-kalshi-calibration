# nba-kalshi-calibration

Data pipeline for the STSCI 4981/5981 final project. The research question: are Kalshi NBA game prices before tipoff well calibrated, and are popular teams overpriced?

This repo collects and cleans the data only. It does not compute calibration results.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows path; .venv/bin/python elsewhere
```

### Secrets (optional)

```bash
cp .env.example .env
python scripts/check_secrets.py   # reports what is set, never prints values
```

An API key is not required. Every Kalshi endpoint used here is public: series, events, markets, `/historical/markets`, `/historical/trades` and `/historical/markets/{ticker}/candlesticks`.

If you add a key, put the key ID in `.env` and the private key at `secrets/kalshi_private_key.pem`. Both are gitignored. `src/kalshi_client.py` signs requests only when both are present.

## Run order

Run each command from the repo root with the venv's Python. Stage 5 runs before Stage 4, because the pull windows are built from the official NBA tipoff, and that tipoff comes from the match.

| Step | Command | What it does |
|---|---|---|
| 2 | `python scripts/probe_kalshi.py` | Feasibility probe: event counts, volume and 30h coverage for sample games. Writes `data/interim/probe_summary.json`. |
| 3 | `python src/nba_schedule.py` | Official 2025-26 schedule and results from stats.nba.com via `nba_api`. |
| 5 | `python src/match_games.py` | Maps Kalshi events to NBA game IDs. Downloads the Kalshi market list once (`--refresh` re-downloads it). |
| 4 | `python src/pull_kalshi.py` | Trades and 1-minute candles for each market, from 30h before tipoff to tipoff. Resumable: cached markets are skipped. |
| 6 | `python src/build_snapshots.py` | Snapshot table at 24h, 6h, 1h and 0h before tipoff. |
| 7 | `python src/quality_report.py` | Data quality report. |
| tests | `python -m pytest -q` | Time-window (no leakage), VWAP, spread and normalization tests. |

Whole-season runtimes on a home connection:
- Stage 3: a few seconds.
- Stage 4 (pull): about an hour for 2,460 markets at 10 requests/s.
- Stage 6: a few minutes.

## Output files

### `data/processed/` (committed)

**`nba_games_2025_26.parquet`**: 1,230 regular-season games.
- Identifiers and time: `game_id`, `game_date` (US Eastern) and `tipoff_utc`, the official scheduled tipoff.
- Teams and result: `away_team`, `home_team` (NBA tricodes), final scores, `home_win` (1/0) and `status` (Final, Final/OT, ...).
- Context: `game_label` (NBA Cup, Global Games, ...), `game_subtype` and `arena_city`.
- Postponed games appear only at their replay date.

**`team_code_map.csv`**: Kalshi team code to NBA tricode, with the names Kalshi uses for each team.

**`game_map.parquet`**: one row per NBA game, with its Kalshi event and the two market tickers (`home_market`, `away_market`). It also holds:
- `match_type`: `exact`, or `date_shift_1d` for a match within one day.
- `kalshi_tipoff_estimate`: Kalshi's listed time, `expiration_time` minus 14 days. Kept for reference only.
- `tipoff_shift_hours`: official tipoff minus that estimate.
- `hours_open_before_tipoff`: how long the market was open before tipoff.
- `kalshi_home_win`: Kalshi's settlement, a sanity check against the NBA result.
- Flags:
  - `postponed_window`: the game moved to a different day after the market opened (GSW@MIN, Jan 24 to Jan 25). Excluded from the main sample.
  - `tipoff_moved`: the official tipoff is 1 to 12 hours from Kalshi's listed time (5 games). Kept in the main sample; check these separately.
  - `main_sample`: matched and not `postponed_window`.

**`match_report.md`**: matching results.
- Matched counts.
- Unmatched games and events on both sides.
- The 3 dropped postponed original events (settled scalar): Jan 8 MIA@CHI, Jan 25 DAL@MIL and Jan 25 DEN@MEM. Their replays are matched.
- Preseason, play-in and playoff events, which are outside the sample.

**`snapshots.parquet`**: one row per game per horizon (`horizon_h` in 24, 6, 1, 0), with `snapshot_utc = tipoff_utc - horizon_h`. Each row carries the game's teams, `home_win` and flags. Per team market (columns prefixed `home_` / `away_`):
- `mid`: the main price. The bid/ask midpoint from the last 1-minute candle that ends strictly before the snapshot.
  - `bid`, `ask`: that candle's closing quote.
  - `spread`: ask minus bid of that candle, in dollars (0.01 = 1¢). Kept even for one-sided books.
  - `quote_age_min`: minutes since that candle ended.
  - `mid_carried_forward`: true when the quote is older than 1 minute. Kalshi only returns candles for minutes with activity, and the quote did not change in the minutes it skips.
  - `one_sided`: true when the bid is 0 or the ask is 1.00. The midpoint is left missing.
- `last`, `last_trade_age_min`: the last trade strictly before the snapshot (backup 1).
- `vwap_10m`: the volume-weighted average price over the 10 minutes before the snapshot (backup 2).
- `n_trades_10m`, `contracts_10m`: activity in that 10-minute window.
- `cum_contracts`: contracts traded from the start of the pulled window (tipoff minus 30h) up to the snapshot. This is not volume since the market opened.

Per row, `max_spread` is the wider of the two sides' spreads. A wide spread makes the midpoint less reliable; filter on it, for example `max_spread > 0.05`.

Normalized, per price type:
- `p_home_<type>` = home / (home + away), where `<type>` is `mid`, `last` or `vwap_10m`.
- `sum_<type>` is the raw home + away sum, before normalization.
- `p_home` is the main estimate: the midpoint pair, else last trade, else VWAP. `price_source` says which, and both sides always use the same source.
- `market_open_before_snapshot` is false for the 24h snapshot of the 7 games Kalshi listed less than 24h before tipoff. Those values stay missing.

**`quality_report.md`**: contents:
- match and cache coverage;
- missing snapshots per horizon;
- quote age;
- bid/ask spread distribution by horizon, with the count of rows above 5¢;
- low-volume games;
- pre-normalization sums.

### `data/raw/` (gitignored, re-pullable)

- `kalshi_markets_KXNBAGAME.json`: the Kalshi market list.
- `kalshi/markets/<ticker>.json.gz`: one gzipped file per market, holding the raw trades and candlesticks returned for its window, plus market metadata.
- `nba_schedule_2025_26.json`: the raw stats.nba.com response.

### `logs/` (gitignored)

- `pull_kalshi.log`: pull progress.
- `pull_kalshi_failures.csv`: markets that failed in the last run.

## Rules this pipeline follows

- Snapshot times always use the official NBA tipoff, never Kalshi's expiration-based estimate.
- Only data strictly before the snapshot time is used. Missing values stay missing, and carried-forward quotes are flagged.
- Secrets, keys and raw data are never committed.

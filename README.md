# nba-kalshi-calibration

Data pipeline for the STSCI 4981/5981 final project. The research question: are Kalshi NBA game prices before tipoff well calibrated, and are popular teams overpriced?

Stages 2 to 7 collect and clean the data. The data used for analysis is frozen at git tag `data-v1`. Stage 8 runs the analysis pre-registered in [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md), which was committed before any result was computed. So far that covers calibration at tipoff (horizon 0) only.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows path; .venv/bin/python elsewhere
```

Set `PYTHONUTF8=1` in every shell that runs the pipeline. It makes Python use UTF-8 for console output and for any library that opens files without naming an encoding. Without it, Windows falls back to the system code page (for example GBK), which cannot encode characters such as "¢". The code already passes `encoding="utf-8"` on every file read and write.

```bash
export PYTHONUTF8=1          # bash / Git Bash
$env:PYTHONUTF8 = "1"        # PowerShell
```

### Secrets (optional)

```bash
cp .env.example .env
python scripts/check_secrets.py   # reports what is set, never prints values
```

An API key is not required. Every Kalshi endpoint used here is public: series, events, markets, `/historical/markets`, `/historical/trades` and `/historical/markets/{ticker}/candlesticks`.

If you add a key, put the key ID in `.env` and the private key at `secrets/kalshi_private_key.pem`. Both are gitignored. `src/kalshi_client.py` signs requests only when both are present.

## Run order

Run each command from the repo root with the venv's Python and `PYTHONUTF8=1`. Stage 5 runs before Stage 4, because the pull windows are built from the official NBA tipoff, and that tipoff comes from the match.

| Order | Stage | Command | What it does |
|---|---|---|---|
| (optional) | 2 | `python scripts/probe_kalshi.py` | Feasibility probe: event counts, volume and 30h coverage for sample games. Writes `data/interim/probe_summary.json`. Not needed to rebuild. |
| 1 | 3 | `python src/nba_schedule.py` | Official 2025-26 schedule and results from stats.nba.com via `nba_api`. `--cached` rebuilds from the saved raw response without a request. |
| 2 | 5 | `python src/match_games.py` | Maps Kalshi events to NBA game IDs. Downloads the Kalshi market list once (`--refresh` re-downloads it). |
| 3 | 4 | `python src/pull_kalshi.py` | Two resumable phases per market: `window` (trades + 1-minute candles, tipoff minus 30h to tipoff) and `pregame` (trades from market open to the window start). Cached markets are skipped. `--phase window\|pregame` runs one phase. |
| 4 | 6 | `python src/build_snapshots.py` | Snapshot table at 24h, 6h, 1h and 0h before tipoff. |
| 5 | 7 | `python src/quality_report.py` | Data quality report. |
| 6 | 8 | `python src/calibration.py` | Calibration at horizon 0, exactly as in `ANALYSIS_PLAN.md`: binned calibration plot, Brier score, logistic recalibration, LR test and date-block bootstrap. Writes `results/`. |
| 7 | 8 | `python src/eda.py` | EDA figures: one example game's 30-hour price path from the raw cache, and average pregame volume by team. `--event <ticker>` plots a different game. |
| 8 | tests | `python -m pytest -q` | Tests for: no-leakage time windows, VWAP, spread, normalization, pregame de-duplication, CSV round trip, CSV-vs-parquet equality, and the calibration statistics (Wilson, Brier, LR test, bootstrap) on synthetic data. |

With the raw cache present, rebuilding from Stage 3 makes no network calls apart from the Kalshi cutoff check in Stage 4:

```bash
python src/nba_schedule.py --cached && python src/match_games.py && python src/pull_kalshi.py \
  && python src/build_snapshots.py && python src/quality_report.py && python src/calibration.py \
  && python src/eda.py && python -m pytest -q
```

Whole-season runtimes on a home connection:
- Stage 3: a few seconds.
- Stage 4 from scratch: about 20 minutes for the window phase and 5 minutes for the pregame phase (2,460 markets at 10 requests/s).
- Stage 6: under a minute.
- Stage 8: about 10 seconds, most of it the 2,000 bootstrap replicates.

## Repo layout

```
nba-kalshi-calibration/
  src/
    nba_schedule.py      Stage 3: NBA schedule and results
    match_games.py       Stage 5: Kalshi event -> NBA game mapping
    pull_kalshi.py       Stage 4: Kalshi trades and candles (window + pregame)
    build_snapshots.py   Stage 6: 24h/6h/1h/0h snapshots
    quality_report.py    Stage 7: data quality report
    calibration.py       Stage 8: calibration at horizon 0 (ANALYSIS_PLAN.md)
    eda.py               Stage 8: EDA figures (example price path, pregame volume by team)
    kalshi_client.py     shared: Kalshi client (rate limit, retries, pagination, optional signing)
    csv_io.py            shared: CSV read/write conventions for data/processed and results
    plot_style.py        shared: figure colors and matplotlib style
  scripts/
    check_secrets.py     reports which .env values are set, never prints them
    probe_kalshi.py      Stage 2 feasibility probe
  tests/
    conftest.py          puts src/ and tests/ on the import path
    fingerprint.py       type-aware DataFrame fingerprint helper
    test_snapshots.py    time windows, VWAP, spread, normalization, pregame volume
    test_csv_outputs.py  CSV round trip and equality with the replaced parquet files
    test_calibration.py  Stage 8 statistics on known values and synthetic data
    fixtures/processed_fingerprint.json
  results/               committed analysis tables (CSV) and figures (PNG)
  data/
    processed/           committed outputs (below)
    raw/                 gitignored raw cache (below)
    interim/             gitignored, probe output only
  logs/                  gitignored pull logs
  secrets/               gitignored, only if you add a Kalshi key
  .env.example
  .gitignore
  ANALYSIS_PLAN.md       pre-registered analysis plan
  requirements.txt
  README.md
```

## Output files

### CSV conventions (`data/processed/*.csv`)

- Timestamps are ISO 8601 in UTC with a Z suffix, for example `2026-01-25T22:30:00Z`. Dates are `YYYY-MM-DD`.
- Booleans are `1` or `0`. Missing values are empty cells.
- Files are UTF-8, with no index column.
- Read them with `csv_io.read_csv`, which restores the column types: zero-padded `game_id` as text, timestamps as UTC, and booleans as bool.
- Plain `pandas.read_csv` also works, but you need `dtype={"game_id": str}` and you must parse the timestamps yourself.

### `data/processed/` (committed)

**`nba_games_2025_26.csv`**: 1,230 regular-season games.
- Identifiers and time: `game_id`, `game_date` (US Eastern) and `tipoff_utc`, the official scheduled tipoff.
- Teams and result: `away_team`, `home_team` (NBA tricodes), final scores, `home_win` (1/0) and `status` (Final, Final/OT, ...).
- Context: `game_label` (NBA Cup, Global Games, ...), `game_subtype` and `arena_city`.
- Postponed games appear only at their replay date.

**`team_code_map.csv`**: Kalshi team code to NBA tricode, with the names Kalshi uses for each team.

**`game_map.csv`**: one row per NBA game, with its Kalshi event and the two market tickers (`home_market`, `away_market`). It also holds:
- `match_type`: `exact`, or `date_shift_1d` for a match within one day.
- `kalshi_tipoff_estimate`: Kalshi's listed time, `expiration_time` minus 14 days. Kept for reference only.
- `tipoff_shift_hours`: official tipoff minus that estimate.
- `market_open_time` and `hours_open_before_tipoff`: when the market opened, and how long before tipoff that was.
- `kalshi_volume`: lifetime contracts for both markets, including in-game trading.
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

**`snapshots.csv`**: one row per game per horizon (`horizon_h` in 24, 6, 1, 0), with `snapshot_utc = tipoff_utc - horizon_h`. Each row carries the game's teams, `home_win` and flags. Per team market (columns prefixed `home_` / `away_`):
- `mid`: the main price. The bid/ask midpoint from the last 1-minute candle that ends strictly before the snapshot.
  - `bid`, `ask`: that candle's closing quote.
  - `spread`: ask minus bid of that candle, in dollars (0.01 = 1¢). Kept even for one-sided books.
  - `quote_age_min`: minutes since that candle ended.
  - `mid_carried_forward`: 1 when the quote is older than 1 minute. Kalshi only returns candles for minutes with activity, and the quote did not change in the minutes it skips.
  - `one_sided`: 1 when the bid is 0 or the ask is 1.00. The midpoint is left missing.
- `last`, `last_trade_age_min`: the last trade strictly before the snapshot (backup 1).
- `vwap_10m`: the volume-weighted average price over the 10 minutes before the snapshot (backup 2).
- `n_trades_10m`, `contracts_10m`: activity in that 10-minute window.
- `cum_contracts`: contracts traded from the start of the pulled window (tipoff minus 30h) up to the snapshot.
- `pregame_contracts`: contracts traded from market open to the official tipoff, strictly before tipoff. This is the full pregame volume, the same value at every horizon. The window and pregame pulls are combined and de-duplicated by `trade_id`.

Per row, `max_spread` is the wider of the two sides' spreads. A wide spread makes the midpoint less reliable; filter on it, for example `max_spread > 0.05`.

Normalized, per price type:
- `p_home_<type>` = home / (home + away), where `<type>` is `mid`, `last` or `vwap_10m`.
- `sum_<type>` is the raw home + away sum, before normalization.
- `p_home` is the main estimate: the midpoint pair, else last trade, else VWAP. `price_source` says which, and both sides always use the same source.

Row-level flags:
- `market_open_before_snapshot` is 0 for the 24h snapshot of the 7 games Kalshi listed less than 24h before tipoff. Those values stay missing.
- `cache_missing` is 1 if any raw cache file for the game was absent.

**`quality_report.md`**: contents:
- match and cache coverage;
- missing snapshots per horizon;
- quote age;
- bid/ask spread distribution by horizon, with the count of rows above 5¢;
- low-volume games (30h window);
- full pregame volume, with consistency checks;
- pre-normalization sums.

### `results/` (committed, Stage 8)

All of these use the main sample at horizon 0: 1,229 games, price `p_home`, outcome `home_win`. Section numbers refer to `ANALYSIS_PLAN.md`.

- `calibration_bins.csv`: the 10 equal-count bins (section 2.1). Columns: n, p range, mean p, home wins, win rate, 95% Wilson interval, and the bootstrap interval with bin edges fixed (`boot_lo`, `boot_hi`).
- `calibration_plot.png`: observed win rate against mean price per bin, with the 45° line.
- `brier.csv`: Brier score, constant baseline, difference and skill score (section 2.2).
- `logistic.csv`: intercept `a` and slope `b` of home_win on logit(p_home), with SEs clustered by home team and Wald tests of a = 0 and b = 1 (section 2.3).
- `lr_test.csv`: likelihood ratio test of a = 0 and b = 1 jointly (section 2.4).
- `bootstrap_summary.csv`: date-block bootstrap with B = 2,000 and seed 4981. Gives percentile intervals for every statistic above (section 3).

EDA files are descriptive and use no outcomes:
- `eda_price_path.png`: one example game. It shows both markets' bid/ask midpoint and band over the 30 hours before tipoff, the four snapshot times, and contracts traded per 30 minutes.
  - The default game is chosen by rule: among games with above-median pregame volume, the one whose `p_home` moved most from 24h to tipoff.
- `eda_team_pregame_volume.csv` / `.png`: each team's average `pregame_contracts` on its own market per game, ranked. This is the popularity measure from section 4.

### `data/raw/` (gitignored, needed to rebuild without re-pulling)

- `kalshi_markets_KXNBAGAME.json`: the Kalshi market list.
- `kalshi/markets/<ticker>.json.gz`: one gzipped file per market for the window phase, holding the raw trades and candlesticks returned for its window, plus market metadata.
- `kalshi/pregame/<ticker>.json.gz`: one gzipped file per market for the pregame phase, holding the raw trades from market open to the window start.
- `nba_schedule_2025_26.json`: the raw stats.nba.com response, used by `--cached`.

### `logs/` (gitignored)

- `pull_kalshi.log`: pull progress.
- `pull_kalshi_failures.csv`: markets that failed in the last run, by phase.

## Rules this pipeline follows

- Snapshot times always use the official NBA tipoff, never Kalshi's expiration-based estimate.
- Only data strictly before the snapshot time is used. Missing values stay missing, and carried-forward quotes are flagged.
- Every file read and write passes `encoding="utf-8"`.
- Secrets, keys and raw data are never committed.

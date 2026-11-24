# nba-kalshi-calibration

Data pipeline for the STSCI 4981/5981 final project. The research question: are Kalshi NBA game prices before tipoff well calibrated, and are popular teams overpriced?

Stages 2 to 7 collect and clean the data, as scripts. The data used for analysis is frozen at git tag `data-v1`. The analysis runs in the notebooks in [notebooks/](notebooks/) and follows [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md), which is committed before any result is computed. Stage 8 covers the EDA and calibration at tipoff (horizon 0). Stage 9 covers the popularity bias test, the comparison of horizons and a trading strategy backtest. Two exploratory checks were added after the Stage 9 results were seen. They are labeled as exploratory in the plan and in the notebooks. Stage 10 adds four extensions: line movement, information arrival (with an injury report event study), subgroups, and a replication on the start of the 2026-27 season. The 2026-27 data is frozen at git tag `data-v2`. Two more exploratory checks were added after the Stage 10 results were seen (injury events by the player's previous status, and the calibration slope by month).

## Setup

With [uv](https://docs.astral.sh/uv/), which installs the exact versions in `uv.lock` (Python 3.12):

```bash
uv sync        # creates .venv from pyproject.toml and uv.lock
```

Without uv, with any Python 3.12 or later:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows path; .venv/bin/python elsewhere
```

`requirements.txt` lists the same packages as `pyproject.toml`, without pinned versions. A test checks that the two lists match. To add a package, run `uv add <name>` and add the same name to `requirements.txt`.

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

An API key is not required. Every Kalshi endpoint used here is public: series, events, markets, `/historical/markets`, `/historical/trades` and `/historical/markets/{ticker}/candlesticks`, and for the 2026-27 season the live versions `/markets`, `/markets/trades` and `/series/{series}/markets/{ticker}/candlesticks`.

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
| 6 | 8 | `python src/export_price_path.py` | Saves one example game's 30-hour price path (quotes and volume per minute) to `data/processed/example_price_path.csv`. This is the last step that needs the raw cache. `--event <ticker>` exports a different game. |
| 7 | 9 | `python src/metro_population.py` | Metro area population per team, from the US Census Bureau (Vintage 2025) and Statistics Canada (Toronto). Writes `data/processed/team_metro_population.csv`. The two source files are downloaded once to `data/raw/population/`. `--refresh` downloads them again. |
| 8 | 10 | `python src/build_price_paths.py` | Every game's price path on a 5-minute grid over the 30 hours before tipoff. Writes `data/processed/price_paths.csv.gz`. Needs the raw cache. |
| 9 | 10 | `uv run --project envs/injuries python envs/injuries/fetch_injury_reports.py` | Downloads and parses every official NBA injury report snapshot of 2025-26. Runs in its own environment (see [The injury report environment](#the-injury-report-environment)). Needs Java. |
| 10 | 10 | `python src/player_minutes.py` | Minutes per player per game, from stats.nba.com via `nba_api`. `--cached` rebuilds without a request. |
| 11 | 10 | `python src/injury_events.py` | Turns the parsed injury reports into one row per status change. Writes `data/processed/injury_status_changes_2025_26.csv`. |
| 12 | 10 | the 2026-27 season (below) | Steps 1 to 5 again with `--season 2026-27`. |
| 13 | 8, 9, 10 | `python scripts/run_notebooks.py` | Runs every notebook in `notebooks/` top to bottom, in name order, and saves each with its outputs. Writes `results/`. See [Notebooks](#notebooks). |
| 14 | tests | `python -m pytest -q` | Tests for: no-leakage time windows, VWAP, spread, normalization, pregame de-duplication, CSV round trip, CSV-vs-parquet equality, the statistics in `stats_tools.py` (Wilson, Brier, clustered logit, LR test, block bootstrap, bootstrap p value, joint Wald test, Holm) on synthetic data, the EDA and calibration tables, the Stage 9 code (popularity gaps and model, horizon comparison, strategy rule and fees) and the Stage 10 code (season settings, price paths, line movement, subgroups, replication, injury events, event study, the common tests table) on hand-made data, the notebook runner, and that `requirements.txt` matches `pyproject.toml`. |

With the raw cache present, rebuilding from Stage 3 makes no network calls apart from the Kalshi cutoff check in Stage 4:

```bash
python src/nba_schedule.py --cached && python src/match_games.py && python src/pull_kalshi.py \
  && python src/build_snapshots.py && python src/quality_report.py \
  && python src/export_price_path.py && python src/metro_population.py \
  && python scripts/run_notebooks.py && python -m pytest -q
```

The notebooks and the tests need only the committed CSVs in `data/processed/`. On a clone without the raw cache, skip to `python scripts/run_notebooks.py`.

### The 2026-27 season

The pipeline scripts take `--season`. Everything that differs between seasons is in `src/seasons.py`: file names, the date range, and which Kalshi endpoints hold the data. 2025-26 is the default and keeps its file names. Another season carries the season in its file names (`game_map_2026_27.csv`, `snapshots_2026_27.csv`, ...).

```bash
python src/nba_schedule.py --season 2026-27 --through 2026-11-22   # final games up to that date
python src/match_games.py --season 2026-27 --refresh
python src/pull_kalshi.py --season 2026-27
python src/build_snapshots.py --season 2026-27
python src/quality_report.py --season 2026-27
```

A season still being played needs `--through`. The games table then holds the final games up to that US Eastern date, and that date ends the season for every later step. The committed 2026-27 files were pulled once, on November 23, 2026, with the cutoff November 22 (248 games). They are frozen at tag `data-v2` and are not extended.

### The injury report environment

`nbainjuries`, the package that reads the NBA's injury report files, needs pandas below 3.0, and this project uses pandas 3. So the fetch runs in its own small uv project in `envs/injuries/` (its own `pyproject.toml`, `uv.lock` and `.venv`). It only writes files. It needs a Java runtime on the PATH, because the PDF tables are read with tabula.

```bash
uv run --project envs/injuries python envs/injuries/fetch_injury_reports.py              # download, parse, combine
uv run --project envs/injuries python envs/injuries/fetch_injury_reports.py --step parse # one step
```

The download is slow on purpose (two workers with a pause): the NBA's delivery network answers a burst of requests with temporary 403 blocks. The script tells those from a file that does not exist and never records a blocked file as missing. The PDFs (about 1 GB) and the parsed rows stay in `data/raw/injury_reports/`, which is not in git.

Whole-season runtimes on a home connection:
- Stage 3: a few seconds.
- Stage 4 from scratch: about 20 minutes for the window phase and 5 minutes for the pregame phase (2,460 markets at 10 requests/s).
- Stage 6: under a minute.
- Price paths: about 3 minutes.
- Injury reports: about 25 minutes to download and 15 minutes to parse.
- 2026-27 pull: about 15 minutes for 496 markets (the live trade endpoint is rate-limited more tightly).
- Notebooks: about a minute in total, most of it the bootstrap replicates.

## Notebooks

The analysis lives in `notebooks/`. Read and run them in this order:

| Notebook | Question | Plan section |
|---|---|---|
| `01_eda.ipynb` | What does the data look like? Pregame volume by team, one example price path, and the distribution of the tipoff price. No outcomes are used. | 4 and 6 |
| `02_calibration.ipynb` | Are prices at tipoff well calibrated? Calibration bins, Brier score, logistic recalibration, LR test and date-block bootstrap. | 1 to 3 |
| `03_popularity.ipynb` | Are popular teams overpriced? Two popularity measures (trading volume and metro population), a logistic model with the popularity gap, and a date-block bootstrap. | 9.1 |
| `04_horizons.ipynb` | How do the 24h, 6h and 1h prices compare with tipoff? Brier score and calibration at each horizon, paired bootstrap of the Brier differences, and a tight-spread check. | 9.2 |
| `05_strategy.ipynb` | Does buying the less popular team at tipoff make money after fees? One fixed rule, a benchmark without the filter, and the same rule with metro population. | 9.3 |

| `06_line_movement.ipynb` | Does the price move from 24h before tipoff to tipoff hold information the tipoff price missed? | 10.3 |
| `07_information_arrival.ipynb` | When do prices move in the 24 hours before tipoff? How does a team's price move when the injury report rules a key player Out? | 10.4 |
| `08_subgroups.ipynb` | Is the tipoff price differently calibrated by trading volume, season half, or back to back games? | 10.5 |
| `09_replication.ipynb` | Do calibration, the popularity effect and the opposite trading rule hold on the start of 2026-27? | 10.6 |
| `10_all_tests.ipynb` | Every Stage 10 test in one table, with Holm adjusted p values across the 8 primary tests. Fits no model. | 10.1, 10.7 |

Notebooks 03 and 05 each end with a section headed "Exploratory (not in the Stage 9 plan)". Those sections follow the plan's "Exploratory, added after Stage 9 results" heading and were added after the planned results were seen. Notebooks 07 and 08 each end with a section headed "Exploratory (not in the Stage 10 plan)", which follows the plan's "Exploratory, added after Stage 10 results" heading.

To run all of them from the repo root:

```bash
python scripts/run_notebooks.py                   # every notebook, in name order
python scripts/run_notebooks.py 02_calibration    # only the named ones
```

The script executes each notebook in a fresh kernel with nbconvert and saves it in place with its outputs. If any cell raises an error, it prints the error and exits with code 1. You can also open a notebook in Jupyter or VS Code and run it yourself. Use the project's `.venv` as the kernel.

How the notebooks are organized:
- They read only the CSVs in `data/processed/`, so they run on a fresh clone without the raw cache. The example price path in notebook 01 comes from `data/processed/example_price_path.csv`. Notebook 10 reads the test tables that notebooks 06 to 09 save in `results/`, so it runs after them.
- The statistics are imported from `src/` and are not written out in the notebooks. `src/stats_tools.py` holds the shared ones: Wilson interval, Brier score, logistic regression with clustered standard errors, likelihood ratio test and block bootstrap by date.
- Each notebook opens with its question and the plan section it follows. Each result is followed by a short explanation.
- Random steps use seed 4981, as in the plan.
- Every figure is saved as PNG and every table as CSV in `results/`.
- Notebooks are committed with their outputs, so the figures show on GitHub. Rerunning a notebook on unchanged data gives the same file.

## Repo layout

```
nba-kalshi-calibration/
  src/
    nba_schedule.py      Stage 3: NBA schedule and results
    match_games.py       Stage 5: Kalshi event -> NBA game mapping
    pull_kalshi.py       Stage 4: Kalshi trades and candles (window + pregame)
    build_snapshots.py   Stage 6: 24h/6h/1h/0h snapshots
    quality_report.py    Stage 7: data quality report
    export_price_path.py Stage 8: example game price path, raw cache -> CSV
    metro_population.py  Stage 9: metro area population per team (Census, Statistics Canada)
    stats_tools.py       shared statistics: Wilson, Brier, clustered logit, LR test, block bootstrap,
                         bootstrap p value, joint Wald test, Holm
    calibration.py       calibration tables and plot, used by notebook 02
    eda.py               EDA tables and figures, used by notebook 01
    popularity.py        popularity gaps, model and tables, used by notebook 03
    horizons.py          horizon comparison, used by notebook 04
    strategy.py          trading rule, fees and backtest, used by notebook 05
    seasons.py           Stage 10: what differs between seasons (file names, dates, Kalshi endpoints)
    build_price_paths.py Stage 10: 5-minute price paths, raw cache -> CSV
    player_minutes.py    Stage 10: minutes per player per game (stats.nba.com)
    injury_events.py     Stage 10: injury report status changes and events, used by notebook 07
    all_tests.py         Stage 10: the common tests table and the Holm adjustment, used by notebooks 06 to 10
    line_movement.py     Stage 10: move model and tables, used by notebook 06
    information_arrival.py  Stage 10: when prices move, and the event study, used by notebook 07
    subgroups.py         Stage 10: splits and interaction models, used by notebook 08
    replication.py       Stage 10: the 2026-27 checks, used by notebook 09
    kalshi_client.py     shared: Kalshi client (rate limit, retries, pagination, optional signing)
    csv_io.py            shared: CSV read/write conventions for data/processed and results
    plot_style.py        shared: figure colors and matplotlib style
  envs/
    injuries/            separate uv project for the injury reports (pandas below 3.0)
      fetch_injury_reports.py  download and parse the NBA injury report snapshots
      pyproject.toml, uv.lock
  scripts/
    check_secrets.py     reports which .env values are set, never prints them
    probe_kalshi.py      Stage 2 feasibility probe
    run_notebooks.py     runs every notebook with nbconvert, fails if a cell errors
  notebooks/
    01_eda.ipynb         volume by team, example price path, tipoff price distribution
    02_calibration.ipynb calibration at tipoff
    03_popularity.ipynb  popularity bias test
    04_horizons.ipynb    24h, 6h and 1h compared with tipoff
    05_strategy.ipynb    trading strategy backtest after fees
    06_line_movement.ipynb        price move before tipoff
    07_information_arrival.ipynb  when prices move, injury report event study
    08_subgroups.ipynb            calibration by volume, season half and rest
    09_replication.ipynb          2026-27 replication
    10_all_tests.ipynb            all Stage 10 tests with Holm adjusted p values
  tests/
    conftest.py          puts src/ and tests/ on the import path
    fingerprint.py       type-aware DataFrame fingerprint helper
    test_snapshots.py    time windows, VWAP, spread, normalization, pregame volume
    test_csv_outputs.py  CSV round trip and equality with the replaced parquet files
    test_calibration.py  Stage 8 statistics on known values and synthetic data
    test_stats_tools.py  every function in stats_tools.py
    test_eda.py          EDA tables and the calibration helpers the notebooks call
    test_run_notebooks.py  the notebook runner: outputs saved, failing cell stops the run
    test_popularity.py   popularity gaps, gap model, metro population table
    test_horizons.py     horizon sample, metrics, paired bootstrap, spread filter
    test_strategy.py     fee, order size, expanding mean and threshold, trades, summary
    test_stage10_tools.py  bootstrap p value, joint Wald test, Holm, the common tests table
    test_seasons.py      season file names, endpoints by source, the season in progress
    test_price_paths.py  5-minute grid, strictly-before rule, carried quotes, gzip output
    test_line_movement.py  move, sample, model, bins
    test_subgroups.py    group definitions, interaction model, joint test, bootstrap
    test_replication.py  2025-26 measure and threshold, opposite rule, benchmark
    test_injury_events.py  status changes, release time, name matching, key players, events
    test_information_arrival.py  step changes, window shares, event windows, placebo, paths
    test_dependencies.py requirements.txt and pyproject.toml list the same packages
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
  pyproject.toml         dependencies for uv
  uv.lock                exact versions for uv
  .python-version        Python version for uv (3.12)
  requirements.txt       the same packages for pip
  README.md
```

## Output files

### CSV conventions (`data/processed/*.csv`)

- Timestamps are ISO 8601 in UTC with a Z suffix, for example `2026-01-25T22:30:00Z`. Dates are `YYYY-MM-DD`.
- Booleans are `1` or `0`. Missing values are empty cells.
- Files are UTF-8, with no index column.
- Read them with `csv_io.read_csv`, which restores the column types: zero-padded `game_id` as text, timestamps as UTC, and booleans as bool. It also reads the gzipped `price_paths.csv.gz`.
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
- `kalshi_tipoff_estimate`: Kalshi's listed time, the market's expiration time minus 14 days (minus 2 days in 2026-27, when Kalshi changed the rule). The market list now calls the field `latest_expiration_time`. Kept for reference only.
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

**`example_price_path.csv`**: one example game, minute by minute, over the 30 hours before the official tipoff. One row per team market per minute (3,600 rows). Written by `src/export_price_path.py`, read by notebook 01.
- `event_ticker`, `game_id`, `tipoff_utc`, `side` (home or away), `team` and `market_ticker` identify the row. `minute_end_utc` is the end of the minute.
- `bid`, `ask`: the minute's closing quote. A minute with no candle keeps the previous quote.
- `mid`: the bid/ask midpoint, missing when the book is one-sided.
- `contracts`: contracts traded in the minute, from `minute_end_utc` minus 60 seconds up to but not including `minute_end_utc`.

**`team_metro_population.csv`**: one row per team, written by `src/metro_population.py`. The second popularity measure of Stage 9.
- `team`, `metro_area`, `area_type` (MSA, or CMA for Toronto) and `area_code` (CBSA code, or the Statistics Canada DGUID).
- `population`: the official estimate for `reference_date` (July 1, 2025).
- `source`, `source_url` and `retrieved` (the date the source files were downloaded).
- Each team gets the metropolitan area that contains its arena. LAL and LAC share a value, and so do NYK and BKN.

**`quality_report.md`**: contents:
- match and cache coverage;
- missing snapshots per horizon;
- quote age;
- bid/ask spread distribution by horizon, with the count of rows above 5¢;
- low-volume games (30h window);
- full pregame volume, with consistency checks;
- pre-normalization sums.

**`price_paths.csv.gz`** (Stage 10): one row per matched 2025-26 game per 5-minute step from 30 hours before tipoff to tipoff (361 steps per game). Written by `src/build_price_paths.py`. The step at `minutes_to_tipoff` uses only data strictly before that time, with the same rules as the snapshots, and equals the snapshot table at 24h, 6h, 1h and 0h.
- `home_mid`, `away_mid`, `p_home_mid`, `max_spread`: as in `snapshots.csv`.
- `home_mid_carried_forward`, `away_mid_carried_forward`: 1 when the quote is older than one minute.
- `contracts`: contracts traded in both markets in the 5 minutes before the step.
- `market_open`: 1 when the market had opened before the step. Steps before that are empty. The quote columns are also empty before the first candle of the pulled 30h window.

**2026-27 files** (Stage 10, tag `data-v2`): `nba_games_2026_27.csv`, `game_map_2026_27.csv`, `snapshots_2026_27.csv`, `match_report_2026_27.md` and `quality_report_2026_27.md`. Same columns and rules as the 2025-26 files, for the 248 final games up to November 22, 2026.

**Injury report files** (Stage 10):
- `injury_report_snapshots_2025_26.csv`: one row per snapshot file of the official NBA injury report. `scheduled_time_et` is the report time in the file name, `posted_utc` the time the file was posted on the server, `on_server` and `parsed` say whether it exists and was read.
- `injury_status_changes_2025_26.csv`: one row each time a player's status for a game differs between two consecutive parsed snapshots. `prev_status` and `status` are empty when the player is not listed. `team_submitted_prev` says whether his team had already submitted its report in the earlier snapshot, and `first_out` marks the first snapshot in which he is Out for that game.
- `injury_report_games_2025_26.csv`: every game and team that appears in a parsed snapshot, for the coverage check.
- `player_minutes_2025_26.csv`: minutes per player per game played, from `src/player_minutes.py`.

### `results/` (committed, written by the notebooks)

From notebook 02. All of these use the main sample at horizon 0: 1,229 games, price `p_home`, outcome `home_win`. Section numbers refer to `ANALYSIS_PLAN.md`.

- `calibration_bins.csv`: the 10 equal-count bins (section 2.1). Columns: n, p range, mean p, home wins, win rate, 95% Wilson interval, and the bootstrap interval with bin edges fixed (`boot_lo`, `boot_hi`).
- `calibration_plot.png`: observed win rate against mean price per bin, with the 45° line.
- `brier.csv`: Brier score, constant baseline, difference and skill score (section 2.2).
- `logistic.csv`: intercept `a` and slope `b` of home_win on logit(p_home), with SEs clustered by home team and Wald tests of a = 0 and b = 1 (section 2.3).
- `lr_test.csv`: likelihood ratio test of a = 0 and b = 1 jointly (section 2.4).
- `bootstrap_summary.csv`: date-block bootstrap with B = 2,000 and seed 4981. Gives percentile intervals for every statistic above (section 3).

From notebook 01. EDA files are descriptive and use no outcomes:
- `eda_price_path.png`: one example game. It shows both markets' bid/ask midpoint and band over the 30 hours before tipoff, the four snapshot times, and contracts traded per 30 minutes.
  - The default game is chosen by rule: among games with above-median pregame volume, the one whose `p_home` moved most from 24h to tipoff.
- `eda_team_pregame_volume.csv` / `.png`: each team's average `pregame_contracts` on its own market per game, ranked. This is the popularity measure from section 4.
- `eda_price_summary.csv`: count, mean, standard deviation and quantiles of `p_home` at tipoff, and the share of games priced above 0.5.
- `eda_price_distribution.csv` / `.png`: games per 0.05-wide bin of `p_home` at tipoff.

From notebook 03 (plan section 9.1). Main sample at horizon 0, 1,229 games:
- `popularity_team_measures.csv`: per team, season-average pregame volume and metro population, with ranks.
- `popularity_correlation.csv` and `popularity_scatter.png`: how the two measures relate across the 30 teams.
- `popularity_logistic.csv`: a, b and c of logit P(home win) = a + b · logit(p) + c · gap, once per measure (`volume`, `metro`), with clustered SEs, Wald tests and bootstrap intervals.
- `popularity_effects.csv`: the standard deviation of the gap, c per standard deviation, and the average effect on win probability, with bootstrap intervals.
- `popularity_bootstrap.csv`: the full date-block bootstrap summary.

From notebook 04 (plan section 9.2). Games with a price at all four horizons, 1,222 games. Tables with a `sample` column hold two versions: all those games, and the games whose `max_spread` is at most 0.05 at every horizon.
- `horizons_sample_counts.csv`, `horizons_price_source.csv`: games per horizon and the price source used.
- `horizons_metrics.csv`: per horizon, the Brier score with baseline, a and b with clustered tests, and the LR test.
- `horizons_brier_diff.csv`: Brier score of 24h, 6h and 1h minus tipoff, with paired bootstrap intervals.
- `horizons_bootstrap.csv`: bootstrap intervals for the Brier score at each horizon and for each difference.
- `horizons_calibration_bins.csv`: 10 equal-count bins per horizon with Wilson intervals.
- `horizons_brier.png`, `horizons_calibration.png`: Brier score by horizon, and the four calibration curves.

From notebook 05 (plan section 9.3). The 1,024 tradable games:
- `strategy_sample_counts.csv`: main sample, eligible, threshold-only and tradable games.
- `strategy_results.csv`: one row per strategy (`volume_rule`, `volume_benchmark`, `metro_rule`): trades, win rate, average ask, cost, fees (rounded and unrounded), profit, return per dollar staked, and bootstrap intervals.
- `strategy_trades.csv`: one row per order, for all three strategies.

Exploratory, added after the Stage 9 results (plan heading "Exploratory, added after Stage 9 results"):
- `exploratory_volume_expanding_logistic.csv`, `exploratory_volume_expanding_effects.csv`, `exploratory_volume_expanding_bootstrap.csv`: the 9.1 volume model refitted with the past-games-only volume measure on the 1,074 eligible games of 9.3 (check E1, notebook 03).
- `exploratory_strategy_before_fees.csv`: profit and return of the three strategies before fees (check E2, notebook 05).

Stage 10 (plan sections 10.3 to 10.7):
- `stage10_all_tests.csv`: every Stage 10 test with its estimate, interval, raw p value and, for the 8 primary tests, the Holm adjusted p value (notebook 10). The column `role` says whether a test is `primary`, `secondary` or `exploratory`. `stage10_tests_<part>.csv` are the same rows per notebook, before the adjustment.
- Notebook 06, 1,222 games: `line_movement_sample_counts.csv`, `line_movement_summary.csv` and `line_movement_distribution.png` (the moves), `line_movement_logistic.csv`, `line_movement_effects.csv` and `line_movement_bootstrap.csv` (a, b and d of the four models), `line_movement_bins.csv` and `line_movement_bins.png` (win rate minus tipoff price by size of the move).
- Notebook 07: `arrival_by_minutes.csv` / `.png`, `arrival_by_clock.csv` / `.png` and `arrival_window_shares.csv` (when prices move); `injury_reliability.csv`, `injury_event_counts.csv`, `injury_events.csv` (one row per event, with its release time and price changes), `injury_event_test.csv`, `injury_event_path.csv` and `injury_event_path.png` (the event study).
- Notebook 08, 1,229 games: `subgroups_groups.csv` (the groups), `subgroups_metrics.csv` (Brier, a and b per group), `subgroups_interaction.csv` (the interaction models), `subgroups_brier_diff.csv`, `subgroups_bootstrap.csv`, `subgroups_calibration_bins.csv` and one calibration figure per split.
- Notebook 09, 248 games of 2026-27: `replication_sample_counts.csv`, `replication_brier.csv`, `replication_logistic.csv`, `replication_lr_test.csv`, `replication_bootstrap.csv`, `replication_calibration_bins.csv` and `replication_calibration_plot.png` (calibration); `replication_popularity_*.csv` (the popularity model); `replication_opposite_rule.csv` and `replication_opposite_trades.csv` (the opposite rule and its benchmark); `replication_slope_width.csv`.

Exploratory, added after the Stage 10 results (plan heading "Exploratory, added after Stage 10 results"):
- `exploratory_injury_by_status.csv`: the injury events split by the player's previous status (Questionable, Doubtful, not listed), with the mean change in the main and the anticipation window (check X1, notebook 07).
- `exploratory_slope_by_month.csv` and `exploratory_slope_by_month.png`: the calibration slope b by calendar month of 2025-26, with clustered intervals (check X2, notebook 08).
- `stage10_tests_exploratory_information_arrival.csv`, `stage10_tests_exploratory_subgroups.csv`: the tests of the two checks in the common layout. They are in `stage10_all_tests.csv` with the role `exploratory` and no Holm adjustment.

### `data/raw/` (gitignored, needed to rebuild without re-pulling)

- `kalshi_markets_KXNBAGAME.json`: the Kalshi market list.
- `kalshi/markets/<ticker>.json.gz`: one gzipped file per market for the window phase, holding the raw trades and candlesticks returned for its window, plus market metadata.
- `kalshi/pregame/<ticker>.json.gz`: one gzipped file per market for the pregame phase, holding the raw trades from market open to the window start.
- `nba_schedule_2025_26.json`: the raw stats.nba.com response, used by `--cached`.
- `kalshi_markets_KXNBAGAME_2026_27.json`, `nba_schedule_2026_27.json`: the same for 2026-27. The 2026-27 market files share the two `kalshi/` folders.
- `player_gamelog_2025_26.json`: the raw player game log response.
- `injury_reports/`: the downloaded report PDFs, the download and parse logs, and every parsed row (`injury_report_rows_2025_26.csv.gz`).

### `logs/` (gitignored)

- `pull_kalshi.log`: pull progress.
- `pull_kalshi_failures.csv`: markets that failed in the last run, by phase.

## Rules this pipeline follows

- Snapshot times always use the official NBA tipoff, never Kalshi's expiration-based estimate.
- Only data strictly before the snapshot time is used. Missing values stay missing, and carried-forward quotes are flagged.
- Every file read and write passes `encoding="utf-8"`.
- Secrets, keys and raw data are never committed.

# Analysis plan

STSCI 4981/5981 final project. Research question: are Kalshi NBA game prices before tipoff well calibrated, and are popular teams overpriced?

This plan was written and committed before any result was computed. The data it uses is frozen at the git tag `data-v1`. Any change made after results are seen will be listed under "Deviations" at the end, with the reason.

## 1. Data and main sample

- Source: `data/processed/snapshots.csv` at tag `data-v1`.
- Rows: `main_sample == 1` and `horizon_h == 0`. That is one row per game, priced from the last quote strictly before the official NBA tipoff.
  - This gives 1,229 regular-season games. The only matched game left out is GSW@MIN, flagged `postponed_window`.
  - The 5 `tipoff_moved` games stay in.
- Price: `p_home`, the normalized home-win price, home / (home + away). At horizon 0, every main-sample row uses the bid/ask midpoint, and `p_home` is never missing.
  - Its range is 0.035 to 0.975, so logit(p_home) is finite and no clipping is needed.
  - If a row ever has `p_home` missing, it is dropped and the count is reported.
- Outcome: `home_win` (1 = home team won), from the official NBA result.
- Notation: p = p_home, y = home_win, n = number of games.

## 2. Tests

All tests are two-sided at α = 0.05.

### 2.1 Calibration plot

- Sort games by p and split them into 10 equal-count bins. Ties are broken by `game_id`, so the bins are reproducible. With 1,229 games, nine bins hold 123 games and one holds 122.
- For each bin, report:
  - the number of games;
  - the p range and mean p;
  - the number of home wins and the observed home-win rate;
  - the 95% Wilson score interval for that rate.
- Wilson interval, with z = 1.96, k wins out of m games and p̂ = k/m:

  center = (p̂ + z²/(2m)) / (1 + z²/m), half-width = z·√(p̂(1−p̂)/m + z²/(4m²)) / (1 + z²/m)

- Figure: observed win rate against mean p, with Wilson error bars and the 45° line of perfect calibration.

### 2.2 Brier score against a constant baseline

- Brier score: BS = (1/n) Σ (p − y)².
- Baseline: a constant forecast equal to the overall home-win rate ȳ in the main sample, so BS₀ = ȳ(1 − ȳ).
  - ȳ is computed in-sample, which slightly favors the baseline.
- Report BS, BS₀, the difference BS − BS₀, and the Brier skill score BSS = 1 − BS/BS₀.

### 2.3 Logistic recalibration regression

- Model: logit P(y = 1) = a + b · logit(p).
- Perfect calibration means a = 0 and b = 1:
  - a ≠ 0 is a constant home or away bias;
  - b < 1 means prices are too extreme;
  - b > 1 means prices are not extreme enough.
- Fit by maximum likelihood. Standard errors are cluster-robust, clustered by `home_team` (30 clusters).
- Report a and b, their clustered standard errors and 95% confidence intervals, and Wald z-tests of a = 0 and of b = 1.

### 2.4 Likelihood ratio test of a = 0 and b = 1

- Restricted log-likelihood (the prices taken as they are): ℓ_R = Σ [y log p + (1 − y) log(1 − p)].
- Unrestricted log-likelihood: ℓ_U, from the fit in 2.3.
- Test statistic: LR = 2(ℓ_U − ℓ_R), compared with χ² on 2 degrees of freedom. Report LR and its p-value.
- The LR test treats games as independent. The bootstrap in section 3 checks whether dependence within a date changes the conclusion.

## 3. Robustness: block bootstrap by game date

- Blocks: `game_date` (US Eastern), which gives 164 dates. Each replicate resamples whole dates with replacement until it has 164 dates. All games on a drawn date come in together.
- Settings: B = 2,000 replicates, random seed 4981.
- Each replicate recomputes:
  - BS, BS₀ (from that replicate's own home-win rate), BS − BS₀ and BSS;
  - a and b from the logistic fit;
  - the LR statistic;
  - the home-win rate in each calibration bin, with the bin edges fixed at the full-sample values.
- Report the 95% percentile intervals.
- The conclusion is called robust when the bootstrap intervals agree with the main tests:
  - whether 0 is inside the interval for a;
  - whether 1 is inside the interval for b;
  - whether 0 is inside the interval for BS − BS₀.

## 4. Popularity measure (defined now, used later)

- A team's popularity is the mean, over all its main-sample games (home and away), of its own market's `pregame_contracts`.
  - This counts contracts traded from market open to the official tipoff, strictly before tipoff.
  - Teams are ranked 1 to 30, with 1 = most traded.
- This stage shows the ranking only as a descriptive EDA figure. Testing whether popular teams are overpriced is a later stage, and its specification will be added to this plan before it is run.

## 5. Out of scope for this stage

- Horizons 24h, 6h and 1h.
- The backup price types (last trade, 10-minute VWAP).
- Popularity regressions.
- Any betting strategy.

## 6. Outputs

- Script: `src/calibration.py` runs sections 2 and 3 on horizon 0 only.
- Tables, as CSV in `results/`:
  - `calibration_bins.csv`
  - `brier.csv`
  - `logistic.csv`
  - `lr_test.csv`
  - `bootstrap_summary.csv`
- Figure, as PNG in `results/`: `calibration_plot.png`.
- EDA, in `src/eda.py`:
  - one example game's price path from the raw cache;
  - average pregame volume by team, using the definition in section 4.

## Stage 9

Popularity bias, horizons and a trading strategy. This section was written and committed before any Stage 9 result was computed. Sections 1 to 6 (Stage 8) are unchanged, and so is the data (tag `data-v1`).

Amendments to 9.3, made on 2026-11-23 after review of the first version of this section and before any Stage 9 result was computed:

- Fee: the formula and its source are fixed as Kalshi's fee schedule filed with the CFTC. The first version took the coefficient from Kalshi's API documentation, pending confirmation.
- Stake: each trade stakes $100, with contracts = floor(100 / ask) and one fee on the whole order. The first version bought one contract per trade.
- Threshold: the 75th percentile of |gap| is taken over eligible games played earlier, and the first 50 eligible games only set the threshold. The first version used the whole season's eligible games, which a trader could not have known. The benchmark and the metro robustness check follow the same rule.

Since Stage 9, the analysis runs in notebooks. `notebooks/01_eda.ipynb` and `notebooks/02_calibration.ipynb` run the Stage 8 code and reproduce the Stage 8 files in `results/` exactly. The shared statistics are in `src/stats_tools.py`.

Settings that apply to all of Stage 9:

- All tests are two-sided at α = 0.05.
- Clustered standard errors use `home_team` (30 clusters), as in section 2.3.
- Every bootstrap is the block bootstrap by `game_date` of section 3: B = 2,000 replicates, seed 4981, 95% percentile intervals.
- "Main sample" means `main_sample == 1`. p is `p_home` and y is `home_win`.
- A team's "own market volume" in a game is `home_pregame_contracts` when it plays at home and `away_pregame_contracts` when it plays away.

### 9.1 Popularity bias test (notebook 03)

Sample: the section 1 sample. Main sample at horizon 0, price `p_home`.

Two popularity measures, so the result does not depend on one choice:

1. **Volume, leave one out.** For each game and each team: the mean of the team's own market volume over all its other main-sample games in the season, home and away. The current game is excluded, so a game's own volume never enters its own regressor.
   - This measure uses games played later in the season. That is acceptable for a test. The strategy in 9.3 uses a version without future games.
2. **Metro area population.** The latest official July 1 population estimate of each team's metropolitan area.
   - US teams: US Census Bureau, Metropolitan Statistical Area population totals, Vintage 2025 (estimates for July 1, 2025, released March 26, 2026). File `cbsa-est2025-alldata.csv` from <https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-metro-and-micro-statistical-areas.html>.
   - Toronto: Statistics Canada, Table 17-10-0148-01, population estimates for July 1, 2025 by census metropolitan area (released January 14, 2026). <https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710014801>
   - Each team is assigned the metropolitan area that contains its home arena. Teams in the same area get the same value: LAL and LAC, NYK and BKN.
   - The table is saved to `data/processed/team_metro_population.csv`, with the area name and code, the population, the reference date, the source and the date retrieved.

For each game and each measure, the popularity gap is

  gap = log(home team's measure) − log(away team's measure).

A positive gap means the home team is the more popular one.

Model, fitted once per measure:

  logit P(y = 1) = a + b · logit(p) + c · gap

- Fit by maximum likelihood, with standard errors clustered by home team.
- Report a, b and c with their clustered standard errors, 95% confidence intervals, and Wald z-tests of a = 0, b = 1 and c = 0.
- The test of interest is c = 0. c < 0 means the more popular team wins less often than its price implies, so it is overpriced. c > 0 means it is underpriced.
- Robustness: the block bootstrap, refitting the model in each replicate. Report percentile intervals for a, b and c. The result for c is called robust when the bootstrap interval and the clustered test agree on whether 0 is excluded.

To read c in plain terms, also report, per measure:

- the standard deviation of the gap across the sample games (SD, with n − 1 in the denominator), and c · SD;
- the average effect on win probability of a one-SD larger gap at the same price: the mean over games of Λ(η_i + c · SD) − Λ(η_i), where η_i is game i's fitted value a + b · logit(p_i) + c · gap_i and Λ is the logistic function. Its bootstrap interval is reported too.

Descriptive, across the 30 teams, with no test attached:

- the Pearson correlation between log season-average volume (the section 4 measure) and log metro population, and the Spearman rank correlation between the two;
- a scatter of season-average volume against metro population, both on log axes, with team labels.

### 9.2 Horizon comparison (notebook 04)

- Sample: main-sample games with `p_home` present at all four horizons (24h, 6h, 1h, 0h). Report the count.
  - This is expected to drop the 7 games Kalshi listed less than 24 hours before tipoff. They are dropped from every horizon, so all horizons use the same games.
  - `p_home` is used as it is at each horizon. Before tipoff it can come from a backup price (last trade or VWAP) when the midpoint is missing. Report the count of each `price_source` per horizon.
- At each horizon, as in Stage 8:
  - Brier score, baseline, difference and skill score (section 2.2). The baseline is the same at every horizon, because the games are the same.
  - Intercept a and slope b with clustered standard errors and Wald tests of a = 0 and b = 1 (section 2.3).
  - The likelihood ratio test of a = 0 and b = 1 (section 2.4).
- Paired block bootstrap: each replicate draws the dates once and computes the Brier score at all four horizons on the same resampled games.
  - Report, for h = 24, 6 and 1, the difference BS_h − BS_0 with its 95% percentile interval. A positive difference means the earlier price is less accurate than the tipoff price.
  - Report the bootstrap interval of the Brier score at each horizon.
- Figures:
  - Brier score by horizon, with the bootstrap intervals.
  - Calibration curves for all four horizons on one plot. Each horizon uses 10 equal-count bins built as in section 2.1 from that horizon's prices (ties broken by `game_id`). The plot shows the bin points joined by lines and the 45° line, without error bars. The Wilson intervals are in the saved table.
- Robustness: drop every game whose `max_spread` is above 0.05 at any horizon, or is missing at any horizon. Report the count. Repeat the table and the paired differences on the remaining games, so all horizons still use the same games.

### 9.3 Strategy backtest (notebook 05)

One rule only, fixed now. Nothing is tuned.

1. **Popularity known at the time.** Order the main-sample games by `tipoff_utc`. For each game and each team, take the expanding mean of the team's own market volume over its main-sample games with an earlier tipoff. The current game and later games are not used.
   - A game is eligible when both teams have at least 10 earlier games.
   - gap = log(home team's expanding mean) − log(away team's expanding mean).
2. **Threshold known at the time.** Order the eligible games by `tipoff_utc`, with ties broken by `game_id`. For each eligible game, the threshold q is the 75th percentile of |gap| over the eligible games with an earlier tipoff. The percentile uses linear interpolation.
   - The first 50 eligible games in this order only set the threshold. No trade is made in them.
   - The games after those 50 are the "tradable games".
3. **Rule.** In every tradable game with |gap| ≥ q, buy YES contracts on the less popular team (the one with the lower expanding mean) at its ask price at tipoff. That is `home_ask` or `away_ask` at horizon 0, the closing ask of the last 1-minute candle strictly before tipoff.
   - Stake: $100 per trade. Contracts C = floor(100 / ask).
   - If that ask is missing, or is not strictly between 0 and 1, there is no trade. If the gap is exactly 0, there is no less popular team and no trade. Report the counts.
4. **Fees.** Every trade pays Kalshi's taker fee, computed once on the whole order:

     fee = 0.07 · C · P · (1 − P), rounded up to the next cent,

   where P is the ask price in dollars and C the number of contracts. There is no settlement fee.
   - Source: Kalshi's fee schedule as filed with the CFTC, <https://cftc.gov/filings/orgrules/rule091222kexdcm003.pdf>. It gives this formula as the general trading fee and states "There is no settlement fee."
   - The Kalshi API (`GET /series/KXNBAGAME`) reports `fee_type = quadratic_with_maker_fees` and `fee_multiplier = 1` for the NBA game series, and `GET /series/fee_changes` lists no fee change for it. So the general formula applies with no special multiplier.
   - Both were checked on 2026-11-23.
   - For reference, the table also reports total fees without the cent rounding, 0.07 · C · P · (1 − P). Profit and return are computed with the rounded fee.
5. **Results.** For each trade: cost = C · ask, payout = C dollars if the team wins and $0 if not, profit = payout − cost − fee. Report:
   - number of trades and win rate;
   - total cost, total fees (rounded and unrounded) and profit;
   - return per dollar staked = profit / (total cost + total fees);
   - block bootstrap intervals for profit and for return per dollar staked. The replicates resample the dates of the tradable games. The trades chosen by the rule stay fixed, and the thresholds are not recomputed.
6. **Benchmark.** The same calculation for buying the less popular team, with the same $100 stake and fee, in every tradable game. This uses the same games the rule could trade in, so it shows the cost of trading without the filter.
7. **Robustness.** The same rule with the metro population gap of 9.1 in place of the volume gap. Eligibility and the first 50 games are the same as above, so the tradable games are the same. The threshold is the 75th percentile of the absolute metro gap over the eligible games with an earlier tipoff, and the contracts bought are on the team from the smaller metro area.
   - Games between two teams of the same metro area have a gap of 0 and are never traded.
   - The metro gap takes few distinct values, so the share of games traded can differ from one quarter. Report the count.

### 9.4 Outputs

- Data: `data/processed/team_metro_population.csv`.
- Notebook 03, in `results/`:
  - `popularity_team_measures.csv`, `popularity_correlation.csv`, `popularity_scatter.png`
  - `popularity_logistic.csv`, `popularity_effects.csv`, `popularity_bootstrap.csv`
- Notebook 04, in `results/`:
  - `horizons_metrics.csv`, `horizons_brier_diff.csv`, `horizons_bootstrap.csv`, `horizons_calibration_bins.csv`
  - `horizons_brier.png`, `horizons_calibration.png`
- Notebook 05, in `results/`:
  - `strategy_results.csv`, `strategy_trades.csv`

### 9.5 Not in Stage 9

- Any other trading rule, threshold or holding period, and any search over them.
- Separate analyses of the backup price types (last trade, 10-minute VWAP).
- Any model or test not listed above. Ideas that come up are listed as suggestions and are not run.

## Exploratory, added after Stage 9 results

Added on 2026-11-23, after the Stage 9 results were seen. Nothing in this section was planned in advance, so its results are exploratory and carry less weight than the planned tests in sections 1 to 9. This text was committed before either check was run. The planned results and their files are not changed.

### E1. Volume model with a past-games-only measure (notebook 03)

Reason: the volume result in 9.1 (c = 0.611) uses a leave-one-out season average, which includes games played after the one being priced. That measure may carry information that was not available at tipoff. This check asks whether c stays positive when the measure uses earlier games only.

- Measure: the expanding mean of 9.3, step 1. For each game and each team, the mean of the team's own market volume over its main-sample games with an earlier tipoff. gap = log(home) − log(away).
- Sample: the eligible games of 9.3, where both teams have at least 10 earlier games. Notebook 05 found 1,074 of them. Price `p_home` at horizon 0.
- Model, clustering and bootstrap: exactly as in 9.1. logit P(y = 1) = a + b · logit(p) + c · gap, standard errors clustered by home team, block bootstrap by game date with B = 2,000 and seed 4981.
- Report a, b and c with clustered standard errors, Wald tests and bootstrap intervals, and c in plain terms as in 9.1: the SD of the gap, c · SD, and the average effect on win probability.
- Limit on the reading: this sample leaves out the early-season games that 9.1 includes. A difference between this c and the 9.1 c can come from the measure or from the sample. No further model is run to separate the two.

### E2. Strategy profit and return before fees (notebook 05)

Reason: to show how much of each strategy's loss is fees.

- For each of the three strategies of 9.3 (the volume rule, the benchmark and the metro rule):
  - profit before fees = total payout − total cost, which equals profit + total fees;
  - return before fees = profit before fees / total cost.
- The trades are the ones already made in 9.3. No new rule is run and no new bootstrap is added.

### Not run

- The opposite rule (buying the more popular team). These results suggest it, so testing it on the same season would not be a fair test.

### Outputs

- `results/exploratory_volume_expanding_logistic.csv`, `exploratory_volume_expanding_effects.csv`, `exploratory_volume_expanding_bootstrap.csv`
- `results/exploratory_strategy_before_fees.csv`

## Stage 10

Four extensions: line movement, information arrival, subgroups, and a replication on the 2026-27 season. This section was written and committed on 2026-11-23, before any Stage 10 result was computed. Everything above it is unchanged, and so are the 2025-26 data (tag `data-v1`) and the Stage 8 and Stage 9 result files.

Amendments, made on 2026-11-23 after review of the first version of this section and before any Stage 10 result was computed:

- 10.2: both pull phases are run, not only the window phase, because `snapshots.csv` needs the pregame trades. `game_map.csv` is checked the same way as `snapshots.csv`. The price paths are saved gzipped, as `price_paths.csv.gz`.
- 10.4, D1: the main descriptive range is 24h to tipoff. The 30h to 24h window is shown separately with its game-step count. The first version put all five windows in one total.
- 10.4, D2: the release time is the report file's actual posting time from the server timestamp where available, with the scheduled report time as the fallback. The first version used the scheduled time only. A secondary anticipation window, from 60 to 5 minutes before the release, is added with the same tests as the main window. The injury reports are fetched in a separate Python environment, because `nbainjuries` needs pandas below 3.0.
- 10.6: the opposite rule gets a benchmark (buy the more popular team in every 2026-27 sample game), as a secondary result. If the live Kalshi endpoints have no field equivalent to `expiration_time`, `tipoff_moved` is left empty for 2026-27 and this is recorded under "Deviations".

Run order: 10.3 line movement and 10.5 subgroups first, then 10.6 replication, then 10.4 information arrival. The order is set by the time each part needs. It does not depend on any result.

### 10.1 Settings and reporting rules

Settings that apply to all of Stage 10, unless a section says otherwise:

- All tests are two-sided at α = 0.05.
- Clustered standard errors use `home_team` (30 clusters).
- Every bootstrap is the block bootstrap by `game_date`: B = 2,000 replicates, seed 4981, 95% percentile intervals. Group labels, cut points, thresholds and chosen trades are fixed at their full-sample values in every replicate.
- "Main sample" means the 2025-26 games with `main_sample == 1`. p is `p_home` and y is `home_win`.
- A replicate in which a model cannot be fitted is dropped, and the number dropped is reported.

Reporting rules:

1. Every Stage 10 test is reported in `results/stage10_all_tests.csv` with its estimate, interval, raw p value and Holm adjusted p value. The Holm adjustment is across the Stage 10 primary tests listed in rule 2. Nulls are reported the same way as significant results.
2. Primary tests (the Holm family, 8 tests):
   - 10.3: d = 0 in the 24h model (1 test).
   - 10.5: the group by slope interaction in each of the three splits (3 tests). The liquidity split has two interactions, tested jointly as one test.
   - 10.4, D2: the mean event change against zero (1 test).
   - 10.6: the calibration LR test, the popularity c = 0, and the opposite rule return against zero (3 tests).

   Everything else is secondary or descriptive. Secondary tests are in the table with a raw p value and no adjusted one.
3. Anything added after seeing results goes under an "Exploratory" heading with its reason.

How the p values are computed:

- Regression coefficients: the Wald z test with clustered standard errors. The interval in the table is the clustered 95% interval. The bootstrap interval is reported next to it as a robustness check.
- Joint tests: the liquidity interaction is a Wald χ² test on 2 degrees of freedom with the clustered covariance. The calibration LR test is the test of section 2.4. A joint test has no single estimate, so its row gives the statistic and degrees of freedom, and its components (the two interactions, or a and b) are separate secondary rows with their own intervals.
- Bootstrap tests (the event study means and the opposite rule return): the interval is the percentile interval, and the p value is

    p = min(1, 2 · min( (1 + #{θ* ≤ 0}) / (B + 1), (1 + #{θ* ≥ 0}) / (B + 1) )),

  where θ* are the bootstrap replicates of the statistic. With B = 2,000 the smallest possible value is about 0.001.
- Holm: sort the m = 8 primary p values, p(1) ≤ ... ≤ p(8). The adjusted value of p(i) is the largest of min(1, (8 − j + 1) · p(j)) over j ≤ i. A primary test is called significant when its adjusted p value is below 0.05.
- If D2 is stopped because the injury reports cannot be used (see 10.4), the family is the other 7 tests and this is recorded under "Deviations".
- The Holm values need all primary tests. A report made before all of them have run gives the raw p value and the bound min(1, 8 · p), which is never below the final Holm value, and says that it is a bound.

### 10.2 Data

- **Raw Kalshi cache, 2025-26.** Pulled again with `src/pull_kalshi.py`, both phases (window and pregame). The market list is downloaded again too. `data/processed/game_map.csv` and `data/processed/snapshots.csv` are rebuilt and compared byte for byte with the committed files before anything else is run. Any difference is reported, and the committed files are kept.
- **Price paths.** `src/build_price_paths.py` writes `data/processed/price_paths.csv.gz` (a gzipped CSV): one row per matched 2025-26 game per 5-minute step, from tipoff − 30h to tipoff (361 steps, `minutes_to_tipoff` = 1800, 1795, ..., 0).
  - For the step at time S = tipoff − `minutes_to_tipoff`, each side's quote comes from the last 1-minute candle ending strictly before S, the same rule as the snapshots. A quote older than one minute is carried forward and flagged.
  - `home_mid`, `away_mid`: bid/ask midpoints. A one-sided book gives no midpoint. `p_home_mid` = home_mid / (home_mid + away_mid) when both exist. `max_spread` is the wider of the two spreads.
  - `contracts`: contracts traded in both markets in [S − 5 min, S).
  - Steps before the market opened are empty. So are steps before the first candle of the pulled 30h window, because the cache holds no earlier quote.
- **Injury reports and minutes** (for 10.4): see 10.4.
- **2026-27 season** (for 10.6): see 10.6.

### 10.3 Line movement (notebook 06)

Question: does the price move before tipoff contain information that the tipoff price missed?

- Sample: main-sample games with `p_home` at 24h and at tipoff. Stage 9 found 1,222 such games. Report the count.
- move = logit(p at tipoff) − logit(p at 24h), for the home side. `p_home` is used as it is at each horizon, as in 9.2.
- Model: logit P(y = 1) = a + b · logit(p at tipoff) + d · move.
  - Standard errors clustered by home team. Block bootstrap by date, refitting the model in each replicate.
  - Report a, b and d with clustered standard errors, 95% intervals, Wald tests of a = 0, b = 1 and d = 0, and bootstrap intervals.
- **Primary test: d = 0.** d = 0 means the tipoff price is efficient with respect to the move. d > 0 means underreaction: the market did not move far enough. d < 0 means overreaction.
- d in plain terms, as in 9.1: the SD of the move, d · SD, and the average effect on win probability of a one-SD larger move at the same tipoff price, with its bootstrap interval. Every replicate uses the full-sample SD.
- Secondary:
  - the same model with the move from 6h to tipoff, and with the move from 1h to tipoff, on the same games;
  - the 24h model after the spread filter of 9.2 (drop every game whose `max_spread` is above 0.05 or missing at any of the four horizons; Stage 9 kept 1,165 games).
- Descriptive, no test attached:
  - the distribution of the 24h move: a histogram, and a table with the mean, SD, quartiles, 5th and 95th percentiles, and the shares of positive, negative and zero moves;
  - a binned plot: sort the games by move into 10 equal-count bins (ties broken by `game_id`). For each bin, plot the home-win rate minus the mean tipoff price against the mean move. The bar is the Wilson interval of the win rate, shifted by the mean tipoff price.

### 10.4 Information arrival (notebook 07)

#### D1. When do prices move (descriptive, no test)

- Sample: every main-sample game in `price_paths.csv`.
- For each game and each 5-minute step, the absolute change in logit(`p_home_mid`) from the previous step. A change is computed only when both steps have a midpoint. Gaps are not bridged.
- Averages:
  - by minutes to tipoff: the mean absolute change at each of the 360 steps, over the games that have one;
  - by US Eastern clock time on the game day: steps whose end time falls on the game's Eastern date, grouped in 5-minute clock bins. Bins with fewer than 50 game-steps are not plotted. The number of game-steps per bin is in the table.
- The main descriptive range is 24h to tipoff. The two plots mark the steps earlier than 24h as a separate range.
- Share of total movement: the sum of absolute changes in each window divided by the sum over the four windows from 24h to tipoff, pooled over games. A step belongs to the window that contains its end time m (minutes to tipoff):
  - 24h to 6h: 360 ≤ m < 1440
  - 6h to 1h: 60 ≤ m < 360
  - 1h to 30 min: 30 ≤ m < 60
  - last 30 min: 0 ≤ m < 30
- The table also gives each window's share per hour, because the windows differ in length, and the number of game-steps with a change out of the number possible.
- The 30h to 24h window (1440 ≤ m < 1800) is shown in a separate row: its sum of absolute changes, its mean absolute change per step and its game-step count. It is not part of the total. Early steps are empty for many games (see 10.2), so this window is understated by construction.

#### D2. Injury report event study (primary)

Data:

- Injury reports: the `nbainjuries` package, every official NBA injury report snapshot of the 2025-26 regular season. The fetch runs in a separate Python environment that only writes CSV files, because the package needs pandas below 3.0.
- Release time of a snapshot:
  - the actual posting time, taken from the server timestamp of the report file (the HTTP `Last-Modified` header), where available;
  - otherwise the scheduled time, which is the report timestamp in the file name (US Eastern).
  - The server timestamp counts as available when the server returns one and it lies between 30 minutes before and 60 minutes after the scheduled time. A timestamp outside that range is taken to be a later re-upload and is not used.
  - Which of the two was used is recorded for every snapshot and every event, and the counts are reported.
  - Snapshots are ordered by their scheduled time. "Consecutive" means consecutive in that order.
- Minutes: `nba_api` player game logs for the 2025-26 regular season.
- Saved to `data/processed/`: one row per report snapshot (time, fetched or not, rows parsed), one row per change of a player's status for a game, and the player minutes.

Reliability rule, fixed now. D2 is stopped, and reported, if any of these fails:

- fewer than 95% of the snapshots that exist on the NBA server can be fetched and parsed;
- fewer than 95% of the main-sample games appear, with both teams, in at least one parsed snapshot;
- in a random draw of 20 events (seed 4981) checked by hand against the original report files, any event is wrong (wrong player, status, game or report time).

No other source is substituted.

Definitions:

- **Status change to Out.** For a main-sample game, a player whose status is Out in a snapshot and was not Out in the snapshot before it. "Not Out before" means he was listed with another status, or he was not listed while his team had already submitted its report for that game in the earlier snapshot. The first snapshot in which a team has submitted does not create events. Only a player's first change to Out for a game counts.
- **Key player.** Among the players of his team, top 3 in average minutes per game played for that team, over the team's regular-season games before that game. A player needs at least 5 such games. Injury report names are matched to game log names within the team after removing case, accents, punctuation and suffixes. The number of changes to Out that cannot be matched to a player is reported.
- **Event.** One team in one game at one snapshot, with at least one key player changing to Out. Several key players of the same team in the same snapshot make one event.
  - The snapshot is released at least 30 minutes before the official tipoff.
  - If both teams of a game have an event in the same snapshot, both are dropped, because the two effects on the price have opposite signs. The count is reported.
- **Event change (main window).** The affected team's normalized midpoint (`p_home_mid` for the home team, 1 − `p_home_mid` for the away team) 30 minutes after the release minus its value 5 minutes before the release, in probability units. A drop is negative. An event with a missing midpoint at either time is dropped and counted.
  - With release time T, the price before is the value of the last 5-minute step at or before T − 5 min. The price after is the value of the first 5-minute step at or after T + 30 min. So the window never covers less than the stated range when T is not on the 5-minute grid.
- **Anticipation change (secondary window).** The midpoint 5 minutes before the release (the same price before as above) minus its value 60 minutes before the release (the last step at or before T − 60 min).
- **Placebo change.** For each window, the same calculation with all times moved 3 hours earlier, for the same game and team, when both midpoints exist.

Tests, with the block bootstrap by game date (events of one date are drawn together):

- **Primary: the mean event change against zero.**
- Secondary, main window: the mean placebo change against zero, and the mean of (event change − placebo change) against zero, over the events that have a placebo.
- Secondary, anticipation window: the same three tests. The mean anticipation change against zero, its placebo mean against zero, and the mean of (anticipation change − its placebo) against zero. The events are the events of the main window that also have both anticipation midpoints.
- Report the number of events at each step of the definitions above.

Figure: the average path of the affected team's normalized midpoint from 60 minutes before to 120 minutes after the release, in 5-minute steps, as the difference from its value 5 minutes before the release, with a pointwise 95% bootstrap band. Each event is lined up on the step used as its price before, which stands for 5 minutes before the release. Steps after tipoff are left out, and the number of events behind each point is in the table.

### 10.5 Where is the market less accurate (notebook 08)

Sample: the main sample at tipoff, 1,229 games. Three splits, each fixed now:

1. **Liquidity.** Total pregame contracts = `home_pregame_contracts` + `away_pregame_contracts`. Sort the games by it (ties broken by `game_id`) and cut into three equal-count groups: low, middle and high (410, 410 and 409 games).
2. **Season half.** Early = games before the median `game_date` of the sample. Late = games on or after it.
3. **Rest.** Back to back = at least one of the two teams played a regular-season game on the previous calendar day (US Eastern), from `data/processed/nba_games_2025_26.csv`. The other group is games where neither team did. The NBA Cup final is not a regular-season game and is not in that file.

For each split:

- Per group: the number of games, the Brier score, its constant baseline and skill score (section 2.2, baseline from the group's own home-win rate), and a and b from the recalibration fit within the group (section 2.3), with clustered standard errors and tests of a = 0 and b = 1. These tests are secondary.
- A calibration plot by group: 5 equal-count bins within each group, built as in section 2.1, with Wilson intervals.
- **Primary test.** One logistic model on all 1,229 games:

    logit P(y = 1) = a + b · logit(p) + Σ_g ( α_g · G_g + β_g · G_g · logit(p) ),

  where G_g is the indicator of group g and the reference groups are low liquidity, early, and no back to back. Standard errors clustered by home team.
  - Season half and rest: the test is β = 0 (Wald z).
  - Liquidity: the test is β_middle = β_high = 0 jointly (Wald χ², 2 df). The two single interactions are secondary rows.
  - β ≠ 0 means the slope of the calibration line differs between groups.
- The difference in Brier score between groups with its bootstrap interval and bootstrap p value (secondary): late − early, back to back − no back to back, and for liquidity high − low, middle − low and high − middle.
- Limit on the reading: a Brier difference between groups can come from a different mix of close and lopsided games, not only from accuracy. The interaction test is the test of calibration.

### 10.6 Replication on the 2026-27 season (notebook 09)

Data:

- Sample: every 2026-27 regular-season game (NBA game ID prefix 002) that is final and has a settled Kalshi `KXNBAGAME` event, up to the cutoff date.
  - The cutoff date is the last US Eastern game date before the day of the pull on which every game is final and settled. The pull is made once. The sample is not extended after any result is seen. The cutoff date and the game count are reported.
- Pipeline: the same scripts as for 2025-26 (`nba_schedule.py`, `match_games.py`, `pull_kalshi.py`, `build_snapshots.py`), with a season parameter and no copied code. Same matching rules, cleaning and flags: postponed originals dropped, `postponed_window` games out of the main sample, `tipoff_moved` flagged and kept. The 2025-26 outputs must stay byte identical after this change.
  - Kalshi's historical endpoints end at their cutoff (2026-09-24 when checked on 2026-11-23). The 2026-27 markets are on the live endpoints, so the client gets live versions of the same three requests (markets, trades, 1-minute candles).
  - `tipoff_moved` is computed from the market's `expiration_time`, which the live endpoints return empty. If a live field carries the same time, it is used and named in the match report. If none does, `tipoff_moved` is left empty for 2026-27 and this is recorded under "Deviations". The flag does not change the sample.
- Outputs in `data/processed/`: `nba_games_2026_27.csv`, `game_map_2026_27.csv`, `snapshots_2026_27.csv` and the match report. The files are frozen with the git tag `data-v2`.
- Price: `p_home` at horizon 0, as in section 1. A game with `p_home` missing is dropped and counted.

Tests:

1. **Calibration at tipoff**, as in Stage 8: Brier score against the constant baseline (2.2), a and b with clustered standard errors (2.3), the LR test of a = 0 and b = 1 (2.4), and the block bootstrap by date (3). **Primary test: the LR test.**
   - Descriptive: a calibration plot with 5 equal-count bins and Wilson intervals.
2. **Popularity.** A team's measure is its 2025-26 season average pregame volume, the section 4 measure, computed from the 2025-26 main sample. It is known before the 2026-27 season starts. gap = log(home team's measure) − log(away team's measure).
   - Model of 9.1: logit P(y = 1) = a + b · logit(p) + c · gap, clustered by home team, with the block bootstrap. **Primary test: c = 0.**
   - c in plain terms as in 9.1: the SD of the gap, c · SD and the average effect on win probability.
   - If a 2026-27 team has no 2025-26 measure, stop and report.
3. **Opposite rule.** Threshold q = the 75th percentile (linear interpolation) of |gap| over the 1,229 games of the 2025-26 main sample, with the same season-average measure.
   - In every 2026-27 sample game with |gap| ≥ q, buy YES contracts on the more popular team (the one with the higher measure) at its ask at tipoff (`home_ask` or `away_ask` at horizon 0).
   - Stake $100, contracts = floor(100 / ask), and the taker fee of 9.3 computed once per order. The fee fields of the series were checked again on 2026-11-23: `fee_type = quadratic_with_maker_fees`, `fee_multiplier = 1`, no fee change listed.
   - No trade if the ask is missing or not strictly between 0 and 1. Report the counts.
   - Report the number of trades, win rate, total cost, fees, profit, return per dollar staked = profit / (total cost + total fees), and profit and return before fees.
   - **Primary test: the return per dollar staked against zero**, with the block bootstrap by date (trades fixed), percentile interval and bootstrap p value. The interval for profit is reported too.
   - Benchmark (secondary): the same calculation for buying the more popular team, with the same stake and fee, in every 2026-27 sample game. A game with a gap of exactly 0 has no more popular team and no trade. Its return is tested against zero the same way, as a secondary test.
4. **Sample size.** The sample is about one month of games. The notebook states this, and reports the width of the 95% interval for the slope b (clustered and bootstrap) next to the 2025-26 width.

### 10.7 All tests and outputs

- Each of notebooks 06 to 09 saves its tests in one common layout as `results/stage10_tests_<part>.csv`. Notebook `10_all_tests` reads those four files, adds the Holm adjusted p values and writes `results/stage10_all_tests.csv`. It fits no model.
- Data, in `data/processed/`: `price_paths.csv.gz`; the injury report and minutes files of 10.4; the 2026-27 files of 10.6.
- Notebook 06, in `results/`: `line_movement_logistic.csv`, `line_movement_bootstrap.csv`, `line_movement_effects.csv`, `line_movement_summary.csv`, `line_movement_bins.csv`, `line_movement_distribution.png`, `line_movement_bins.png`.
- Notebook 07, in `results/`: `arrival_by_minutes.csv`, `arrival_by_clock.csv`, `arrival_window_shares.csv`, `arrival_by_minutes.png`, `arrival_by_clock.png`, `injury_event_counts.csv`, `injury_events.csv`, `injury_event_test.csv`, `injury_event_path.csv`, `injury_event_path.png`.
- Notebook 08, in `results/`: `subgroups_groups.csv`, `subgroups_metrics.csv`, `subgroups_interaction.csv`, `subgroups_brier_diff.csv`, `subgroups_calibration_bins.csv`, and one calibration figure per split, `subgroups_calibration_liquidity.png`, `subgroups_calibration_season_half.png`, `subgroups_calibration_rest.png`.
- Notebook 09, in `results/`: `replication_sample_counts.csv`, `replication_brier.csv`, `replication_logistic.csv`, `replication_lr_test.csv`, `replication_bootstrap.csv`, `replication_calibration_bins.csv`, `replication_calibration_plot.png`, `replication_popularity_logistic.csv`, `replication_popularity_effects.csv`, `replication_popularity_bootstrap.csv`, `replication_opposite_rule.csv`, `replication_opposite_trades.csv`.
- Every table shown in a notebook is saved. A table not listed here is recorded under "Deviations".

### 10.8 Not in Stage 10

- Any other split, cut point, window length, event definition, trading rule or threshold, and any search over them.
- The opposite rule on the 2025-26 season.
- Any model or test not listed above. Ideas that come up are listed as suggestions and are not run.

## Exploratory, added after Stage 10 results

Added on 2026-11-23, after the Stage 10 results were seen. Nothing in this section was planned in advance, so its results are exploratory and carry less weight than the planned tests of Stage 10. This text was committed before either check was run. The planned results and their files are not changed, with one exception: `results/stage10_all_tests.csv` gains the rows of these checks, marked `exploratory`. They get no Holm adjustment, and the Holm family stays the 8 primary tests of 10.1.

Also decided on 2026-11-23, after the Stage 10 results: `tipoff_moved` for 2026-27 stays as computed, with the 2-day rule recorded under "Deviations".

### X1. Injury events by the player's previous status (notebook 07)

Reason: the event study of 10.4 pools events in which the player was Questionable, Doubtful or not listed before he was ruled Out. A player who was Doubtful was already expected to miss the game, and one who was not listed was not. This check asks whether the size of the price change, at the release and in the hour before it, goes with how much of a surprise the news was.

- Events: the 302 events of the main window in 10.4. Nothing about the events, the windows or the release times changes.
- Groups, by the previous status of the event's key players:
  - Questionable: every key player of the event was Questionable in the snapshot before;
  - Doubtful: every key player was Doubtful;
  - not listed: every key player was not listed, while the team had already submitted.
  - Events that fit none of the three (a previous status of Probable or Available, or several key players with different statuses) are left out. Their count is reported.
- For each group and for each of the two windows (main: 5 minutes before the release to 30 minutes after; anticipation: 60 to 5 minutes before): the number of events and game dates, the mean and median change, the 95% bootstrap interval of the mean and the bootstrap p value against zero. The block bootstrap is by game date within the group, B = 2,000, seed 4981.
- That is 6 means. No test of a difference between the groups is run, and no placebo is recomputed by group.
- Limit on the reading: the not listed group is small, and the groups can differ in other ways, such as how long before tipoff the report came out and which players are in them.

### X2. Calibration slope by calendar month (notebook 08)

Reason: the season-half test of 10.5 found a lower slope in the early half of 2025-26 than in the late half. A cut at one date cannot show whether the slope changed at one point or drifted over the season. This check shows the slope month by month.

- Sample: the main sample at tipoff, 1,229 games, as in 10.5.
- Groups: the calendar month of `game_date` (US Eastern). A first or last month with fewer than 100 games is combined with the month next to it (October with November, April with March). The number of games in each group is reported.
- For each group: the recalibration fit logit P(y = 1) = a + b · logit(p) within the group, with standard errors clustered by home team. Report a and b, their clustered standard errors and 95% intervals, and the Wald test of b = 1.
- Figure: b by month with its clustered 95% interval and a reference line at 1.
- No test of a difference between months is run, and no other grouping of the dates is tried.
- Limit on the reading: these are 5 to 7 unadjusted tests on small groups, so a single month with p below 0.05 is weak evidence. January holds both halves of the 10.5 split, whose cut is January 16.

### Outputs

- Notebook 07: `results/exploratory_injury_by_status.csv`.
- Notebook 08: `results/exploratory_slope_by_month.csv`, `results/exploratory_slope_by_month.png`.
- The tests of the two checks are saved in the common layout as `results/stage10_tests_exploratory_information_arrival.csv` and `results/stage10_tests_exploratory_subgroups.csv`, with the role `exploratory`. Notebook 10 adds them to `results/stage10_all_tests.csv`.

## Deviations

Stage 8: none.

Stage 9: sections 9.1 to 9.3 ran as planned, with 9.3 as amended before any result was computed. No planned test, sample or rule was changed after results were seen. Two details are recorded here because the plan did not settle them:

- 9.1, average effect: the plan did not say which standard deviation the bootstrap uses. Every replicate keeps the step at the full-sample SD of the gap, so all replicates measure the effect of the same change.
- 9.4, outputs: three small tables are saved in addition to the listed files, because every table shown in a notebook is saved: `horizons_sample_counts.csv`, `horizons_price_source.csv` and `strategy_sample_counts.csv`. `strategy_results.csv` also reports the average ask and the number of contracts of each strategy.

Exploratory checks X1 and X2 (after Stage 10), run on 2026-11-24: both ran as written. In X2, October (80 games) and April (96 games) are each under 100 games, so October is combined with November and April with March, which gives five groups. The planned Stage 10 files are unchanged, apart from the exploratory rows added to `stage10_all_tests.csv`.

Stage 10, recorded part by part as each one is run.

Data (10.2), run on 2026-11-23:

- The rebuilt `game_map.csv` and `snapshots.csv` are byte identical to the committed files.
- Kalshi's market list no longer returns the field `expiration_time`, which the match uses for `kalshi_tipoff_estimate`. The same time is now in `latest_expiration_time`: minus 14 days, it equals the committed `kalshi_tipoff_estimate` for all 1,230 games. `src/match_games.py` reads the old field and falls back to the new one. No data changed.

Information arrival (10.4), run on 2026-11-23: D1 and D2 ran as planned. The reliability rule passed on all three checks, so D2 was not stopped and the Holm family has all 8 tests. No definition, window or test was changed after results were seen. These points are recorded:

- D1, the 30h to 24h window: the plan said this window is understated by construction. It is not, in the main. 97% of its game-steps have a price change, and only its first two steps miss many games. It is still shown apart and left out of the total, as planned.
- D2, scheduled time: the plan took the scheduled time from the file name. Hourly file names carry no minute, so the scheduled time is the report time printed in the file's header, which was read for every parsed file. It differs from the file name's half-past-the-hour rule for 66 files of December 19 to 22, 2025, which are stamped at a quarter to the hour. This changes no release time: every event uses the posting time.
- D2, fetching: `nbainjuries` supplies the file names and the table parser. The files are downloaded by the project's own script, so that the server's `Last-Modified` time can be recorded, and are then parsed with the package from disk. The NBA's delivery network answered a first fast download with temporary 403 blocks for 4,970 files. Those were told apart from files that do not exist (a 403 from the file store itself) and fetched again slowly. In the end all 11,753 files that exist were fetched and parsed. The 484 that do not exist are the All-Star break (480) and the hour skipped on March 8, 2026 (4).
- D2, hand check: the 20 drawn events were checked by reading the plain text of the two original PDF files of each event a second way (PyPDF2, not the table parser) and comparing the game, the player, both statuses and the report time. Two of the 20 were also read line by line. All 20 agree. The check table is saved as `data/processed/injury_event_check_2025_26.csv`.
- D2, one snapshot (1:30 am on November 2, 2025, when the clocks changed back) has an ambiguous scheduled time and no release time. No event falls on it.
- Outputs: one table is saved in addition to the listed files, `injury_reliability.csv`. `data/processed/injury_report_games_2025_26.csv` holds the games seen in the reports, for the coverage check.

Replication (10.6), pulled and run on 2026-11-23: the tests ran as planned. The cutoff date is 2026-11-22 and the sample is 248 games. No test, measure, threshold or rule was changed after results were seen. Four points on the data are recorded here:

- `tipoff_moved` is computed for 2026-27, not left empty. The live market list has `latest_expiration_time`, the field that replaced `expiration_time`. But in 2026-27 Kalshi sets it 2 days after the listed tipoff, not 14: it is exactly 48 hours after the official tipoff in 246 of the 248 games. With that 2-day rule, the other two games are flagged `tipoff_moved` (official tipoff 1 hour and 1.5 hours from the listed time). Both stay in the sample, as planned. The rule is a season setting in `src/seasons.py` and is named in the match report.
- Some 2026-27 markets opened before Kalshi's historical cutoff (2026-09-24). Their trades before the cutoff are on the historical endpoints and the later ones on the live endpoints. The pull reads both and keeps each trade once. Every 30h window lies after the cutoff.
- The live candles name their prices `close_dollars`, not `close`. The loader reads both.
- One table is saved in addition to the listed files: `replication_slope_width.csv` (the interval widths of 10.6, step 4). A data quality report for the new season is saved as `data/processed/quality_report_2026_27.md`.

Line movement (10.3) and subgroups (10.5), run on 2026-11-23: both ran as planned. No test, sample, split or cut point was changed after results were seen. Two tables are saved in addition to the listed files, because every table shown or used in a notebook is saved: `line_movement_sample_counts.csv` and `subgroups_bootstrap.csv`.

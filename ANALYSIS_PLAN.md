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

## Deviations

Stage 8: none.

Stage 9: sections 9.1 to 9.3 ran as planned, with 9.3 as amended before any result was computed. No planned test, sample or rule was changed after results were seen. Two details are recorded here because the plan did not settle them:

- 9.1, average effect: the plan did not say which standard deviation the bootstrap uses. Every replicate keeps the step at the full-sample SD of the gap, so all replicates measure the effect of the same change.
- 9.4, outputs: three small tables are saved in addition to the listed files, because every table shown in a notebook is saved: `horizons_sample_counts.csv`, `horizons_price_source.csv` and `strategy_sample_counts.csv`. `strategy_results.csv` also reports the average ask and the number of contracts of each strategy.

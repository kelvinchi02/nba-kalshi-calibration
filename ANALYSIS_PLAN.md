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

## Deviations

None so far.

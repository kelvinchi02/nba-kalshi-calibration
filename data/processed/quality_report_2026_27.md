# Data quality report (2026-27 regular season)

Coverage and data quality only. No calibration results.

## Games and markets

- NBA regular-season games: 248
- Matched to Kalshi: 248 (exact 248, +/-1 day 0)
- main_sample (matched, not postponed_window): 248
- postponed_window: 
- tipoff_moved: KXNBAGAME-26OCT29CLEATL, KXNBAGAME-26NOV18GSWDAL
- Market files cached: 496 of 496; failures in last pull run: 0
- Snapshot rows: 992 (248 games x 4 horizons)

## Missing snapshots per horizon

mid/last/vwap10 = that price type missing on at least one side. p_home = main estimate (mid, else last, else vwap10).

| horizon_h | rows | market_not_open | mid_missing | one_sided_quote | last_missing | vwap10_missing | p_home_missing | src_mid | src_last | src_vwap |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 248 | 0 | 0 | 0 | 0 | 96 | 0 | 248 | 0 | 0 |
| 6 | 248 | 0 | 0 | 0 | 0 | 3 | 0 | 248 | 0 | 0 |
| 1 | 248 | 0 | 0 | 0 | 0 | 0 | 0 | 248 | 0 | 0 |
| 0 | 248 | 0 | 0 | 0 | 0 | 0 | 0 | 248 | 0 | 0 |

Rows with no price at all:

(none)

## Quote age (minutes since the candle used for the midpoint, worse side)

Minimum is 1 minute by construction (the candle ending at the snapshot is not used).

| horizon_h | count | 50% | 95% | 99% | max | over_30min |
|---|---|---|---|---|---|---|
| 24 | 248.0 | 2.0 | 14.0 | 50.5 | 77.0 | 6 |
| 6 | 248.0 | 1.0 | 3.0 | 4.0 | 4.0 | 0 |
| 1 | 248.0 | 1.0 | 1.0 | 2.0 | 2.0 | 0 |
| 0 | 248.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |

## Bid/ask spread (dollars, from the candle used for the midpoint)

Distribution over both team markets:

| horizon_h | count | mean | 50% | 90% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|
| 24 | 496.000 | 0.014 | 0.010 | 0.020 | 0.020 | 0.030 | 0.040 |
| 6 | 496.000 | 0.010 | 0.010 | 0.010 | 0.010 | 0.020 | 0.020 |
| 1 | 496.000 | 0.010 | 0.010 | 0.010 | 0.010 | 0.020 | 0.020 |
| 0 | 496.000 | 0.010 | 0.010 | 0.010 | 0.010 | 0.011 | 0.020 |

Rows where either side's spread is above 5 cents (max_spread > 0.05):

| horizon_h | rows | rows_spread_gt_5c | home_gt_5c | away_gt_5c | rows_spread_missing |
|---|---|---|---|---|---|
| 24 | 248 | 0 | 0 | 0 | 0 |
| 6 | 248 | 0 | 0 | 0 | 0 |
| 1 | 248 | 0 | 0 | 0 | 0 |
| 0 | 248 | 0 | 0 | 0 | 0 |

Widest spreads:

| event_ticker | horizon_h | home_bid | home_ask | away_bid | away_ask | max_spread | home_one_sided | away_one_sided |
|---|---|---|---|---|---|---|---|---|
| KXNBAGAME-26OCT24HOUATL | 24 | 0.470 | 0.510 | 0.500 | 0.520 | 0.040 | False | False |
| KXNBAGAME-26OCT31PORUTA | 24 | 0.300 | 0.330 | 0.670 | 0.710 | 0.040 | False | False |
| KXNBAGAME-26OCT24MIACHA | 24 | 0.460 | 0.490 | 0.520 | 0.530 | 0.030 | False | False |
| KXNBAGAME-26OCT24MILPHI | 24 | 0.830 | 0.850 | 0.150 | 0.180 | 0.030 | False | False |
| KXNBAGAME-26OCT24WASCHI | 24 | 0.450 | 0.470 | 0.520 | 0.550 | 0.030 | False | False |
| KXNBAGAME-26OCT26MEMUTA | 24 | 0.520 | 0.550 | 0.470 | 0.490 | 0.030 | False | False |
| KXNBAGAME-26OCT28MIAATL | 24 | 0.570 | 0.580 | 0.410 | 0.440 | 0.030 | False | False |
| KXNBAGAME-26OCT31CLEMIL | 24 | 0.240 | 0.260 | 0.730 | 0.760 | 0.030 | False | False |
| KXNBAGAME-26NOV01BOSORL | 24 | 0.310 | 0.340 | 0.660 | 0.680 | 0.030 | False | False |
| KXNBAGAME-26NOV02TORLAC | 24 | 0.350 | 0.380 | 0.630 | 0.650 | 0.030 | False | False |

## Low-volume games

Contracts traded (both markets) in the 30h before tipoff:

min 81,897, p1 111,556, p5 196,891, p25 394,486, median 659,184, max 6,508,705

Games at or below the 2% quantile (140,488 contracts):

| event_ticker | window_contracts | trades_10m | market_open_before_snapshot |
|---|---|---|---|
| KXNBAGAME-26OCT28DENNOP | 81,897 | 69 | True |
| KXNBAGAME-26OCT30LACMIN | 94,943 | 153 | True |
| KXNBAGAME-26OCT28ORLTOR | 104,775 | 97 | True |
| KXNBAGAME-26OCT24BKNCLE | 119,203 | 90 | True |
| KXNBAGAME-26NOV18ATLOKC | 123,779 | 121 | True |

Games with no trade in the 10 minutes before tipoff (either side): 0

## Full pregame volume (market open to tipoff)

Contracts traded (both markets), strictly before the official tipoff:

min 82,754, p1 118,849, p5 200,979, p25 401,950, median 674,375, max 6,544,712

Share of pregame volume traded in the last 30h: median 0.987, min 0.879

Consistency checks (each should be 0):

- games with pregame volume missing: 0
- games where pregame < 30h-window volume: 0
- games where pregame > Kalshi lifetime volume (incl. in-game): 0

## Two-sided sums before normalization (home + away)

Far from 1 means the two markets disagree (spread, stale price or thin book).

| price | horizon_h | n | median | min | max | abs_dev_gt_0.05 | abs_dev_gt_0.1 |
|---|---|---|---|---|---|---|---|
| last | 24 | 248 | 1.010 | 0.980 | 1.040 | 0 | 0 |
| last | 6 | 248 | 1.010 | 0.990 | 1.020 | 0 | 0 |
| last | 1 | 248 | 1.010 | 0.990 | 1.020 | 0 | 0 |
| last | 0 | 248 | 1.010 | 0.990 | 1.020 | 0 | 0 |
| mid | 24 | 248 | 1.000 | 0.980 | 1.025 | 0 | 0 |
| mid | 6 | 248 | 1.000 | 0.990 | 1.010 | 0 | 0 |
| mid | 1 | 248 | 1.000 | 0.990 | 1.010 | 0 | 0 |
| mid | 0 | 248 | 1.000 | 0.990 | 1.010 | 0 | 0 |
| vwap_10m | 24 | 152 | 1.010 | 0.990 | 1.040 | 0 | 0 |
| vwap_10m | 6 | 245 | 1.010 | 0.995 | 1.020 | 0 | 0 |
| vwap_10m | 1 | 248 | 1.010 | 0.980 | 1.020 | 0 | 0 |
| vwap_10m | 0 | 248 | 1.010 | 0.993 | 1.020 | 0 | 0 |

Largest midpoint-sum deviations:

| event_ticker | horizon_h | home_bid | home_ask | away_bid | away_ask | sum_mid |
|---|---|---|---|---|---|---|
| KXNBAGAME-26NOV01LACGSW | 24 | 0.670 | 0.680 | 0.340 | 0.360 | 1.025 |
| KXNBAGAME-26OCT25LALUTA | 24 | 0.280 | 0.300 | 0.680 | 0.700 | 0.980 |
| KXNBAGAME-26NOV22OKCMIA | 24 | 0.650 | 0.660 | 0.360 | 0.370 | 1.020 |
| KXNBAGAME-26OCT26MEMUTA | 24 | 0.520 | 0.550 | 0.470 | 0.490 | 1.015 |
| KXNBAGAME-26NOV18MEMCLE | 24 | 0.640 | 0.660 | 0.350 | 0.380 | 1.015 |
| KXNBAGAME-26NOV18PHIORL | 24 | 0.450 | 0.470 | 0.550 | 0.560 | 1.015 |
| KXNBAGAME-26NOV18DETNOP | 24 | 0.290 | 0.300 | 0.680 | 0.700 | 0.985 |
| KXNBAGAME-26NOV21PORNYK | 24 | 0.640 | 0.660 | 0.360 | 0.370 | 1.015 |
| KXNBAGAME-26NOV13MILORL | 24 | 0.760 | 0.780 | 0.210 | 0.220 | 0.985 |
| KXNBAGAME-26NOV16GSWNOP | 24 | 0.330 | 0.350 | 0.640 | 0.650 | 0.985 |

## Flagged games: snapshot availability

| event_ticker | horizon_h | postponed_window | tipoff_moved | price_source | sum_mid |
|---|---|---|---|---|---|
| KXNBAGAME-26OCT29CLEATL | 24 | False | True | mid | 1.010 |
| KXNBAGAME-26OCT29CLEATL | 6 | False | True | mid | 1.000 |
| KXNBAGAME-26OCT29CLEATL | 1 | False | True | mid | 1.000 |
| KXNBAGAME-26OCT29CLEATL | 0 | False | True | mid | 1.000 |
| KXNBAGAME-26NOV18GSWDAL | 24 | False | True | mid | 1.000 |
| KXNBAGAME-26NOV18GSWDAL | 6 | False | True | mid | 1.000 |
| KXNBAGAME-26NOV18GSWDAL | 1 | False | True | mid | 1.000 |
| KXNBAGAME-26NOV18GSWDAL | 0 | False | True | mid | 1.000 |


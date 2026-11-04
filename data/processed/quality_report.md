# Data quality report (2025-26 regular season)

Coverage and data quality only. No calibration results.

## Games and markets

- NBA regular-season games: 1230
- Matched to Kalshi: 1230 (exact 1229, +/-1 day 1)
- main_sample (matched, not postponed_window): 1229
- postponed_window: KXNBAGAME-26JAN24GSWMIN
- tipoff_moved: KXNBAGAME-26JAN24WASCHA, KXNBAGAME-26JAN26INDATL, KXNBAGAME-26JAN26PHICHA, KXNBAGAME-26FEB01ORLSAS, KXNBAGAME-26FEB02NOPCHA
- Market files cached: 2460 of 2460; failures in last pull run: 0
- Snapshot rows: 4920 (1230 games x 4 horizons)

## Missing snapshots per horizon

mid/last/vwap10 = that price type missing on at least one side. p_home = main estimate (mid, else last, else vwap10).

| horizon_h | rows | market_not_open | mid_missing | one_sided_quote | last_missing | vwap10_missing | p_home_missing | src_mid | src_last | src_vwap |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 1230 | 7 | 7 | 0 | 7 | 558 | 7 | 1223 | 0 | 0 |
| 6 | 1230 | 0 | 0 | 0 | 0 | 16 | 0 | 1230 | 0 | 0 |
| 1 | 1230 | 0 | 0 | 0 | 0 | 2 | 0 | 1230 | 0 | 0 |
| 0 | 1230 | 0 | 0 | 0 | 0 | 0 | 0 | 1230 | 0 | 0 |

Rows with no price at all:

| event_ticker | horizon_h | market_open_before_snapshot |
|---|---|---|
| KXNBAGAME-26JAN31SASCHA | 24 | False |
| KXNBAGAME-26JAN31ATLIND | 24 | False |
| KXNBAGAME-26JAN31NOPPHI | 24 | False |
| KXNBAGAME-26JAN31CHIMIA | 24 | False |
| KXNBAGAME-26JAN31MINMEM | 24 | False |
| KXNBAGAME-26JAN31DALHOU | 24 | False |
| KXNBAGAME-26MAR11CHASAC | 24 | False |

## Quote age (minutes since the candle used for the midpoint, worse side)

Minimum is 1 minute by construction (the candle ending at the snapshot is not used).

| horizon_h | count | 50% | 95% | 99% | max | over_30min |
|---|---|---|---|---|---|---|
| 24 | 1223.0 | 2.0 | 10.0 | 22.0 | 49.0 | 3 |
| 6 | 1230.0 | 1.0 | 2.0 | 4.0 | 6.0 | 0 |
| 1 | 1230.0 | 1.0 | 1.0 | 2.0 | 4.0 | 0 |
| 0 | 1230.0 | 1.0 | 1.0 | 1.0 | 2.0 | 0 |

## Bid/ask spread (dollars, from the candle used for the midpoint)

Distribution over both team markets:

| horizon_h | count | mean | 50% | 90% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|
| 24 | 2446.000 | 0.020 | 0.020 | 0.030 | 0.040 | 0.100 | 0.350 |
| 6 | 2460.000 | 0.011 | 0.010 | 0.020 | 0.020 | 0.020 | 0.030 |
| 1 | 2460.000 | 0.011 | 0.010 | 0.010 | 0.020 | 0.020 | 0.020 |
| 0 | 2460.000 | 0.011 | 0.010 | 0.010 | 0.020 | 0.020 | 0.030 |

Rows where either side's spread is above 5 cents (max_spread > 0.05):

| horizon_h | rows | rows_spread_gt_5c | home_gt_5c | away_gt_5c | rows_spread_missing |
|---|---|---|---|---|---|
| 24 | 1230 | 57 | 41 | 44 | 7 |
| 6 | 1230 | 0 | 0 | 0 | 0 |
| 1 | 1230 | 0 | 0 | 0 | 0 |
| 0 | 1230 | 0 | 0 | 0 | 0 |

Widest spreads:

| event_ticker | horizon_h | home_bid | home_ask | away_bid | away_ask | max_spread | home_one_sided | away_one_sided |
|---|---|---|---|---|---|---|---|---|
| KXNBAGAME-25OCT25INDMEM | 24 | 0.260 | 0.610 | 0.490 | 0.770 | 0.350 | False | False |
| KXNBAGAME-25NOV10MILDAL | 24 | 0.400 | 0.430 | 0.310 | 0.580 | 0.270 | False | False |
| KXNBAGAME-25OCT26NYKMIA | 24 | 0.220 | 0.450 | 0.550 | 0.780 | 0.230 | False | False |
| KXNBAGAME-25OCT26CHAWAS | 24 | 0.380 | 0.590 | 0.410 | 0.610 | 0.210 | False | False |
| KXNBAGAME-25OCT26INDMIN | 24 | 0.610 | 0.820 | 0.210 | 0.390 | 0.210 | False | False |
| KXNBAGAME-25OCT27PORLAL | 24 | 0.390 | 0.580 | 0.410 | 0.610 | 0.200 | False | False |
| KXNBAGAME-25OCT26BOSDET | 24 | 0.470 | 0.560 | 0.400 | 0.580 | 0.180 | False | False |
| KXNBAGAME-25OCT25OKCATL | 24 | 0.260 | 0.430 | 0.610 | 0.710 | 0.170 | False | False |
| KXNBAGAME-25OCT30ORLCHA | 24 | 0.310 | 0.460 | 0.530 | 0.620 | 0.150 | False | False |
| KXNBAGAME-25DEC20ORLUTA | 24 | 0.230 | 0.380 | 0.690 | 0.790 | 0.150 | False | False |

## Low-volume games

Contracts traded (both markets) in the 30h before tipoff:

min 98,908, p1 181,271, p5 246,797, p25 453,425, median 740,002, max 4,932,946

Games at or below the 2% quantile (204,220 contracts):

| event_ticker | window_contracts | trades_10m | market_open_before_snapshot |
|---|---|---|---|
| KXNBAGAME-25NOV07OKCSAC | 98,908 | 25 | True |
| KXNBAGAME-25DEC23MEMUTA | 106,085 | 46 | True |
| KXNBAGAME-26APR10BKNMIL | 115,957 | 28 | True |
| KXNBAGAME-26MAR21LACDAL | 125,942 | 55 | True |
| KXNBAGAME-25DEC27INDMIA | 128,214 | 63 | True |
| KXNBAGAME-26MAR27NOPTOR | 153,003 | 31 | True |
| KXNBAGAME-26MAR25WASUTA | 157,095 | 89 | True |
| KXNBAGAME-26MAR16MEMCHI | 162,644 | 60 | True |
| KXNBAGAME-26JAN07PHXMEM | 171,472 | 72 | True |
| KXNBAGAME-25DEC14MILBKN | 172,060 | 59 | True |
| KXNBAGAME-25NOV12PORNOP | 172,112 | 44 | True |
| KXNBAGAME-25DEC20INDNOP | 175,392 | 70 | True |
| KXNBAGAME-26FEB26NOPUTA | 180,648 | 92 | True |
| KXNBAGAME-25DEC14NOPCHI | 182,796 | 96 | True |
| KXNBAGAME-25DEC29MINCHI | 184,976 | 73 | True |
| KXNBAGAME-25NOV07DALMEM | 185,626 | 47 | True |
| KXNBAGAME-25NOV16BKNWAS | 188,313 | 93 | True |
| KXNBAGAME-25NOV28MEMLAC | 188,442 | 45 | True |
| KXNBAGAME-25DEC20PORSAC | 190,278 | 62 | True |
| KXNBAGAME-25DEC18TORMIL | 191,371 | 53 | True |
| KXNBAGAME-26APR05WASBKN | 191,765 | 90 | True |
| KXNBAGAME-26MAR10CHIGSW | 195,799 | 116 | True |
| KXNBAGAME-25DEC23MILIND | 196,418 | 91 | True |
| KXNBAGAME-25NOV01ORLWAS | 198,459 | 38 | True |
| KXNBAGAME-26MAR29SACBKN | 204,149 | 120 | True |

Games with no trade in the 10 minutes before tipoff (either side): 0

## Two-sided sums before normalization (home + away)

Far from 1 means the two markets disagree (spread, stale price or thin book).

| price | horizon_h | n | median | min | max | abs_dev_gt_0.05 | abs_dev_gt_0.1 |
|---|---|---|---|---|---|---|---|
| last | 24 | 1223 | 1.020 | 0.920 | 1.260 | 61 | 11 |
| last | 6 | 1230 | 1.010 | 0.980 | 1.030 | 0 | 0 |
| last | 1 | 1230 | 1.010 | 0.980 | 1.030 | 0 | 0 |
| last | 0 | 1230 | 1.010 | 0.980 | 1.030 | 0 | 0 |
| mid | 24 | 1223 | 1.005 | 0.860 | 1.065 | 4 | 1 |
| mid | 6 | 1230 | 1.000 | 0.980 | 1.020 | 0 | 0 |
| mid | 1 | 1230 | 1.000 | 0.980 | 1.020 | 0 | 0 |
| mid | 0 | 1230 | 1.000 | 0.990 | 1.025 | 0 | 0 |
| vwap_10m | 24 | 672 | 1.020 | 0.968 | 1.190 | 15 | 1 |
| vwap_10m | 6 | 1214 | 1.010 | 0.971 | 1.030 | 0 | 0 |
| vwap_10m | 1 | 1228 | 1.010 | 0.991 | 1.032 | 0 | 0 |
| vwap_10m | 0 | 1230 | 1.010 | 0.992 | 1.032 | 0 | 0 |

Largest midpoint-sum deviations:

| event_ticker | horizon_h | home_bid | home_ask | away_bid | away_ask | sum_mid |
|---|---|---|---|---|---|---|
| KXNBAGAME-25NOV10MILDAL | 24 | 0.400 | 0.430 | 0.310 | 0.580 | 0.860 |
| KXNBAGAME-25OCT25INDMEM | 24 | 0.260 | 0.610 | 0.490 | 0.770 | 1.065 |
| KXNBAGAME-25NOV02UTACHA | 24 | 0.420 | 0.500 | 0.440 | 0.540 | 0.950 |
| KXNBAGAME-25NOV04CHANOP | 24 | 0.380 | 0.510 | 0.480 | 0.530 | 0.950 |
| KXNBAGAME-26JAN14DENDAL | 24 | 0.440 | 0.530 | 0.410 | 0.530 | 0.955 |
| KXNBAGAME-25NOV20SACMEM | 24 | 0.600 | 0.610 | 0.430 | 0.450 | 1.045 |
| KXNBAGAME-25DEC20ORLUTA | 24 | 0.230 | 0.380 | 0.690 | 0.790 | 1.045 |
| KXNBAGAME-25OCT30ORLCHA | 24 | 0.310 | 0.460 | 0.530 | 0.620 | 0.960 |
| KXNBAGAME-25NOV05GSWSAC | 24 | 0.300 | 0.390 | 0.690 | 0.700 | 1.040 |
| KXNBAGAME-25NOV28CLEATL | 24 | 0.370 | 0.380 | 0.660 | 0.670 | 1.040 |

## Flagged games: snapshot availability

| event_ticker | horizon_h | postponed_window | tipoff_moved | price_source | sum_mid |
|---|---|---|---|---|---|
| KXNBAGAME-26JAN24WASCHA | 24 | False | True | mid | 1.000 |
| KXNBAGAME-26JAN24WASCHA | 6 | False | True | mid | 1.000 |
| KXNBAGAME-26JAN24WASCHA | 1 | False | True | mid | 1.010 |
| KXNBAGAME-26JAN24WASCHA | 0 | False | True | mid | 1.005 |
| KXNBAGAME-26JAN24GSWMIN | 24 | True | False | mid | 1.000 |
| KXNBAGAME-26JAN24GSWMIN | 6 | True | False | mid | 0.995 |
| KXNBAGAME-26JAN24GSWMIN | 1 | True | False | mid | 1.015 |
| KXNBAGAME-26JAN24GSWMIN | 0 | True | False | mid | 1.000 |
| KXNBAGAME-26JAN26INDATL | 24 | False | True | mid | 1.000 |
| KXNBAGAME-26JAN26INDATL | 6 | False | True | mid | 1.005 |
| KXNBAGAME-26JAN26INDATL | 1 | False | True | mid | 1.000 |
| KXNBAGAME-26JAN26INDATL | 0 | False | True | mid | 1.010 |
| KXNBAGAME-26JAN26PHICHA | 24 | False | True | mid | 1.015 |
| KXNBAGAME-26JAN26PHICHA | 6 | False | True | mid | 1.010 |
| KXNBAGAME-26JAN26PHICHA | 1 | False | True | mid | 1.000 |
| KXNBAGAME-26JAN26PHICHA | 0 | False | True | mid | 1.000 |
| KXNBAGAME-26FEB01ORLSAS | 24 | False | True | mid | 1.005 |
| KXNBAGAME-26FEB01ORLSAS | 6 | False | True | mid | 1.000 |
| KXNBAGAME-26FEB01ORLSAS | 1 | False | True | mid | 1.000 |
| KXNBAGAME-26FEB01ORLSAS | 0 | False | True | mid | 1.010 |
| KXNBAGAME-26FEB02NOPCHA | 24 | False | True | mid | 1.005 |
| KXNBAGAME-26FEB02NOPCHA | 6 | False | True | mid | 0.980 |
| KXNBAGAME-26FEB02NOPCHA | 1 | False | True | mid | 1.000 |
| KXNBAGAME-26FEB02NOPCHA | 0 | False | True | mid | 1.010 |


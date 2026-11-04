# Kalshi to NBA match report (2025-26 regular season)

- NBA regular-season games: 1230
- Matched to a Kalshi event: 1230 (exact 1229, +/-1 day 1)
- Main sample (matched, not postponed_window): 1229
- tipoff_moved (official tipoff 1-12h from Kalshi's listed time): 5
- Kalshi settlement agrees with NBA home_win: 1230 of 1230
- Kalshi 2025-26 events: 1363; status counts: matched 1230, unmatched 129, dropped_postponed_original 3, non_nba_team 1

## NBA games with no Kalshi event

(none)

## Flagged: postponed_window (excluded from main sample)

| game_id | event_ticker | kalshi_game_date | game_date | tipoff_utc | kalshi_tipoff_estimate |
|---|---|---|---|---|---|
| 0022500644 | KXNBAGAME-26JAN24GSWMIN | 2026-01-24 | 2026-01-25 | 2026-01-25 22:30:00+00:00 | 2026-01-24 22:30:00+00:00 |

## Flagged: tipoff_moved (kept in main sample)

| game_id | event_ticker | tipoff_utc | kalshi_tipoff_estimate | tipoff_shift_hours |
|---|---|---|---|---|
| 0022500645 | KXNBAGAME-26JAN24WASCHA | 2026-01-24 17:00:00+00:00 | 2026-01-24 23:00:00+00:00 | -6.0 |
| 0022500659 | KXNBAGAME-26JAN26INDATL | 2026-01-26 18:30:00+00:00 | 2026-01-27 00:30:00+00:00 | -6.0 |
| 0022500657 | KXNBAGAME-26JAN26PHICHA | 2026-01-26 20:00:00+00:00 | 2026-01-27 00:00:00+00:00 | -4.0 |
| 0022500703 | KXNBAGAME-26FEB01ORLSAS | 2026-02-02 02:00:00+00:00 | 2026-02-01 21:00:00+00:00 | 5.0 |
| 0022500712 | KXNBAGAME-26FEB02NOPCHA | 2026-02-02 20:00:00+00:00 | 2026-02-03 00:00:00+00:00 | -4.0 |

## Kalshi events dropped: postponed originals (settled scalar)

| event_ticker | home_result | away_result |
|---|---|---|
| KXNBAGAME-26JAN25DALMIL | scalar | scalar |
| KXNBAGAME-26JAN25DENMEM | scalar | scalar |
| KXNBAGAME-26JAN08MIACHI | scalar | scalar |

## Kalshi events not matched to a regular-season game

Counts by phase (from game date): playoffs 81, preseason 43, play_in 6

Unmatched events dated inside the regular season:

(none)

All other unmatched events are preseason, play-in or playoff games, which are outside the regular-season sample.

## Team code checks

All Kalshi team names match the aliases in team_code_map.csv.

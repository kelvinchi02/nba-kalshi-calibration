# Kalshi to NBA match report (2026-27 regular season)

Season in progress. Games and Kalshi events up to 2026-11-22, market list from the live endpoints.

The listed tipoff (kalshi_tipoff_estimate) is the market's latest_expiration_time minus 2 days.

- NBA regular-season games: 248
- Matched to a Kalshi event: 248 (exact 248, +/-1 day 0)
- Main sample (matched, not postponed_window): 248
- tipoff_moved (official tipoff 1-12h from Kalshi's listed time): 2
- Kalshi settlement agrees with NBA home_win: 248 of 248
- Kalshi 2026-27 events: 315; status counts: matched 248, unmatched 66, non_nba_team 1

## NBA games with no Kalshi event

(none)

## Flagged: postponed_window (excluded from main sample)

(none)

## Flagged: tipoff_moved (kept in main sample)

| game_id | event_ticker | tipoff_utc | kalshi_tipoff_estimate | tipoff_shift_hours |
|---|---|---|---|---|
| 0022600144 | KXNBAGAME-26OCT29CLEATL | 2026-10-30 00:00:00+00:00 | 2026-10-29 23:00:00+00:00 | 1.0 |
| 0022600267 | KXNBAGAME-26NOV18GSWDAL | 2026-11-19 01:00:00+00:00 | 2026-11-19 02:30:00+00:00 | -1.5 |

## Kalshi events dropped: postponed originals (settled scalar)

(none)

## Kalshi events not matched to a regular-season game

Counts by phase (from game date): preseason 67

Unmatched events dated inside the regular season:

(none)

All other unmatched events are preseason, play-in or playoff games, which are outside the regular-season sample.

## Team code checks

All Kalshi team names match the aliases in team_code_map.csv.

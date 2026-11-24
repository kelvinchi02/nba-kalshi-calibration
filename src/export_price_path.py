"""Save one example game's 30-hour price path to data/processed/example_price_path.csv.

This is the only step of the EDA that needs the raw cache (data/raw/, gitignored).
notebooks/01_eda.ipynb reads the saved CSV, so it runs without the cache.

One row per team market per minute of the 30 hours before the official tipoff: closing bid
and ask (carried forward through minutes without a candle), their midpoint (missing for a
one-sided book) and the contracts traded in that minute.

Default game: among main-sample games with above-median pregame volume, the one whose
p_home moved most between the 24h snapshot and tipoff. --event picks another game.

Usage:
    python src/export_price_path.py
    python src/export_price_path.py --event KXNBAGAME-25DEC13SASOKC
"""
from __future__ import annotations

import argparse

from csv_io import write_csv
from eda import PRICE_PATH_CSV, REPO_ROOT, main_sample, pick_example, price_path_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", help="event ticker (default: chosen by the rule above)")
    args = parser.parse_args()
    sn = main_sample()
    event = args.event or pick_example(sn)
    pp = price_path_table(sn, event)
    write_csv(pp, PRICE_PATH_CSV)
    print(f"{event}: {len(pp):,} rows, {pp['contracts'].sum():,.0f} contracts "
          f"-> {PRICE_PATH_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

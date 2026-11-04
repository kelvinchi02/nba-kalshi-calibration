"""CSV output tests.

1. Round trip: write_csv then read_csv returns the same types and values.
2. Each processed CSV reads back with the same row count, columns and values as the
   parquet file it replaced. The parquet fingerprints were recorded at the switch in
   tests/fixtures/processed_fingerprint.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csv_io import read_csv, write_csv
from fingerprint import fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "processed_fingerprint.json"
PROCESSED = REPO_ROOT / "data" / "processed"


def test_round_trip_types_and_values(tmp_path):
    df = pd.DataFrame({
        "game_id": ["0022500001", "0022500002"],
        "game_date": [pd.Timestamp("2025-10-21").date(), pd.Timestamp("2026-01-25").date()],
        "tipoff_utc": pd.to_datetime(["2025-10-21T23:30:00Z", "2026-01-25T22:30:00Z"], utc=True),
        "market_open_time": pd.to_datetime(["2025-10-19T22:41:00Z", None], utc=True),
        "main_sample": [True, False],
        "home_win": pd.array([1, pd.NA], dtype="Int64"),
        "home_mid": [0.695, np.nan],
        "sum_mid": [0.1 + 0.2, 1.0],  # 0.30000000000000004 must survive exactly
        "price_source": ["mid", None],
    })
    path = tmp_path / "x.csv"
    write_csv(df, path)
    text = path.read_text(encoding="utf-8")
    assert "2025-10-21T23:30:00Z" in text            # ISO 8601, UTC, Z suffix
    assert text.splitlines()[0].split(",")[0] == "game_id"  # no index column
    assert ",1," in text.splitlines()[1]               # booleans as 0/1
    back = read_csv(path)
    assert fingerprint(back) == fingerprint(df)
    assert back["game_id"].tolist() == ["0022500001", "0022500002"]
    assert back["sum_mid"].iloc[0] == 0.1 + 0.2
    assert back["main_sample"].dtype == bool


def test_missing_values_are_empty_cells(tmp_path):
    df = pd.DataFrame({"a": [1.5, np.nan], "b": ["x", None]})
    path = tmp_path / "y.csv"
    write_csv(df, path)
    assert path.read_text(encoding="utf-8").splitlines()[2] == ","


EXPECTED = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.exists() else {}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_csv_matches_replaced_parquet(name):
    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"{name} not built yet")
    got, want = fingerprint(read_csv(path)), EXPECTED[name]
    assert got["rows"] == want["rows"]
    assert got["columns"] == want["columns"]
    bad = [c for c in want["columns"]
           if got["column_fingerprints"][c] != want["column_fingerprints"][c]]
    assert not bad, f"columns differ from the parquet original: {bad}"

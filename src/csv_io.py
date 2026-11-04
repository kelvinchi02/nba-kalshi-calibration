"""Read and write the CSV files in data/processed with one set of conventions.

Writing:
  * timestamps: ISO 8601 in UTC with a Z suffix, e.g. 2026-01-25T22:30:00Z
  * dates: YYYY-MM-DD
  * booleans: 1 or 0
  * missing values: empty cells
  * UTF-8, no index column, '\n' line endings

Reading turns the text back into typed columns. Column types are fixed by name below,
because a CSV cell cannot tell 0/1 booleans from integers or a zero-padded ID from a number.
Columns not listed are inferred: integers -> Int64, other numbers -> float64, else text.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATETIME_COLUMNS = {"tipoff_utc", "snapshot_utc", "kalshi_tipoff_estimate", "market_open_time"}
DATE_COLUMNS = {"game_date", "kalshi_game_date"}
STRING_COLUMNS = {"game_id"}  # zero-padded NBA IDs, e.g. 0022500001
BOOL_COLUMNS = {"postponed_window", "tipoff_moved", "matched", "main_sample",
                "market_open_before_snapshot", "cache_missing"}
BOOL_SUFFIXES = ("_mid_carried_forward", "_one_sided")

_INT_RE = re.compile(r"^-?\d+$")


def is_bool_column(name: str) -> bool:
    return name in BOOL_COLUMNS or name.endswith(BOOL_SUFFIXES)


def _format_timestamps(s: pd.Series) -> pd.Series:
    s = s.dt.tz_convert("UTC")
    has_fraction = (s.dropna().dt.microsecond != 0).any()
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if has_fraction else "%Y-%m-%dT%H:%M:%SZ"
    return s.dt.strftime(fmt).where(s.notna(), "")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out[col] = _format_timestamps(s)
        elif pd.api.types.is_datetime64_dtype(s.dtype):
            raise ValueError(f"{col}: timestamps must be timezone-aware UTC")
        elif col in DATE_COLUMNS:
            out[col] = pd.to_datetime(s).dt.strftime("%Y-%m-%d").where(s.notna(), "")
        elif is_bool_column(col):
            out[col] = s.astype("boolean").astype("Int64")
    out.to_csv(path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")


def read_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    out = {}
    for col in raw.columns:
        text = raw[col]
        empty = text == ""
        if col in DATETIME_COLUMNS:
            out[col] = pd.to_datetime(text.where(~empty), utc=True, format="ISO8601")
        elif col in DATE_COLUMNS:
            out[col] = pd.to_datetime(text.where(~empty), format="%Y-%m-%d").dt.date
        elif is_bool_column(col):
            values = text.map({"1": True, "0": False, "": pd.NA})
            if values.isna().any():
                out[col] = values.astype("boolean")
            else:
                out[col] = values.astype(bool)
        elif col in STRING_COLUMNS:
            out[col] = text.where(~empty, None).astype("str")
        else:
            out[col] = _infer(text, empty)
    return pd.DataFrame(out)


def _infer(text: pd.Series, empty: pd.Series) -> pd.Series:
    present = text[~empty]
    if present.empty:
        return text.where(~empty, None).astype("str")
    if present.map(lambda v: bool(_INT_RE.match(v))).all():
        return pd.array(pd.to_numeric(text.where(~empty)), dtype="Int64")
    numeric = pd.to_numeric(present, errors="coerce")
    if numeric.notna().all():
        # Python float() parses the shortest repr exactly (round trip).
        return text.map(lambda v: float(v) if v != "" else float("nan")).astype("float64")
    return text.where(~empty, None).astype("str")

"""Type-aware fingerprint of a DataFrame, used to check the CSV outputs against the
parquet files they replaced (tests/fixtures/processed_fingerprint.json).

For every column it records a kind (datetime, date, bool, int, float, string) and a SHA-256
of the values in a canonical text form. Missing values and empty strings both canonicalize
to "", since a CSV stores both as an empty cell.
"""
from __future__ import annotations

import hashlib
import math

import pandas as pd

from csv_io import DATE_COLUMNS


def column_kind(name: str, s: pd.Series) -> str:
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return "datetime"
    if name in DATE_COLUMNS:
        return "date"
    if pd.api.types.is_bool_dtype(s.dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(s.dtype):
        return "int"
    if pd.api.types.is_float_dtype(s.dtype):
        return "float"
    return "string"


def canonical(value, kind: str) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if kind == "datetime":
        return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if kind == "date":
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if kind == "bool":
        return "1" if bool(value) else "0"
    if kind == "int":
        return str(int(value))
    if kind == "float":
        return repr(float(value))
    return str(value)


def fingerprint(df: pd.DataFrame) -> dict:
    cols = {}
    for name in df.columns:
        kind = column_kind(name, df[name])
        text = "\x1f".join(canonical(v, kind) for v in df[name].tolist())
        cols[name] = {"kind": kind, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return {"rows": len(df), "columns": list(df.columns), "column_fingerprints": cols}

"""Stage 10 tests in one common layout (ANALYSIS_PLAN.md section 10.1 and 10.7).

Each of notebooks 06 to 09 saves its tests as results/stage10_tests_<part>.csv with the
columns below. Notebook 10 reads those files, adds the Holm adjusted p values across the
primary tests and writes results/stage10_all_tests.csv.

One row per test:
  part, test_id        plan section and a unique name
  role                 "primary" (in the Holm family) or "secondary"
  description, null_hypothesis
  n, unit              sample size and what it counts (games, events, trades)
  estimate             the estimated quantity (empty for a joint test)
  ci_lo, ci_hi         the 95% interval that goes with the p value
  interval_type        "clustered Wald" or "bootstrap percentile" (empty for a joint test)
  boot_lo, boot_hi     the bootstrap percentile interval, as a robustness check
  statistic, statistic_name, df    the test statistic (z, chi2 or LR), when there is one
  p_raw, p_method      the unadjusted two-sided p value and how it was computed
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stats_tools import Z, bootstrap_p_value, holm

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
ALL_TESTS_PATH = RESULTS_DIR / "stage10_all_tests.csv"

# The Holm family (plan 10.1, rule 2).
PRIMARY = (
    "line_move_24h_d",
    "subgroup_liquidity_slope",
    "subgroup_season_half_slope",
    "subgroup_rest_slope",
    "injury_event_mean",
    "replication_lr",
    "replication_popularity_c",
    "replication_opposite_return",
)
PARTS = ("line_movement", "information_arrival", "subgroups", "replication")
COLUMNS = ["part", "test_id", "role", "description", "null_hypothesis", "n", "unit",
           "estimate", "ci_lo", "ci_hi", "interval_type", "boot_lo", "boot_hi",
           "statistic", "statistic_name", "df", "p_raw", "p_method"]


def part_path(part: str) -> Path:
    return RESULTS_DIR / f"stage10_tests_{part}.csv"


def _row(part: str, test_id: str, description: str, null_hypothesis: str, n: int,
         unit: str, **values) -> dict:
    row = dict.fromkeys(COLUMNS, np.nan)
    row.update({"part": part, "test_id": test_id,
                "role": "primary" if test_id in PRIMARY else "secondary",
                "description": description, "null_hypothesis": null_hypothesis,
                "n": int(n), "unit": unit, "interval_type": "", "statistic_name": ""})
    row.update(values)
    return row


def wald_row(part: str, test_id: str, description: str, null_hypothesis: str, n: int,
             estimate: float, se: float, p_value: float, null_value: float = 0.0,
             boot: tuple[float, float] | None = None, unit: str = "games") -> dict:
    """A coefficient tested with a clustered Wald z test."""
    boot_lo, boot_hi = boot if boot is not None else (np.nan, np.nan)
    return _row(part, test_id, description, null_hypothesis, n, unit,
                estimate=estimate, ci_lo=estimate - Z * se, ci_hi=estimate + Z * se,
                interval_type="clustered Wald", boot_lo=boot_lo, boot_hi=boot_hi,
                statistic=(estimate - null_value) / se, statistic_name="z",
                p_raw=p_value, p_method="clustered Wald z")


def joint_row(part: str, test_id: str, description: str, null_hypothesis: str, n: int,
              statistic: float, df: int, p_value: float, statistic_name: str,
              p_method: str, unit: str = "games") -> dict:
    """A joint test: a statistic and its degrees of freedom, no single estimate."""
    return _row(part, test_id, description, null_hypothesis, n, unit,
                statistic=statistic, statistic_name=statistic_name, df=int(df),
                p_raw=p_value, p_method=p_method)


def bootstrap_row(part: str, test_id: str, description: str, null_hypothesis: str, n: int,
                  estimate: float, replicates, unit: str, null_value: float = 0.0) -> dict:
    """A statistic tested with the block bootstrap: percentile interval and bootstrap p."""
    x = pd.Series(replicates, dtype=float).dropna()
    lo, hi = float(x.quantile(0.025)), float(x.quantile(0.975))
    return _row(part, test_id, description, null_hypothesis, n, unit,
                estimate=estimate, ci_lo=lo, ci_hi=hi, interval_type="bootstrap percentile",
                boot_lo=lo, boot_hi=hi, p_raw=bootstrap_p_value(x, null_value),
                p_method="block bootstrap")


def table(rows: list[dict]) -> pd.DataFrame:
    """The rows of one part as a table in the common layout."""
    out = pd.DataFrame(rows, columns=COLUMNS)
    if out["test_id"].duplicated().any():
        raise ValueError("duplicate test_id")
    if out["p_raw"].isna().any():
        raise ValueError("every test needs a raw p value")
    out["df"] = out["df"].astype("Int64")
    return out


def bonferroni_bound(p_raw, family_size: int = len(PRIMARY)):
    """min(1, m * p): never below the Holm adjusted value. For reports made before all
    primary tests have run."""
    bound = np.minimum(1.0, family_size * np.asarray(p_raw, dtype=float))
    return pd.Series(bound, index=p_raw.index) if isinstance(p_raw, pd.Series) else bound


def combine(frames: list[pd.DataFrame], family: tuple[str, ...] = PRIMARY) -> pd.DataFrame:
    """All parts in one table, with Holm adjusted p values across the primary tests."""
    out = pd.concat(frames, ignore_index=True)
    if out["test_id"].duplicated().any():
        raise ValueError("duplicate test_id")
    is_primary = out["role"] == "primary"
    found = set(out.loc[is_primary, "test_id"])
    if found != set(family):
        raise ValueError(f"primary tests do not match the family: missing "
                         f"{sorted(set(family) - found)}, unexpected "
                         f"{sorted(found - set(family))}")
    out["p_holm"] = np.nan
    out.loc[is_primary, "p_holm"] = holm(out.loc[is_primary, "p_raw"])
    out["holm_family_size"] = pd.array([len(family) if p else pd.NA for p in is_primary],
                                       dtype="Int64")
    out["significant_holm"] = pd.array(
        [int(v < 0.05) if p else pd.NA for p, v in zip(is_primary, out["p_holm"])],
        dtype="Int64")
    return out

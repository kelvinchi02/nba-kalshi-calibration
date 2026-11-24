"""Metro area population per NBA team, for the Stage 9 popularity measure (plan 9.1).

Sources, both official estimates for July 1, 2025:
  * US teams: US Census Bureau, Metropolitan Statistical Area population totals,
    Vintage 2025, file cbsa-est2025-alldata.csv.
  * Toronto: Statistics Canada, Table 17-10-0148-01, population estimates by census
    metropolitan area (2021 boundaries).

Each team gets the metropolitan area that contains its home arena. Teams in the same area
get the same value (LAL and LAC, NYK and BKN).

The two source files are downloaded once to data/raw/population/ (gitignored), with the
download date in retrieved.txt. --refresh downloads them again.

Output:
  data/processed/team_metro_population.csv   one row per team

Usage:
    python src/metro_population.py [--refresh]
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from csv_io import read_csv, write_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "population"
OUT_PATH = REPO_ROOT / "data" / "processed" / "team_metro_population.csv"
RETRIEVED_PATH = RAW_DIR / "retrieved.txt"

YEAR = 2025
REFERENCE_DATE = f"{YEAR}-07-01"

CENSUS_FILE = "cbsa-est2025-alldata.csv"
CENSUS_FILE_URL = ("https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/"
                   "metro/totals/" + CENSUS_FILE)
CENSUS_PAGE_URL = ("https://www.census.gov/data/tables/time-series/demo/popest/"
                   "2020s-total-metro-and-micro-statistical-areas.html")
CENSUS_SOURCE = ("US Census Bureau, Metropolitan Statistical Area population totals, "
                 "Vintage 2025")

STATCAN_FILE = "17100148-eng.zip"
STATCAN_FILE_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/" + STATCAN_FILE
STATCAN_PAGE_URL = "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710014801"
STATCAN_SOURCE = ("Statistics Canada, Table 17-10-0148-01, population estimates, July 1, "
                  "by census metropolitan area")
TORONTO_DGUID = "2021S0503535"

# Team -> (CBSA code, the first principal city in the CBSA title). The city is checked
# against the title in the Census file, so a wrong or outdated code cannot pass silently.
TEAM_CBSA = {
    "ATL": ("12060", "Atlanta"),
    "BKN": ("35620", "New York"),
    "BOS": ("14460", "Boston"),
    "CHA": ("16740", "Charlotte"),
    "CHI": ("16980", "Chicago"),
    "CLE": ("17410", "Cleveland"),
    "DAL": ("19100", "Dallas"),
    "DEN": ("19740", "Denver"),
    "DET": ("19820", "Detroit"),
    "GSW": ("41860", "San Francisco"),
    "HOU": ("26420", "Houston"),
    "IND": ("26900", "Indianapolis"),
    "LAC": ("31080", "Los Angeles"),
    "LAL": ("31080", "Los Angeles"),
    "MEM": ("32820", "Memphis"),
    "MIA": ("33100", "Miami"),
    "MIL": ("33340", "Milwaukee"),
    "MIN": ("33460", "Minneapolis"),
    "NOP": ("35380", "New Orleans"),
    "NYK": ("35620", "New York"),
    "OKC": ("36420", "Oklahoma City"),
    "ORL": ("36740", "Orlando"),
    "PHI": ("37980", "Philadelphia"),
    "PHX": ("38060", "Phoenix"),
    "POR": ("38900", "Portland"),
    "SAC": ("40900", "Sacramento"),
    "SAS": ("41700", "San Antonio"),
    "UTA": ("41620", "Salt Lake City"),
    "WAS": ("47900", "Washington"),
}
CANADA_TEAMS = {"TOR": TORONTO_DGUID}


# ---------------------------------------------------------------- download

def download(refresh: bool) -> str:
    """Fetch the two source files if needed. Returns the date they were retrieved."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {CENSUS_FILE: CENSUS_FILE_URL, STATCAN_FILE: STATCAN_FILE_URL}
    have_all = all((RAW_DIR / name).exists() for name in files) and RETRIEVED_PATH.exists()
    if have_all and not refresh:
        return RETRIEVED_PATH.read_text(encoding="utf-8").strip()
    for name, url in files.items():
        resp = requests.get(url, timeout=300, headers={"User-Agent": "python-requests"})
        resp.raise_for_status()
        (RAW_DIR / name).write_bytes(resp.content)
    today = date.today().isoformat()
    RETRIEVED_PATH.write_text(today + "\n", encoding="utf-8")
    return today


# ---------------------------------------------------------------- parsing

def census_metro_areas(path: Path, year: int = YEAR) -> pd.DataFrame:
    """CBSA code, title and July 1 population of every Metropolitan Statistical Area."""
    raw = pd.read_csv(path, dtype=str, encoding="latin-1")
    msa = raw[raw["LSAD"] == "Metropolitan Statistical Area"]
    return pd.DataFrame({"area_code": msa["CBSA"].to_numpy(),
                         "metro_area": msa["NAME"].to_numpy(),
                         "population": msa[f"POPESTIMATE{year}"].astype(int).to_numpy()})


def statcan_population(zip_path: Path, dguid: str, year: int = YEAR) -> tuple[str, int]:
    """(area name, July 1 population) of one area: total gender, all ages."""
    with zipfile.ZipFile(zip_path) as archive:
        member = STATCAN_FILE.replace("-eng.zip", ".csv")
        with archive.open(member) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
            return statcan_row(reader, dguid, year)


def statcan_row(rows, dguid: str, year: int) -> tuple[str, int]:
    matches = [r for r in rows
               if r["DGUID"] == dguid and r["REF_DATE"] == str(year)
               and r["Gender"] == "Total - gender" and r["Age group"] == "All ages"]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {dguid} in {year}, found {len(matches)}")
    return matches[0]["GEO"], int(matches[0]["VALUE"])


def build_table(census: pd.DataFrame, canada: dict[str, tuple[str, int]],
                retrieved: str) -> pd.DataFrame:
    """One row per team. `canada` maps a team to its (area name, population)."""
    by_code = census.set_index("area_code")
    rows = []
    for team, (code, city) in TEAM_CBSA.items():
        if code not in by_code.index:
            raise ValueError(f"{team}: CBSA {code} is not a metropolitan area in the file")
        name = by_code.loc[code, "metro_area"]
        if not name.startswith(city):
            raise ValueError(f"{team}: CBSA {code} is '{name}', expected '{city}...'")
        rows.append({"team": team, "metro_area": name, "area_type": "MSA", "area_code": code,
                     "population": int(by_code.loc[code, "population"]),
                     "reference_date": REFERENCE_DATE, "source": CENSUS_SOURCE,
                     "source_url": CENSUS_PAGE_URL, "retrieved": retrieved})
    for team, dguid in CANADA_TEAMS.items():
        name, population = canada[team]
        rows.append({"team": team, "metro_area": name, "area_type": "CMA", "area_code": dguid,
                     "population": population, "reference_date": REFERENCE_DATE,
                     "source": STATCAN_SOURCE, "source_url": STATCAN_PAGE_URL,
                     "retrieved": retrieved})
    return pd.DataFrame(rows).sort_values("team", ignore_index=True)


def load_metro_population() -> pd.DataFrame:
    """The saved table: one row per team."""
    return read_csv(OUT_PATH)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="download the sources again")
    args = parser.parse_args()
    retrieved = download(args.refresh)
    census = census_metro_areas(RAW_DIR / CENSUS_FILE)
    canada = {team: statcan_population(RAW_DIR / STATCAN_FILE, dguid)
              for team, dguid in CANADA_TEAMS.items()}
    table = build_table(census, canada, retrieved)
    write_csv(table, OUT_PATH)
    print(f"Wrote {len(table)} teams to {OUT_PATH.relative_to(REPO_ROOT)} "
          f"(sources retrieved {retrieved})")
    print(table[["team", "metro_area", "population"]].to_string(index=False))


if __name__ == "__main__":
    main()

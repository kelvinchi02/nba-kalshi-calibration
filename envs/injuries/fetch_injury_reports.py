"""Download and parse the official NBA injury reports of a season (plan 10.4, part D2).

This script runs in its own Python environment (envs/injuries), because the `nbainjuries`
package needs pandas below 3.0 and the main project uses pandas 3. It only writes files.
The analysis reads them from the main environment.

Source: the report files the NBA publishes at
https://ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf, one per
snapshot (hourly until December 19, 2025, every 15 minutes from December 22, 2025).
`nbainjuries` supplies the file names and the PDF table parser. No other source is used.

Three steps, each resumable:

  download  every snapshot file from --start to --end (US Eastern dates). For each file the
            HTTP status and the server's Last-Modified header are recorded. Last-Modified
            is the time the file was posted.
            The server answers 403 in two different cases. A 403 from the file store
            (header "Server: AmazonS3") means the snapshot does not exist. A 403 from the
            delivery network in front of it (any other Server) is a temporary block after
            many requests: the script waits and asks again, and never records such a file
            as missing.
  parse     every downloaded file, with nbainjuries. The report time printed in the file's
            header is read too.
  combine   write the two outputs below.

Outputs:
  data/processed/injury_report_snapshots_2025_26.csv
      one row per snapshot file: its name, the scheduled report time (US Eastern), whether
      it exists on the server, when it was posted, and whether it was parsed
  data/raw/injury_reports/injury_report_rows_2025_26.csv.gz   (not in git)
      every row of every parsed snapshot: game date, game time, matchup, team, player,
      status and reason

Usage (from the repo root):
    uv run --project envs/injuries python envs/injuries/fetch_injury_reports.py
    uv run --project envs/injuries python envs/injuries/fetch_injury_reports.py --step parse
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "injury_reports"
PDF_DIR = RAW_DIR / "pdf"
PARSED_DIR = RAW_DIR / "parsed"
DOWNLOAD_LOG = RAW_DIR / "download_log.jsonl"
PARSE_LOG = RAW_DIR / "parse_log.jsonl"
ROWS_PATH = RAW_DIR / "injury_report_rows_2025_26.csv.gz"
SNAPSHOTS_PATH = REPO_ROOT / "data" / "processed" / "injury_report_snapshots_2025_26.csv"

SEASON_START, SEASON_END = date(2025, 10, 20), date(2026, 4, 12)
REPORT_COLUMNS = ["Game Date", "Game Time", "Matchup", "Team", "Player Name",
                  "Current Status", "Reason"]
HEADER_TIME = re.compile(r"Injury\s+Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s*([AP]M)")
NAME_TIME = re.compile(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(?:_(\d{2}))?([AP]M)\.pdf")
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like "
              "Gecko) Chrome/144.0.7559.132 Safari/537.36")


# ---------------------------------------------------------------- snapshot files

def snapshot_files(start: date, end: date) -> list[dict]:
    """Every snapshot file name from start to end: name, url and the time used to ask
    nbainjuries for it. Hourly names appear once, 15-minute names once per quarter hour."""
    from nbainjuries import injury

    seen, out = set(), []
    t = datetime(start.year, start.month, start.day)
    stop = datetime(end.year, end.month, end.day) + timedelta(days=1)
    while t < stop:
        try:
            url = injury.gen_url(t)
        except ValueError:      # the hours on December 19, 2025 between the two formats
            url = None
        if url and url not in seen:
            seen.add(url)
            out.append({"report_file": url.rsplit("/", 1)[1], "url": url,
                        "lookup_time": t.isoformat()})
        t += timedelta(minutes=15)
    return out


def scheduled_time_from_name(report_file: str) -> datetime:
    """The scheduled report time from the file name (US Eastern, no time zone attached).

    A 15-minute name carries the minute. An hourly name does not: hourly reports were
    stamped at half past the hour.
    """
    m = NAME_TIME.fullmatch(report_file)
    if not m:
        raise ValueError(f"unexpected report file name {report_file}")
    day, hour, minute, half = m.groups()
    text = f"{day} {hour}:{minute or '30'}{half}"
    return datetime.strptime(text, "%Y-%m-%d %I:%M%p")


# ---------------------------------------------------------------- logs (resumable)

def read_log(path: Path) -> dict[str, dict]:
    """The last record per report file."""
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records[rec["report_file"]] = rec
    return records


def append_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------- download

FILE_STORE = "AmazonS3"          # the Server header of the store that holds the files
REQUEST_PAUSE_S = 0.25           # pause after every request, per worker
BLOCK_WAITS_S = (20, 40, 80, 160, 320)


def download_one(item: dict, session: requests.Session) -> dict:
    """Fetch one file.

    200: saved. 403 or 404 from the file store: the snapshot does not exist. 403 from the
    delivery network: a temporary block, so wait and ask again. Anything still unresolved
    is recorded as an error, and a later run tries it again.
    """
    rec = {"report_file": item["report_file"], "url": item["url"],
           "lookup_time": item["lookup_time"], "http_status": None, "server": None,
           "last_modified": None, "n_bytes": None, "outcome": "error", "error": None}
    for wait in BLOCK_WAITS_S:
        try:
            resp = session.get(item["url"], headers={"User-Agent": USER_AGENT}, timeout=60)
        except requests.RequestException as exc:
            rec["error"] = repr(exc)[:200]
            time.sleep(wait)
            continue
        time.sleep(REQUEST_PAUSE_S)
        server = resp.headers.get("Server", "")
        rec.update(http_status=resp.status_code, server=server)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            (PDF_DIR / item["report_file"]).write_bytes(resp.content)
            rec.update(outcome="saved", n_bytes=len(resp.content), error=None,
                       last_modified=resp.headers.get("Last-Modified"))
            return rec
        if resp.status_code in (403, 404) and server == FILE_STORE:
            rec.update(outcome="not_on_server", error=None)
            return rec
        rec["error"] = f"HTTP {resp.status_code} from {server or 'unknown server'}"
        time.sleep(wait)
    return rec


def download(files: list[dict], workers: int) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    # A "not on server" record counts only when the file store itself said so.
    done = {k for k, v in read_log(DOWNLOAD_LOG).items()
            if (v["outcome"] == "not_on_server" and v.get("server") == FILE_STORE)
            or (v["outcome"] == "saved" and (PDF_DIR / k).exists())}
    todo = [f for f in files if f["report_file"] not in done]
    print(f"download: {len(files)} snapshot files, {len(files) - len(todo)} done, "
          f"{len(todo)} to fetch", flush=True)
    session = requests.Session()
    counts = {"saved": 0, "not_on_server": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, f, session) for f in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            append_log(DOWNLOAD_LOG, rec)
            counts[rec["outcome"]] += 1
            if i % 500 == 0 or i == len(todo):
                print(f"download: {i}/{len(todo)} {counts}", flush=True)


# ---------------------------------------------------------------- parse

def header_time(pdf_path: Path) -> str | None:
    """The report time printed at the top of the first page, as ISO text (US Eastern)."""
    import PyPDF2      # only in the injuries environment, like nbainjuries

    with pdf_path.open("rb") as f:
        text = PyPDF2.PdfReader(f).pages[0].extract_text() or ""
    m = HEADER_TIME.search(text)
    if not m:
        return None
    return datetime.strptime(" ".join(m.groups()), "%m/%d/%y %I:%M %p").isoformat()


def parse_one(report_file: str, lookup_time: str) -> dict:
    """Parse one downloaded file with nbainjuries and save its rows."""
    from nbainjuries import injury

    rec = {"report_file": report_file, "parsed": False, "n_rows": None, "header_time": None,
           "error": None}
    try:
        rec["header_time"] = header_time(PDF_DIR / report_file)
        df = injury.get_reportdata(datetime.fromisoformat(lookup_time), local=True,
                                   localdir=str(PDF_DIR), return_df=True)
        df = df[REPORT_COLUMNS]
        df.to_csv(PARSED_DIR / (report_file[:-4] + ".csv"), index=False, lineterminator="\n")
        rec.update(parsed=True, n_rows=len(df))
    except Exception as exc:  # noqa: BLE001 - a file that cannot be parsed is recorded
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec


def parse(workers: int) -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    saved = {k: v for k, v in read_log(DOWNLOAD_LOG).items() if v["outcome"] == "saved"}
    done = read_log(PARSE_LOG)
    todo = [k for k in sorted(saved) if k not in done]
    print(f"parse: {len(saved)} downloaded files, {len(saved) - len(todo)} done, "
          f"{len(todo)} to parse", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(parse_one, k, saved[k]["lookup_time"]) for k in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            append_log(PARSE_LOG, rec)
            failed += not rec["parsed"]
            if i % 250 == 0 or i == len(todo):
                print(f"parse: {i}/{len(todo)}, failed {failed}", flush=True)


# ---------------------------------------------------------------- combine

def posted_utc(last_modified: str | None) -> str:
    """The Last-Modified header as an ISO 8601 UTC time, or empty."""
    if not last_modified:
        return ""
    return parsedate_to_datetime(last_modified).strftime("%Y-%m-%dT%H:%M:%SZ")


def combine(files: list[dict]) -> None:
    downloads, parses = read_log(DOWNLOAD_LOG), read_log(PARSE_LOG)
    snapshots = []
    with gzip.GzipFile(filename="", mode="wb", fileobj=ROWS_PATH.open("wb"), mtime=0) as gz:
        out = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["report_file", "game_date", "game_time", "matchup", "team",
                         "player_name", "status", "reason"])
        for item in sorted(files, key=lambda f: scheduled_time_from_name(f["report_file"])):
            name = item["report_file"]
            d, p = downloads.get(name, {}), parses.get(name, {})
            scheduled = scheduled_time_from_name(name)
            snapshots.append({
                "report_file": name,
                "scheduled_time_et": scheduled.strftime("%Y-%m-%dT%H:%M:%S"),
                "header_time_et": p.get("header_time") or "",
                "on_server": {"saved": 1, "not_on_server": 0}.get(d.get("outcome"), ""),
                "http_status": d.get("http_status") or "",
                "posted_utc": posted_utc(d.get("last_modified")),
                "n_bytes": d.get("n_bytes") or "",
                "parsed": int(bool(p.get("parsed"))) if d.get("outcome") == "saved" else "",
                "n_rows": p.get("n_rows") if p.get("n_rows") is not None else "",
                "error": p.get("error") or d.get("error") or "",
            })
            if p.get("parsed"):
                rows = pd.read_csv(PARSED_DIR / (name[:-4] + ".csv"), dtype=str,
                                   keep_default_na=False)
                for r in rows.itertuples(index=False):
                    writer.writerow([name, *r])
        out.flush()
        out.detach()
    pd.DataFrame(snapshots).to_csv(SNAPSHOTS_PATH, index=False, lineterminator="\n")
    table = pd.DataFrame(snapshots)
    print(f"combine: {len(table)} snapshot files, on server {(table['on_server'] == 1).sum()}, "
          f"parsed {(table['parsed'] == 1).sum()}, "
          f"download errors {(table['on_server'] == '').sum()}")
    print(f"Saved {SNAPSHOTS_PATH.relative_to(REPO_ROOT)} and {ROWS_PATH.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["download", "parse", "combine", "all"],
                        default="all")
    parser.add_argument("--start", type=date.fromisoformat, default=SEASON_START)
    parser.add_argument("--end", type=date.fromisoformat, default=SEASON_END)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--parse-workers", type=int, default=6)
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = snapshot_files(args.start, args.end)
    if args.step in ("download", "all"):
        download(files, args.download_workers)
    if args.step in ("parse", "all"):
        parse(args.parse_workers)
    if args.step in ("combine", "all"):
        combine(files)


if __name__ == "__main__":
    main()

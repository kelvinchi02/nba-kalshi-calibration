"""Check sampled injury events against the original report files (plan 10.4, reliability rule).

The events come from the table parser (tabula, through nbainjuries). This check reads the
same PDF files a second way: the plain text of every page, with PyPDF2. For each sampled
event it looks up the player in the snapshot of the event and in the snapshot before it,
and compares the game, the status and the report time with what the event says.

Input:  a CSV with one row per player of each sampled event (written by
        src/injury_events.py --sample): event_id, report_file, prev_report_file, game_date,
        matchup, team_name, player_name, prev_status, scheduled_time_et
Output: the same rows with what the text of the two files says, and a verdict per row.

Usage (from the repo root):
    uv run --project envs/injuries python envs/injuries/check_events.py <in.csv> <out.csv>
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import PyPDF2

REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = REPO_ROOT / "data" / "raw" / "injury_reports" / "pdf"
STATUSES = {"Out", "Questionable", "Doubtful", "Probable", "Available"}
MATCHUP = re.compile(r"^[A-Z]{3}@[A-Z]{3}$")
GAME_DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
HEADER_TIME = re.compile(r"Injury\s+Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s*([AP]M)")


def page_text(report_file: str) -> str:
    with (PDF_DIR / report_file).open("rb") as f:
        return "\n".join(page.extract_text() or "" for page in PyPDF2.PdfReader(f).pages)


def header_time(text: str) -> str:
    m = HEADER_TIME.search(text)
    return datetime.strptime(" ".join(m.groups()), "%m/%d/%y %I:%M %p").isoformat() if m else ""


def player_statuses(text: str, player_name: str) -> list[tuple[str, str, str]]:
    """Every (game date, matchup, status) under which the player appears in the text."""
    tokens = text.split()
    name = player_name.split()
    found, game_date, matchup = [], "", ""
    for i, tok in enumerate(tokens):
        if GAME_DATE.match(tok):
            game_date = tok
        elif MATCHUP.match(tok):
            matchup = tok
        if tokens[i:i + len(name)] == name:
            status = tokens[i + len(name)] if i + len(name) < len(tokens) else ""
            found.append((game_date, matchup, status if status in STATUSES else "?"))
    return found


def status_for_game(found: list, game_date: str, matchup: str) -> str:
    """The player's status for one game: a status, "" when he is not listed for it."""
    hits = [s for d, m, s in found if d == game_date and m == matchup]
    return hits[0] if hits else ""


def main() -> None:
    rows = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
    texts: dict[str, str] = {}
    out = []
    for r in rows.itertuples(index=False):
        for name in (r.report_file, r.prev_report_file):
            if name not in texts:
                texts[name] = page_text(name)
        now = status_for_game(player_statuses(texts[r.report_file], r.player_name),
                              r.game_date, r.matchup)
        before = status_for_game(player_statuses(texts[r.prev_report_file], r.player_name),
                                 r.game_date, r.matchup)
        time_in_file = header_time(texts[r.report_file])
        ok = (now == "Out" and before == r.prev_status
              and (time_in_file == "" or time_in_file == r.scheduled_time_et))
        out.append({**r._asdict(), "text_status": now, "text_prev_status": before,
                    "text_report_time": time_in_file, "agrees": int(ok)})
    table = pd.DataFrame(out)
    table.to_csv(sys.argv[2], index=False, encoding="utf-8", lineterminator="\n")
    print(f"{len(table)} player rows in {table['event_id'].nunique()} events: "
          f"{int(table['agrees'].sum())} agree with the text of the report files")
    print(table.loc[table["agrees"] == 0].to_string(index=False))


if __name__ == "__main__":
    main()

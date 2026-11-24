"""The parts of the injury report downloader that need neither the network nor nbainjuries."""
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "envs" / "injuries" / "fetch_injury_reports.py"
spec = importlib.util.spec_from_file_location("fetch_injury_reports", PATH)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def test_scheduled_time_from_file_name():
    # Hourly names carry no minute: those reports were stamped at half past the hour.
    assert fetch.scheduled_time_from_name("Injury-Report_2025-11-03_05PM.pdf") == datetime(
        2025, 11, 3, 17, 30)
    assert fetch.scheduled_time_from_name("Injury-Report_2025-11-03_12AM.pdf") == datetime(
        2025, 11, 3, 0, 30)
    # 15-minute names carry it.
    assert fetch.scheduled_time_from_name("Injury-Report_2026-01-14_05_15PM.pdf") == datetime(
        2026, 1, 14, 17, 15)
    assert fetch.scheduled_time_from_name("Injury-Report_2026-01-14_12_00PM.pdf") == datetime(
        2026, 1, 14, 12, 0)
    with pytest.raises(ValueError):
        fetch.scheduled_time_from_name("something_else.pdf")


def test_posted_time_is_the_last_modified_header_in_utc():
    assert fetch.posted_utc("Wed, 14 Jan 2026 05:00:10 GMT") == "2026-01-14T05:00:10Z"
    assert fetch.posted_utc(None) == "" and fetch.posted_utc("") == ""


class Response:
    def __init__(self, status, server, content=b"", last_modified=None):
        self.status_code, self.content = status, content
        self.headers = {"Server": server}
        if last_modified:
            self.headers["Last-Modified"] = last_modified


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


ITEM = {"report_file": "Injury-Report_2026-01-14_05_15PM.pdf", "url": "https://x/y.pdf",
        "lookup_time": "2026-01-14T17:15:00"}


@pytest.fixture
def quick(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch, "PDF_DIR", tmp_path)
    return tmp_path


def test_a_file_is_saved_with_its_posting_time(quick):
    s = Session([Response(200, "AmazonS3", b"%PDF-1.7 ...", "Wed, 14 Jan 2026 22:15:06 GMT")])
    rec = fetch.download_one(ITEM, s)
    assert rec["outcome"] == "saved" and rec["last_modified"] == "Wed, 14 Jan 2026 22:15:06 GMT"
    assert (quick / ITEM["report_file"]).read_bytes().startswith(b"%PDF")


def test_403_from_the_file_store_means_the_snapshot_does_not_exist(quick):
    s = Session([Response(403, "AmazonS3", b"<Error><Code>AccessDenied</Code></Error>")])
    rec = fetch.download_one(ITEM, s)
    assert rec["outcome"] == "not_on_server" and rec["server"] == "AmazonS3" and s.calls == 1


def test_403_from_the_delivery_network_is_a_block_and_is_asked_again(quick):
    block = Response(403, "AkamaiGHost", b"<HTML>Access Denied</HTML>")
    s = Session([block, block, Response(200, "AmazonS3", b"%PDF-1.7", "x")])
    rec = fetch.download_one(ITEM, s)
    assert rec["outcome"] == "saved" and s.calls == 3


def test_a_block_that_does_not_lift_is_an_error_never_a_missing_file(quick):
    s = Session([Response(403, "AkamaiGHost")] * len(fetch.BLOCK_WAITS_S))
    rec = fetch.download_one(ITEM, s)
    assert rec["outcome"] == "error" and "AkamaiGHost" in rec["error"]
    assert not (quick / ITEM["report_file"]).exists()


def test_a_200_that_is_not_a_pdf_is_not_saved(quick):
    s = Session([Response(200, "AmazonS3", b"<html>oops</html>")] * len(fetch.BLOCK_WAITS_S))
    assert fetch.download_one(ITEM, s)["outcome"] == "error"

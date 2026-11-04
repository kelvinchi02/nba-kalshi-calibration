"""Small Kalshi REST client: rate limiting, retries with backoff, cursor pagination,
optional request signing.

All endpoints this project uses are public. Signing is only switched on when BOTH
KALSHI_API_KEY_ID and a readable KALSHI_PRIVATE_KEY_PATH are set in .env, and secret
values are never logged.

Example:
    client = KalshiClient()
    for market in client.paginate("/historical/markets", "markets", {"series_ticker": "KXNBAGAME"}):
        ...
"""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from tenacity import (before_sleep_log, retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

log = logging.getLogger("kalshi_client")


class RetryableHTTPError(Exception):
    """429 or 5xx: worth retrying."""


class KalshiClient:
    def __init__(self, base_url: str | None = None, max_requests_per_sec: float = 10.0,
                 timeout: float = 30.0):
        load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
        self.base_url = (base_url or os.getenv("KALSHI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.base_path = urlparse(self.base_url).path  # e.g. /trade-api/v2, needed for signing
        self.timeout = timeout
        self.min_interval = 1.0 / max_requests_per_sec  # public basic tier allows 20 reads/s
        self._last_request = 0.0
        self._lock = threading.Lock()
        self.session = requests.Session()
        self._key_id, self._private_key = self._load_signing_key()

    # ------------------------------------------------------------ signing (optional)

    def _load_signing_key(self):
        key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not key_id or not key_path:
            return None, None
        path = Path(key_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            return None, None
        from cryptography.hazmat.primitives import serialization
        private_key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        log.info("Request signing enabled (key file found)")
        return key_id, private_key

    @property
    def signing(self) -> bool:
        return self._private_key is not None

    def _auth_headers(self, method: str, path: str) -> dict:
        """Kalshi signature: RSA-PSS/SHA256 over timestamp_ms + METHOD + path (no query)."""
        if not self.signing:
            return {}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts_ms = str(int(time.time() * 1000))
        message = (ts_ms + method.upper() + self.base_path + path).encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    # ------------------------------------------------------------ requests

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    @retry(retry=retry_if_exception_type((RetryableHTTPError, requests.ConnectionError,
                                          requests.Timeout)),
           stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=60),
           before_sleep=before_sleep_log(log, logging.WARNING), reraise=True)
    def get(self, path: str, params: dict | None = None) -> dict:
        """GET a path such as '/historical/markets'. Retries on 429, 5xx and network errors."""
        self._throttle()
        resp = self.session.get(self.base_url + path, params=params, timeout=self.timeout,
                                headers=self._auth_headers("GET", path))
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableHTTPError(f"{resp.status_code} on {path}")
        if not resp.ok:
            # Client errors are not retried; include Kalshi's message to make them debuggable.
            raise requests.HTTPError(f"{resp.status_code} on {path} {params}: {resp.text[:300]}",
                                     response=resp)
        return resp.json()

    def paginate(self, path: str, key: str, params: dict | None = None) -> Iterator[dict]:
        """Yield every item under `key`, following the cursor until it is empty."""
        params = dict(params or {})
        while True:
            data = self.get(path, params)
            yield from data.get(key) or []
            cursor = data.get("cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    # ------------------------------------------------------------ convenience

    def historical_cutoff(self) -> dict:
        return self.get("/historical/cutoff")

    def historical_markets(self, series_ticker: str) -> list[dict]:
        return list(self.paginate("/historical/markets", "markets",
                                  {"series_ticker": series_ticker, "limit": 1000}))

    def historical_trades(self, ticker: str, min_ts: int, max_ts: int) -> list[dict]:
        """Trades with min_ts <= created_time <= max_ts (unix seconds), newest first."""
        return list(self.paginate("/historical/trades", "trades",
                                  {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts,
                                   "limit": 1000}))

    def historical_candlesticks(self, ticker: str, start_ts: int, end_ts: int,
                                period_interval: int = 1) -> list[dict]:
        data = self.get(f"/historical/markets/{ticker}/candlesticks",
                        {"start_ts": start_ts, "end_ts": end_ts,
                         "period_interval": period_interval})
        return data.get("candlesticks") or []

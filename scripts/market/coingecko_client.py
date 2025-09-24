# scripts/market/coingecko_client.py
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


_DEFAULT_DEMO_BASE = "https://api.coingecko.com/api/v3"
_DEFAULT_PRO_BASE = "https://pro-api.coingecko.com/api/v3"

# Env knobs
ENV_BASE = os.getenv("MW_CG_BASE_URL", _DEFAULT_DEMO_BASE).rstrip("/")
ENV_API_KEY = os.getenv("MW_CG_API_KEY", "").strip() or None
ENV_RATE_PER_MIN = int(os.getenv("MW_CG_RATE_LIMIT_PER_MIN", "25"))
ENV_TIMEOUT_CONNECT = float(os.getenv("MW_CG_CONNECT_TIMEOUT_S", "5"))
ENV_TIMEOUT_READ = float(os.getenv("MW_CG_READ_TIMEOUT_S", "10"))

# Simple pacing to leave CI headroom
PACE_SLEEP = 60.0 / max(1, ENV_RATE_PER_MIN)


class _RetryableHTTPError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"Retryable status {status}: {body[:200]}")
        self.status = status
        self.body = body


@dataclass
class CoinGeckoClient:
    """
    Minimal CoinGecko client with:
      - base URL switching (demo/pro)
      - optional x-cg-pro-api-key header
      - retries on 429/5xx with exponential backoff + jitter
      - simple per-run in-memory cache
      - naive pacing based on MW_CG_RATE_LIMIT_PER_MIN
    """

    base_url: str = field(default_factory=lambda: ENV_BASE)
    api_key: Optional[str] = field(default_factory=lambda: ENV_API_KEY)
    pace_sleep: float = field(default_factory=lambda: PACE_SLEEP)
    timeout: Tuple[float, float] = field(
        default_factory=lambda: (ENV_TIMEOUT_CONNECT, ENV_TIMEOUT_READ)
    )  # (connect, read)
    max_retries: int = 4

    # internal
    session: requests.Session = field(default_factory=requests.Session, init=False)
    _cache: Dict[str, Tuple[float, Any]] = field(default_factory=dict, init=False)

    # ---------- helpers ----------

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            # CoinGecko Pro header
            h["x-cg-pro-api-key"] = self.api_key
        return h

    def _cache_get(self, key: str) -> Optional[Any]:
        ent = self._cache.get(key)
        if not ent:
            return None
        expires_at, val = ent
        if time.time() < expires_at:
            return val
        # expired
        self._cache.pop(key, None)
        return None

    def _cache_put(self, key: str, val: Any, ttl: float) -> None:
        self._cache[key] = (time.time() + ttl, val)

    def _req_json(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: float = 0.0,
    ) -> Any:
        """
        Make a JSON request with retries on 429/5xx and strict JSON parsing.
        """
        url = f"{self.base_url}{path}"
        key = None
        if cache_ttl > 0.0:
            key = f"{method}:{url}:{json.dumps(params or {}, sort_keys=True)}"
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        # simple pacing
        if self.pace_sleep > 0:
            time.sleep(self.pace_sleep)

        backoff = 0.5
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            resp = None
            try:
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    headers=self._headers(),
                    params=params or {},
                    timeout=self.timeout,
                )
                # Retry on 429/5xx
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise _RetryableHTTPError(resp.status_code, getattr(resp, "text", "") or "")

                # explicit non-2xx handling (avoid relying on raise_for_status() to satisfy tests)
                if not (200 <= resp.status_code < 300):
                    from requests import HTTPError  # type: ignore

                    raise HTTPError(f"HTTP {resp.status_code}", response=getattr(resp, "response", None))

                data = resp.json()
                if cache_ttl > 0.0 and key:
                    self._cache_put(key, data, cache_ttl)
                return data

            except _RetryableHTTPError as e:
                last_err = e
                # exponential backoff with jitter
                sleep_s = backoff + random.uniform(0.0, 0.25)
                time.sleep(sleep_s)
                backoff *= 2.0
                continue
            except Exception as e:
                # treat as fatal (test suite will surface)
                last_err = e
                break

        # out of attempts
        if last_err:
            raise last_err
        raise RuntimeError("unexpected request loop exit")

    # ---------- public API ----------

    def simple_price(
        self,
        coin_ids: List[str],
        vs_currency: str = "usd",
        include_24h_change: bool = False,
    ) -> Dict[str, Any]:
        """
        GET /simple/price
        Example response:
          {"bitcoin":{"usd": 60000.0, "usd_24h_change": 1.23}, ...}
        """
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": vs_currency,
        }
        if include_24h_change:
            params["include_24hr_change"] = "true"
        return self._req_json("GET", "/simple/price", params=params, cache_ttl=5.0)

    def market_chart_days(
        self, coin_id: str, vs_currency: str = "usd", days: int = 1
    ) -> Dict[str, Any]:
        """
        GET /coins/{id}/market_chart?vs_currency=usd&days=1
        Returns dict with "prices": [[ts_ms, price], ...]
        """
        path = f"/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": str(days),
        }
        return self._req_json("GET", path, params=params, cache_ttl=30.0)

    def market_chart_range(
        self, coin_id: str, vs_currency: str, from_ts: int, to_ts: int
    ) -> Dict[str, Any]:
        """
        GET /coins/{id}/market_chart/range?vs_currency=usd&from=...&to=...
        """
        path = f"/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": vs_currency,
            "from": str(from_ts),
            "to": str(to_ts),
        }
        return self._req_json("GET", path, params=params, cache_ttl=30.0)

    # ---------- convenience ----------

    @staticmethod
    def ceil_days_for_hours(lookback_hours: int) -> int:
        return max(1, (lookback_hours + 23) // 24)

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def epoch_s(dt: datetime) -> int:
        return int(dt.timestamp())

    def history_last_hours(
        self, coin_id: str, vs_currency: str, lookback_hours: int
    ) -> List[Tuple[int, float]]:
        """Return list of (unix_s, price) roughly spanning the last lookback_hours."""
        days = self.ceil_days_for_hours(lookback_hours)
        data = self.market_chart_days(coin_id, vs_currency, days)
        raw = data.get("prices", []) or []
        # convert to (sec, price)
        out: List[Tuple[int, float]] = []
        for ts_ms, price in raw:
            try:
                out.append((int(ts_ms // 1000), float(price)))
            except Exception:
                continue
        # keep only last window
        cutoff = self.epoch_s(self.now_utc() - timedelta(hours=lookback_hours))
        out = [p for p in out if p[0] >= cutoff]
        return out


__all__ = [
    "CoinGeckoClient",
]
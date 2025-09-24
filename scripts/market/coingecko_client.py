# scripts/market/coingecko_client.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests


class _RetryableHTTPError(Exception):
    def __init__(self, code: int, body: str = ""):
        super().__init__(f"Retryable HTTP error {code}: {body[:200]}")
        self.code = code
        self.body = body


@dataclass
class CoinGeckoConfig:
    base_url: str = os.getenv("MW_CG_BASE_URL", "https://api.coingecko.com/api/v3").rstrip("/")
    api_key: str = os.getenv("MW_CG_API_KEY", "")
    timeout: float = float(os.getenv("MW_CG_TIMEOUT", "15"))  # seconds
    max_retries: int = int(os.getenv("MW_CG_MAX_RETRIES", "3"))
    pace_sleep: float = float(os.getenv("MW_CG_PACE_SLEEP", "0.0"))  # seconds between calls
    # soft client-side rate limit (simple pacing): requests per minute
    rate_limit_per_min: int = int(os.getenv("MW_CG_RATE_LIMIT_PER_MIN", "25"))


class CoinGeckoClient:
    """
    Minimal CG client with:
      - Header injection (x-cg-pro-api-key) when key present
      - Retry on 429 / 5xx
      - Optional small cache by (method, url, params)
      - Defensive handling for tests that monkeypatch requests.Session.request and
        return fake Response objects without raise_for_status()
    """

    def __init__(self, cfg: Optional[CoinGeckoConfig] = None):
        self.cfg = cfg or CoinGeckoConfig()
        self.base_url = self.cfg.base_url
        self.timeout = self.cfg.timeout
        self.max_retries = self.cfg.max_retries
        self.pace_sleep = self.cfg.pace_sleep
        self.session = requests.Session()
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key -> (expiry_ts, value)
        # simple rate pacing
        self._min_spacing = 60.0 / max(1, self.cfg.rate_limit_per_min)

    # ---------- helpers ----------

    def _headers(self) -> Dict[str, str]:
        h = {}
        if self.cfg.api_key:
            h["x-cg-pro-api-key"] = self.cfg.api_key
        return h

    def _cache_get(self, key: str) -> Optional[Any]:
        now = time.time()
        hit = self._cache.get(key)
        if not hit:
            return None
        exp, val = hit
        if now <= exp:
            return val
        self._cache.pop(key, None)
        return None

    def _cache_put(self, key: str, ttl: float, value: Any) -> None:
        self._cache[key] = (time.time() + ttl, value)

    def _req_json(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, cache_ttl: float = 0.0) -> Any:
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

                # Some tests return a fake object without raise_for_status(); handle defensively
                if hasattr(resp, "raise_for_status"):
                    resp.raise_for_status()
                else:
                    # emulate raise_for_status for non-2xx
                    if not (200 <= int(getattr(resp, "status_code", 0)) < 300):
                        raise Exception(f"HTTP {getattr(resp, 'status_code', '??')} without raise_for_status")

                # Parse JSON strictly
                data = resp.json()
                if cache_ttl > 0.0 and key:
                    self._cache_put(key, cache_ttl, data)
                # respect min spacing after success
                if self._min_spacing > 0:
                    time.sleep(self._min_spacing)
                return data

            except _RetryableHTTPError as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                break
            except Exception as e:
                last_err = e
                # treat all other exceptions as fail-fast (not retry), unless you want to retry JSON/timeout as well
                break

        assert last_err is not None
        raise last_err

    # ---------- public endpoints we use ----------

    def ping(self) -> Any:
        return self._req_json("GET", "/ping", {}, cache_ttl=5.0)

    def simple_price(self, ids: List[str], vs_currency: str, include_24h_change: bool = True) -> Dict[str, Any]:
        params = {
            "ids": ",".join(ids),
            "vs_currencies": vs_currency,
            "include_24hr_change": str(include_24h_change).lower(),
        }
        data = self._req_json("GET", "/simple/price", params=params, cache_ttl=5.0)
        if not isinstance(data, dict):
            raise TypeError(f"Unexpected simple/price payload type: {type(data)}")
        return data

    def market_chart_days(self, coin_id: str, vs_currency: str, days: int) -> Dict[str, Any]:
        """
        Standard market_chart endpoint with `days` granularity.
        """
        params = {"vs_currency": vs_currency, "days": str(days)}
        data = self._req_json("GET", f"/coins/{coin_id}/market_chart", params=params, cache_ttl=0.0)
        if not isinstance(data, dict) or "prices" not in data:
            raise TypeError(f"Unexpected market_chart payload: keys={list(data) if isinstance(data, dict) else type(data)}")
        return data

    def market_chart_hours(self, coin_id: str, vs_currency: str, hours: int) -> Dict[str, Any]:
        """
        Convenience wrapper: use `days=ceil(hours/24)` then let caller trim.
        """
        days = max(1, (hours + 23) // 24)
        return self.market_chart_days(coin_id, vs_currency, days)
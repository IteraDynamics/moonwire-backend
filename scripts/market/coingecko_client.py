# scripts/market/coingecko_client.py
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from json import JSONDecodeError


class _RetryableHTTPError(Exception):
    def __init__(self, status: int, body: str = ""):
        super().__init__(f"retryable HTTP error {status}: {body[:240]}")
        self.status = status
        self.body = body


def _safe_snip(s: str, n: int = 240) -> str:
    s = (s or "").replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


class CoinGeckoClient:
    """
    Minimal CoinGecko client with:
      - optional x-cg-pro-api-key header
      - retry on 429/5xx with exponential backoff
      - strict JSON + shape checks, good error messages
      - tiny in-process cache for short TTLs
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 2,
        pace_sleep: float = 0.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url or os.getenv("MW_CG_BASE_URL", "https://pro-api.coingecko.com/api/v3")
        self.api_key = api_key or os.getenv("MW_CG_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.pace_sleep = pace_sleep
        self.session = session or requests.Session()
        self._cache_store: Dict[str, tuple[float, Any]] = {}

    # ----------------- public endpoints -----------------

    def simple_price(self, ids: List[str], vs_currency: str, include_24h_change: bool = True) -> Dict[str, Any]:
        params = {
            "ids": ",".join(ids),
            "vs_currencies": vs_currency,
            "include_24hr_change": str(include_24h_change).lower(),
        }
        data = self._req_json("GET", "/simple/price", params=params, cache_ttl=5.0)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict from /simple/price, got {type(data).__name__}")
        return data

    def market_chart_days(self, coin_id: str, vs_currency: str, days: int) -> Dict[str, Any]:
        path = f"/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": str(days)}
        data = self._req_json("GET", path, params=params, cache_ttl=10.0)
        if not (isinstance(data, dict) and "prices" in data):
            raise TypeError(
                f"Expected dict with 'prices' from market_chart, got {type(data).__name__} "
                f"keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}"
            )
        return data

    # ----------------- internals -----------------

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            # CoinGecko Pro header
            h["x-cg-pro-api-key"] = self.api_key
        return h

    def _cache_get(self, key: str) -> Optional[Any]:
        now = time.time()
        hit = self._cache_store.get(key)
        if not hit:
            return None
        exp, val = hit
        if exp < now:
            self._cache_store.pop(key, None)
            return None
        return val

    def _cache_set(self, key: str, val: Any, ttl: float) -> None:
        self._cache_store[key] = (time.time() + ttl, val)

    def _req_json(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: float = 0.0,
    ) -> Any:
        """
        Make a JSON request with retries on 429/5xx and strict JSON parsing.
        On failure, raise an exception with enough detail to show *why* it failed.
        """
        url = f"{self.base_url}{path}"

        key = None
        if cache_ttl > 0.0:
            key = f"{method}:{url}:{json.dumps(params or {}, sort_keys=True)}"
            cached = self._cache_get(key)
            if cached is not None:
                return cached

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

                # 429/5xx => retry
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise _RetryableHTTPError(resp.status_code, getattr(resp, "text", "") or "")

                # Raise for other non-2xx (support fake response in tests)
                if hasattr(resp, "raise_for_status"):
                    try:
                        resp.raise_for_status()
                    except Exception as http_err:
                        body = _safe_snip(getattr(resp, "text", "") or "")
                        ctype = (getattr(resp, "headers", {}) or {}).get("Content-Type", "")
                        raise RuntimeError(
                            f"HTTP {resp.status_code} for {path} (ctype={ctype}) body={body}"
                        ) from http_err
                else:
                    # Manual check (unit tests' fake response objects)
                    if not (200 <= int(getattr(resp, "status_code", 0)) < 300):
                        body = _safe_snip(getattr(resp, "text", "") or "")
                        raise RuntimeError(f"HTTP {resp.status_code} for {path} body={body}")

                # Parse JSON (give helpful errors)
                try:
                    data = resp.json()
                except JSONDecodeError as jde:
                    body = _safe_snip(getattr(resp, "text", "") or "")
                    ctype = (getattr(resp, "headers", {}) or {}).get("Content-Type", "")
                    raise RuntimeError(
                        f"JSON decode failed for {path} (ctype={ctype}): {body}"
                    ) from jde
                except Exception as e:
                    # In some cases requests may raise a generic ValueError; surface context
                    body = _safe_snip(getattr(resp, "text", "") or "")
                    raise RuntimeError(f"JSON parse error for {path}: {e}: {body}") from e

                if cache_ttl > 0.0 and key:
                    self._cache_set(key, data, cache_ttl)
                return data

            except _RetryableHTTPError as e:
                last_err = e
                time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
                continue
            except Exception as e:
                last_err = e
                break  # non-retryable (JSON/shape/4xx other than 429)

        raise RuntimeError(f"request failed for {path}: {last_err.__class__.__name__}: {last_err}") from last_err


# Convenience for local quick checks
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
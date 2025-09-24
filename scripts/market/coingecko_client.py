# scripts/market/coingecko_client.py
from __future__ import annotations

import os
import time
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests


@dataclass
class _CacheEntry:
    value: Any
    ttl_s: float
    t0: float


class CoinGeckoClient:
    """
    Minimal CoinGecko client with:
      - base URL switching (demo vs pro)
      - x-cg-pro-api-key header (if provided)
      - simple in-run cache
      - retries with exponential backoff + jitter on 429/5xx
      - rate pacing (best-effort)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_per_min: int = 25,
        timeout_connect: int = 5,
        timeout_read: int = 10,
        max_retries: int = 4,
    ):
        # If caller didn’t supply base_url, pick one based on whether an API key is set.
        if not base_url:
            base_url = "https://pro-api.coingecko.com/api/v3" if api_key else "https://api.coingecko.com/api/v3"

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = (timeout_connect, timeout_read)
        self.max_retries = max_retries
        self.session = requests.Session()
        self.cache: Dict[str, _CacheEntry] = {}
        self.pace_sleep = max(60.0 / max(1, max_per_min) * 1.10, 0.0)  # tiny headroom

    # -----------------------
    # Internal HTTP helpers
    # -----------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["x-cg-pro-api-key"] = self.api_key
        return h

    def _cache_get(self, key: str) -> Optional[Any]:
        ent = self.cache.get(key)
        if not ent:
            return None
        if time.time() - ent.t0 <= ent.ttl_s:
            return ent.value
        # expired
        self.cache.pop(key, None)
        return None

    def _cache_put(self, key: str, value: Any, ttl_s: float = 15.0):
        self.cache[key] = _CacheEntry(value=value, ttl_s=ttl_s, t0=time.time())

    def _req_json(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, cache_ttl: float = 0.0) -> Any:
        """
        Make a JSON request with retries on 429/5xx.
        Returns parsed JSON (dict/list).
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
                # Handle 429/5xx with retry
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise _RetryableHTTPError(resp.status_code, resp.text)

                # On other non-2xx, raise with detail
                resp.raise_for_status()

                data = resp.json()
                if cache_ttl > 0.0 and key is not None:
                    self._cache_put(key, data, cache_ttl)
                return data
            except _RetryableHTTPError as e:
                if attempt >= self.max_retries:
                    raise
                # exponential backoff + jitter
                time.sleep(backoff + random.random() * 0.25)
                backoff *= 2.0
            except requests.RequestException as e:
                # network-ish errors are retryable
                if attempt >= self.max_retries:
                    # surface body when available for debugging
                    body = None
                    try:
                        body = resp.text if resp is not None else None  # type: ignore
                    except Exception:
                        pass
                    detail = f"{e.__class__.__name__}: {str(e)}"
                    if body:
                        detail += f" | body={body[:200]}"
                    raise RuntimeError(detail)
                time.sleep(backoff + random.random() * 0.25)
                backoff *= 2.0

        raise RuntimeError("unreachable retry loop")

    # -----------------------
    # Public API
    # -----------------------

    def get_simple_price(self, ids: List[str], vs_currency: str) -> Dict[str, Any]:
        """
        GET /simple/price
        Example return: {"bitcoin":{"usd":12345.6}, "ethereum":{"usd":3210.0}}
        """
        if not ids:
            return {}
        params = {
            "ids": ",".join(ids),
            "vs_currencies": vs_currency,
            "include_24hr_change": "true",
        }
        data = self._req_json("GET", "/simple/price", params=params, cache_ttl=5.0)
        if not isinstance(data, dict):
            raise TypeError(f"/simple/price returned non-dict: {type(data)}")
        return data

    def get_market_chart(self, coin_id: str, vs_currency: str, days: int, interval: Optional[str] = None) -> Dict[str, Any]:
        """
        GET /coins/{id}/market_chart?vs_currency=usd&days=3
        Returns dict with 'prices': [[ts_ms, price], ...]
        """
        if days <= 0:
            days = 1
        params = {"vs_currency": vs_currency, "days": days}
        if interval:
            params["interval"] = interval
        data = self._req_json("GET", f"/coins/{coin_id}/market_chart", params=params, cache_ttl=5.0)
        if not isinstance(data, dict):
            raise TypeError(f"/market_chart returned non-dict: {type(data)}")
        if "prices" not in data or not isinstance(data["prices"], list):
            raise TypeError(f"/market_chart missing 'prices' list: keys={list(data.keys())}")
        return data


class _RetryableHTTPError(Exception):
    def __init__(self, status_code: int, body: Optional[str]):
        super().__init__(f"retryable status {status_code}")
        self.status_code = status_code
        self.body = body
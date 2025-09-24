# scripts/market/coingecko_client.py
from __future__ import annotations

import os
import json
import time
import math
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import requests


class _RetryableHTTPError(Exception):
    def __init__(self, status_code: int, text: str):
        super().__init__(f"HTTP {status_code}: {text[:200]}")
        self.status_code = status_code
        self.text = text


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_unix_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _as_bool_string(val: bool) -> str:
    # CoinGecko expects 'true'/'false' strings for some params
    return "true" if bool(val) else "false"


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


class CoinGeckoClient:
    """
    Minimal CoinGecko HTTP client with:
    - base URL switching (demo vs pro) by env
    - optional x-cg-pro-api-key
    - basic pacing (headroom for CI)
    - retries with jitter for 429/5xx
    - tiny in-run cache to avoid duplicate hits
    """

    def __init__(self):
        base_env = os.getenv("MW_CG_BASE_URL", "").strip()
        api_key = os.getenv("MW_CG_API_KEY", "").strip()
        # Default base URL: Demo if no key; Pro if key present (but allow override by MW_CG_BASE_URL)
        if base_env:
            self.base_url = base_env.rstrip("/")
        else:
            self.base_url = "https://pro-api.coingecko.com/api/v3" if api_key else "https://api.coingecko.com/api/v3"

        self.api_key = api_key or None

        # pacing: keep headroom under plan limits
        rpm = int(os.getenv("MW_CG_RATE_LIMIT_PER_MIN", "25") or "25")
        self.pace_sleep = max(0.0, 60.0 / max(1, rpm))  # seconds per call

        self.timeout = (5.0, 10.0)  # connect, read
        self.max_retries = 4

        self.session = requests.Session()
        self._cache: Dict[str, Tuple[float, Any]] = {}

    def _headers(self) -> Dict[str, str]:
        h = {"accept": "application/json"}
        if self.api_key:
            h["x-cg-pro-api-key"] = self.api_key
        return h

    def _cache_get(self, key: str) -> Optional[Any]:
        item = self._cache.get(key)
        if not item:
            return None
        expires, value = item
        if time.time() < expires:
            return value
        self._cache.pop(key, None)
        return None

    def _cache_put(self, key: str, value: Any, ttl: float):
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

                resp.raise_for_status()
                try:
                    data = resp.json()
                except Exception as e:
                    raise TypeError(f"invalid JSON: {e}") from e

                if cache_ttl > 0.0 and key:
                    self._cache_put(key, data, cache_ttl)
                return data

            except _RetryableHTTPError as e:
                last_err = e
                # exponential backoff + jitter
                time.sleep(backoff + random.random() * 0.25)
                backoff = min(4.0, backoff * 2.0)
            except requests.RequestException as e:
                # network-level or 4xx non-429: do not retry (except you could retry 408/409 if desired)
                last_err = e
                break
            except Exception as e:
                # parsing etc.
                last_err = e
                break

        if last_err:
            raise last_err
        raise RuntimeError("unreachable")

    # ---------- Public endpoints ----------

    def simple_price(self, ids: List[str], vs: str, include_24h_change: bool = True) -> Dict[str, Any]:
        """
        GET /simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true
        Returns dict keyed by coin id.
        """
        if not ids:
            return {}
        params = {
            "ids": ",".join(ids),
            "vs_currencies": vs,
            "include_24hr_change": _as_bool_string(include_24h_change),
        }
        data = self._req_json("GET", "/simple/price", params=params, cache_ttl=5.0)
        if not isinstance(data, dict):
            raise TypeError(f"/simple/price: expected dict, got {type(data).__name__}")
        # schema guard: ensure nested keys present and numeric
        out: Dict[str, Dict[str, float]] = {}
        for cid in ids:
            entry = data.get(cid, {})
            if not isinstance(entry, dict):
                continue
            val = entry.get(vs)
            if _is_number(val):
                out[cid] = {vs: float(val)}
                chg = entry.get(f"{vs}_24h_change")
                if _is_number(chg):
                    out[cid][f"{vs}_24h_change"] = float(chg)
        return out

    def market_chart_days(self, coin_id: str, vs: str, days: int) -> List[Tuple[int, float]]:
        """
        GET /coins/{id}/market_chart?vs_currency=usd&days=3
        Returns list of (ts_ms, price)
        """
        params = {"vs_currency": vs, "days": str(max(1, int(days)))}
        data = self._req_json("GET", f"/coins/{coin_id}/market_chart", params=params, cache_ttl=10.0)
        prices = self._extract_prices_list(data)
        return prices

    def market_chart_range(self, coin_id: str, vs: str, from_ts: int, to_ts: int) -> List[Tuple[int, float]]:
        """
        GET /coins/{id}/market_chart/range?vs_currency=usd&from=...&to=...
        """
        params = {"vs_currency": vs, "from": str(int(from_ts)), "to": str(int(to_ts))}
        data = self._req_json("GET", f"/coins/{coin_id}/market_chart/range", params=params, cache_ttl=10.0)
        prices = self._extract_prices_list(data)
        return prices

    # ---------- Helpers ----------

    @staticmethod
    def _extract_prices_list(data: Any) -> List[Tuple[int, float]]:
        # Expect {"prices": [[ts_ms, price], ...]}
        if not isinstance(data, dict):
            raise TypeError(f"market_chart: expected dict, got {type(data).__name__}")
        raw = data.get("prices")
        if not isinstance(raw, list):
            raise TypeError("market_chart: 'prices' missing or not a list")
        out: List[Tuple[int, float]] = []
        for row in raw:
            if isinstance(row, (list, tuple)) and len(row) >= 2 and _is_number(row[0]) and _is_number(row[1]):
                ts_ms = int(row[0])
                price = float(row[1])
                out.append((ts_ms, price))
        if not out:
            raise ValueError("market_chart: no valid price points parsed")
        return out

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass

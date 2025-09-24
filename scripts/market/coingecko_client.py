from __future__ import annotations

import os, time, json, math, random
from typing import Any, Dict, Iterable, Optional
from dataclasses import dataclass, field

import requests


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "on")


@dataclass
class RateLimiter:
    per_minute: int = 25
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default_factory=time.time, init=False)

    def allow(self) -> None:
        """Simple token bucket: blocks (sleep) until 1 token is available."""
        rate = max(1, self.per_minute)
        now = time.time()
        elapsed = now - self._last
        self._last = now
        # refill
        self._tokens = min(rate, self._tokens + elapsed * (rate / 60.0))
        if self._tokens < 1.0:
            # need to wait
            need = 1.0 - self._tokens
            sleep_s = need * (60.0 / rate)
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0


@dataclass
class CoinGeckoClient:
    base_url: str = field(default_factory=lambda: os.getenv("MW_CG_BASE_URL", "https://api.coingecko.com/api/v3"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("MW_CG_API_KEY"))
    timeout_connect: float = 5.0
    timeout_read: float = 10.0
    per_minute: int = field(default_factory=lambda: int(os.getenv("MW_CG_RATE_LIMIT_PER_MIN", "25")))
    session: requests.Session = field(default_factory=requests.Session, init=False)
    cache: Dict[str, Any] = field(default_factory=dict, init=False)
    demo: bool = field(default_factory=lambda: _env_bool("MW_DEMO", False))

    def __post_init__(self):
        # If key is present and user forgot to switch base_url, prefer pro endpoint automatically
        if self.api_key and "pro-api" not in self.base_url:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.rl = RateLimiter(self.per_minute)

    # ---- private http ----
    def _headers(self) -> Dict[str, str]:
        h = {"accept": "application/json"}
        if self.api_key:
            h["x-cg-pro-api-key"] = self.api_key
        return h

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if self.demo:
            raise RuntimeError("demo-mode: HTTP disabled")  # caller handles demo synth
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        key = f"{method}|{url}|{json.dumps(params or {}, sort_keys=True)}"
        if key in self.cache:
            return self.cache[key]

        backoff = 0.5
        for attempt in range(5):
            self.rl.allow()
            try:
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    headers=self._headers(),
                    timeout=(self.timeout_connect, self.timeout_read),
                )
            except requests.RequestException:
                if attempt >= 3:
                    raise
                time.sleep(backoff + random.random() * 0.1)
                backoff *= 2
                continue

            if resp.status_code == 200:
                data = resp.json()
                self.cache[key] = data
                return data
            if resp.status_code in (429, 500, 502, 503, 504):
                # exponential backoff with jitter
                time.sleep(backoff + random.random() * 0.2)
                backoff = min(backoff * 2, 8.0)
                continue
            # other errors: raise
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:200]
            raise RuntimeError(f"HTTP {resp.status_code} for {path}: {detail}")

        raise RuntimeError(f"Failed after retries for {path}")

    # ---- public API ----
    def simple_price(self, ids: Iterable[str], vs: str = "usd", include_24h_change: bool = True) -> Dict[str, Any]:
        params = {
            "ids": ",".join(ids),
            "vs_currencies": vs,
            "include_24hr_change": "true" if include_24h_change else "false",
        }
        return self._request("GET", "/simple/price", params=params)

    def market_chart_days(self, coin_id: str, vs: str, days: int) -> Dict[str, Any]:
        # /coins/{id}/market_chart?vs_currency=usd&days=3
        path = f"/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs, "days": str(days), "interval": "hourly"}
        return self._request("GET", path, params=params)

    # helpers for tests
    def _set_demo(self, v: bool):
        self.demo = v
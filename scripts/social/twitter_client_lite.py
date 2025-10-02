# scripts/social/twitter_client_lite.py
from __future__ import annotations

import os
import time
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Iterable, List, Optional, Tuple

import requests


ISO = "%Y-%m-%dT%H:%M:%SZ"

DEFAULT_TIMEOUTS = (5.0, 10.0)  # (connect, read)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(ISO)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _split_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _backoff_sleep(attempt: int, base: float = 0.5, cap: float = 8.0) -> None:
    # exponential backoff with jitter
    delay = min(cap, base * (2 ** attempt)) * (0.5 + random.random())
    time.sleep(delay)


@dataclass
class TwitterLiteConfig:
    mode: str  # "api" | "demo" | "auto"
    keywords: List[str]
    lookback_h: int
    rate_limit_per_min: int
    bearer_token: Optional[str]
    demo: bool

    @classmethod
    def from_env(cls) -> "TwitterLiteConfig":
        mode = (os.getenv("MW_TWITTER_MODE") or "auto").lower().strip()
        keywords = _split_csv(os.getenv("MW_TWITTER_KEYWORDS") or "bitcoin,ethereum,solana,crypto")
        lookback_h = int(os.getenv("MW_TWITTER_LOOKBACK_H") or "72")
        rate_limit = int(os.getenv("MW_TWITTER_RATE_LIMIT_PER_MIN") or "25")
        bearer = os.getenv("MW_TWITTER_BEARER_TOKEN")
        demo_flag = _load_env_bool("MW_DEMO", False)

        if mode == "auto":
            if bearer and not demo_flag:
                mode = "api"
            else:
                mode = "demo"

        # If no bearer, force demo
        if not bearer:
            mode = "demo"

        return cls(
            mode=mode,
            keywords=keywords,
            lookback_h=lookback_h,
            rate_limit_per_min=rate_limit,
            bearer_token=bearer,
            demo=demo_flag or (mode == "demo"),
        )


class _RetryableHTTPError(RuntimeError):
    pass


class TwitterClientLite:
    """
    Minimal client for Twitter API v2 search recent + a synthetic demo generator fallback.
    """

    SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

    def __init__(self, cfg: TwitterLiteConfig):
        self.cfg = cfg
        self.session = requests.Session()

    # ---------------------
    # Public entry point(s)
    # ---------------------
    def fetch_recent(self) -> List[Dict[str, Any]]:
        """
        Return normalized tweet dicts (id, text, created_utc, author_id, metrics)
        within lookback window for provided keywords.
        """
        if self.cfg.mode == "demo":
            return self._demo_synthetic()

        return self._api_search_recent()

    # ---------------------
    # API mode
    # ---------------------
    def _api_search_recent(self) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.cfg.bearer_token}",
            "User-Agent": "MoonWire-TwitterLite/0.1",
        }
        now = _now_utc()
        start_time = now - timedelta(hours=self.cfg.lookback_h)
        start_iso = _iso(start_time)

        # Twitter API: build query with OR keywords
        query = " OR ".join([f'"{k}"' if " " in k else k for k in self.cfg.keywords]) or "crypto"

        max_results = 100
        params = {
            "query": query,
            "tweet.fields": "created_at,public_metrics,author_id,lang",
            "max_results": str(max_results),
            "start_time": start_iso,  # ISO-8601
        }

        out: List[Dict[str, Any]] = []
        next_token: Optional[str] = None

        # crude rate pacing
        min_interval = 60.0 / max(1, self.cfg.rate_limit_per_min)
        last_call = 0.0

        for page in range(10):  # hard cap ~1000 tweets to be polite
            if next_token:
                params["next_token"] = next_token
            else:
                params.pop("next_token", None)

            # respect pacing
            since = time.time() - last_call
            if since < min_interval:
                time.sleep(min_interval - since)

            # retry loop
            resp_json = None
            for attempt in range(5):
                try:
                    r = self.session.get(
                        self.SEARCH_URL,
                        headers=headers,
                        params=params,
                        timeout=DEFAULT_TIMEOUTS,
                    )
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise _RetryableHTTPError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    resp_json = r.json()
                    break
                except _RetryableHTTPError:
                    _backoff_sleep(attempt)
                    continue
                except requests.RequestException as e:
                    # for other network errors, try a couple of times
                    if attempt < 2:
                        _backoff_sleep(attempt)
                        continue
                    raise e
            last_call = time.time()

            if not resp_json:
                break

            data = resp_json.get("data") or []
            meta = resp_json.get("meta") or {}
            for tw in data:
                created = tw.get("created_at")
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    continue
                if created_dt < start_time:
                    continue

                pm = tw.get("public_metrics") or {}
                out.append(
                    {
                        "tweet_id": tw.get("id"),
                        "text": tw.get("text") or "",
                        "created_utc": _iso(created_dt),
                        "author_id": str(tw.get("author_id") or ""),
                        "metrics": {
                            "retweets": int(pm.get("retweet_count") or 0),
                            "likes": int(pm.get("like_count") or 0),
                        },
                        "lang": tw.get("lang") or "",
                        "source": "twitter_api",
                        "origin": "twitter",
                    }
                )

            next_token = meta.get("next_token")
            if not next_token:
                break

        return out

    # ---------------------
    # Demo mode
    # ---------------------
    def _demo_synthetic(self) -> List[Dict[str, Any]]:
        """
        Deterministic synthetic generator, stable across runs.
        """
        seed = 13
        random.seed(seed)

        now = _now_utc().replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=self.cfg.lookback_h)

        # base rates by keyword to create some variance
        base_rate = {
            "bitcoin": 10,
            "ethereum": 8,
            "solana": 6,
            "crypto": 5,
        }
        # Ensure keywords cover defaults
        kws = self.cfg.keywords or ["bitcoin", "ethereum", "solana", "crypto"]
        for k in kws:
            base_rate.setdefault(k, 4)

        out: List[Dict[str, Any]] = []
        t = start
        tweet_id_ctr = 1000000000
        while t < now:
            # per-hour bursty factor
            hour_factor = 1.0 + 0.8 * math.sin((t.hour / 24.0) * 2 * math.pi)
            for k in kws:
                lam = max(1, int(base_rate[k] * hour_factor))
                n = int(random.expovariate(1.0 / max(1, lam)))  # stochastic-ish
                for _ in range(min(n, 12)):
                    txt = f"{k.capitalize()} moves; volatility watch. #{k}"
                    out.append(
                        {
                            "tweet_id": str(tweet_id_ctr),
                            "text": txt,
                            "created_utc": _iso(t + timedelta(minutes=random.randint(0, 59))),
                            "author_id": str(900000 + random.randint(0, 9999)),
                            "metrics": {"retweets": random.randint(0, 50), "likes": random.randint(0, 200)},
                            "lang": "en",
                            "source": "demo",
                            "origin": "twitter",
                        }
                    )
                    tweet_id_ctr += 1
            t += timedelta(hours=1)

        return out
# scripts/social/reddit_client_lite.py
from __future__ import annotations

import os
import time
import json
import math
import random
import string
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import requests
import xml.etree.ElementTree as ET

# ---- helpers ----

_ISO = "%Y-%m-%dT%H:%M:%SZ"
logger = logging.getLogger(__name__)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(_ISO)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and str(v).strip() != "" else default


def _sleep_s(seconds: float) -> None:
    try:
        time.sleep(seconds)
    except Exception:
        pass


# ---- client configuration ----

DEFAULT_TIMEOUT_CONNECT = 5.0
DEFAULT_TIMEOUT_READ = 10.0


@dataclass
class LiteConfig:
    mode: str = "rss"  # "rss" | "api"
    subs: List[str] = None
    sort: str = "new"  # rss-only: "new" | "hot"
    lookback_h: int = 72
    rate_limit_per_min: int = 60  # api pacing
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    demo: bool = False

    @staticmethod
    def from_env() -> "LiteConfig":
        subs_env = _env("MW_REDDIT_SUBS", "CryptoCurrency,Bitcoin,ethtrader,Solana")
        subs = [s.strip() for s in subs_env.split(",") if s.strip()]
        return LiteConfig(
            mode=_env("MW_REDDIT_MODE", "rss").lower(),
            subs=subs,
            sort=_env("MW_REDDIT_SORT", "new").lower(),
            lookback_h=int(_env("MW_REDDIT_LOOKBACK_H", "72")),
            rate_limit_per_min=int(_env("MW_REDDIT_RATE_LIMIT_PER_MIN", "60")),
            client_id=_env("MW_REDDIT_CLIENT_ID"),
            client_secret=_env("MW_REDDIT_CLIENT_SECRET"),
            demo=(str(_env("MW_DEMO", "false")).lower() == "true"),
        )


# ---- in-run tiny cache ----

class TinyCache:
    def __init__(self):
        self._kv: Dict[str, Any] = {}

    def get(self, k: str) -> Any:
        return self._kv.get(k)

    def set(self, k: str, v: Any) -> None:
        self._kv[k] = v


# ---- HTTP + retry/backoff ----

class _Retryable(Exception):
    pass


def _backoff_delays(max_attempts: int = 4, base: float = 0.6, jitter: float = 0.4) -> Iterable[float]:
    for i in range(max_attempts):
        # exponential-ish with jitter
        yield base * (2 ** i) + random.random() * jitter


def _requests_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


# ---- RSS mode ----

def _parse_rss(xml_text: str) -> List[Dict[str, Any]]:
    """
    Return list of items with minimal fields:
    id/guid, title, published, author?, permalink (from link)
    """
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # RSS can be <entry> (atom) or <item> (rss)
    # Try both.
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "dc": "http://purl.org/dc/elements/1.1/",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }

    # Prefer Atom entries
    entries = list(root.findall(".//{http://www.w3.org/2005/Atom}entry"))
    if entries:
        for e in entries:
            pid = (e.findtext("{http://www.w3.org/2005/Atom}id") or "").strip()
            title = (e.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
            published = (e.findtext("{http://www.w3.org/2005/Atom}updated") or e.findtext("{http://www.w3.org/2005/Atom}published") or "").strip()
            author_el = e.find("{http://www.w3.org/2005/Atom}author")
            author = ""
            if author_el is not None:
                author = (author_el.findtext("{http://www.w3.org/2005/Atom}name") or "").strip()
            link_href = ""
            for link in e.findall("{http://www.w3.org/2005/Atom}link"):
                if (link.get("rel") in (None, "alternate")) and link.get("href"):
                    link_href = link.get("href")
                    break
            items.append({
                "id": pid or link_href or title,
                "title": title,
                "published": published,
                "author": author,
                "permalink": link_href,
            })
        return items

    # RSS items
    for i in root.findall(".//item"):
        guid = (i.findtext("guid") or "").strip()
        title = (i.findtext("title") or "").strip()
        pubdate = (i.findtext("pubDate") or "").strip()
        author = (i.findtext("author") or i.findtext("{http://purl.org/dc/elements/1.1/}creator") or "").strip()
        link = (i.findtext("link") or "").strip()
        items.append({
            "id": guid or link or title,
            "title": title,
            "published": pubdate,
            "author": author,
            "permalink": link,
        })
    return items


def _parse_any_datetime(s: str) -> Optional[datetime]:
    """
    Try a couple common RSS/Atom formats.
    """
    if not s:
        return None
    s = s.strip()
    # RFC 822-ish: Wed, 24 Sep 2025 19:04:05 GMT
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    # Last resort: try dateutil if available
    try:
        from dateutil import parser
        dt = parser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


class RedditLiteClient:
    """
    Two-mode client:
      - RSS (default): GET https://www.reddit.com/r/{sub}/{sort}.rss
      - API (optional): OAuth client credentials + listing JSON

    Handles retries, 429s, simple pacing, and a tiny in-run cache.
    """

    def __init__(self, cfg: Optional[LiteConfig] = None) -> None:
        self.cfg = cfg or LiteConfig.from_env()
        # mode auto-detect: if API creds exist but MW_REDDIT_MODE not "rss", use api
        if self.cfg.mode not in ("rss", "api"):
            self.cfg.mode = "rss"
        if self.cfg.mode == "api" and not (self.cfg.client_id and self.cfg.client_secret):
            self.cfg.mode = "rss"  # fallback

        self.session = _requests_session(user_agent="MoonWire/0.6.8 (reddit-lite)")
        self.cache = TinyCache()

        # API token (if api mode)
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

        # pacing
        self._min_interval = 60.0 / max(1, self.cfg.rate_limit_per_min)
        self._last_req_at = 0.0

    # ---- core request wrapper ----

    def _pace(self):
        delta = time.time() - self._last_req_at
        wait = self._min_interval - delta
        if wait > 0:
            _sleep_s(wait)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        self._pace()
        timeout = kwargs.pop("timeout", (DEFAULT_TIMEOUT_CONNECT, DEFAULT_TIMEOUT_READ))
        for delay in _backoff_delays():
            try:
                resp = self.session.request(method=method.upper(), url=url, timeout=timeout, **kwargs)
                self._last_req_at = time.time()
                if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                    raise _Retryable(f"{resp.status_code} {getattr(resp, 'text', '')[:200]}")
                resp.raise_for_status()
                return resp
            except _Retryable:
                _sleep_s(delay)
            except requests.RequestException:
                _sleep_s(delay)
        # final attempt
        resp = self.session.request(method=method.upper(), url=url, timeout=timeout, **kwargs)
        self._last_req_at = time.time()
        resp.raise_for_status()
        return resp

    # ---- API auth ----

    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 30:
            return self._token
        auth = requests.auth.HTTPBasicAuth(self.cfg.client_id, self.cfg.client_secret)  # type: ignore[arg-type]
        data = {"grant_type": "client_credentials", "duration": "1h"}
        resp = self._request("POST", "https://www.reddit.com/api/v1/access_token", data=data, auth=auth)
        payload = resp.json()
        tok = payload.get("access_token")
        expires = float(payload.get("expires_in", 3600))
        if not tok:
            raise RuntimeError("failed to obtain reddit token")
        self._token = tok
        self._token_expiry = time.time() + expires
        return tok

    # ---- public API ----

    def fetch_rss(self, sub: str) -> List[Dict[str, Any]]:
        sort = self.cfg.sort if self.cfg.sort in ("new", "hot") else "new"
        url = f"https://www.reddit.com/r/{sub}/{sort}.rss"
        cache_key = f"rss:{url}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        resp = self._request("GET", url)
        items = _parse_rss(resp.text)
        self.cache.set(cache_key, items)
        return items

    def fetch_api_listing(self, sub: str, limit: int = 100, after: Optional[str] = None) -> Dict[str, Any]:
        tok = self._ensure_token()
        params = {"limit": str(min(100, max(1, limit)))}
        if after:
            params["after"] = after
        url = f"https://oauth.reddit.com/r/{sub}/new.json?{urlencode(params)}"
        headers = {"Authorization": f"bearer {tok}"}
        resp = self._request("GET", url, headers=headers)
        return resp.json()

    def list_recent_posts(self, sub: str) -> List[Dict[str, Any]]:
        """
        Fetch posts per mode and normalize to a shared lite schema:

        {
          "id": "...",
          "title": "...",
          "created_utc": <datetime>,
          "author": "name" | "",   # rss may miss
          "permalink": "/r/..../comments/....",
          "fields": { "score": int, "num_comments": int }  # api only
        }
        """
        lookback_start = _now_utc() - timedelta(hours=self.cfg.lookback_h)
        out: List[Dict[str, Any]] = []

        if self.cfg.demo:
            # deterministic demo seeds
            rnd = random.Random(42 + sum(ord(c) for c in sub))
            base = _now_utc().replace(minute=0, second=0, microsecond=0) - timedelta(hours=self.cfg.lookback_h)
            for i in range(self.cfg.lookback_h):
                # ~Poisson(λ≈1.2) posts per hour
                n = 0
                # simple deterministic pattern
                if i % 7 in (1, 3, 5):
                    n = 1
                if i % 11 == 0:
                    n += 1
                for j in range(n):
                    ts = base + timedelta(hours=i, minutes=5 * j)
                    out.append({
                        "id": f"t3_demo_{sub}_{i}_{j}",
                        "title": f"{sub} demo post {i}-{j}",
                        "created_utc": ts,
                        "author": f"demo_user_{rnd.randint(1, 100)}",
                        "permalink": f"/r/{sub}/comments/demo_{i}_{j}",
                        "fields": {"score": rnd.randint(1, 500), "num_comments": rnd.randint(0, 120)},
                    })
            return out

        if self.cfg.mode == "api":
            try:
                after: Optional[str] = None
                for _ in range(5):  # modest pagination cap
                    payload = self.fetch_api_listing(sub, limit=100, after=after)
                    children = (payload.get("data") or {}).get("children", [])
                    for ch in children:
                        d = ch.get("data") or {}
                        created = d.get("created_utc")
                        dt = datetime.fromtimestamp(float(created), tz=timezone.utc) if created else None
                        if not dt or dt < lookback_start:
                            continue
                        out.append({
                            "id": d.get("name") or f"t3_{d.get('id','')}",
                            "title": (d.get("title") or "").strip(),
                            "created_utc": dt,
                            "author": (d.get("author") or "").strip(),
                            "permalink": (d.get("permalink") or "").strip(),
                            "fields": {
                                "score": int(d.get("score") or 0),
                                "num_comments": int(d.get("num_comments") or 0),
                            },
                        })
                    after = (payload.get("data") or {}).get("after")
                    if not after:
                        break
                    # light pacing between pages
                    _sleep_s(0.5)
                return out
            except Exception as e:
                logger.warning("reddit api failed, falling back to rss: %s", e)

        # RSS fallback
        try:
            items = self.fetch_rss(sub)
            for it in items:
                dt = _parse_any_datetime(it.get("published") or "")
                if not dt or dt < lookback_start:
                    continue
                out.append({
                    "id": (it.get("id") or it.get("permalink") or it.get("title") or "").strip(),
                    "title": (it.get("title") or "").strip(),
                    "created_utc": dt,
                    "author": (it.get("author") or "").strip(),
                    "permalink": (it.get("permalink") or "").strip(),
                    "fields": {},  # rss mode has no score/comments reliably
                })
            return out
        except Exception as e:
            logger.error("rss fetch failed for /r/%s: %s", sub, e)
            return []
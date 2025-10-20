# scripts/social/reddit_api_client.py
from __future__ import annotations

import os
import time
import random
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

REDDIT_OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE = "https://oauth.reddit.com"


def _utc_iso(ts: float | int) -> str:
    """Convert epoch seconds to UTC ISO (Z)."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_float_header(val: Optional[str], default: float) -> float:
    """
    Reddit sometimes returns comma-separated values in x-ratelimit headers.
    Take the first parseable float.
    """
    if not val:
        return default
    for part in str(val).split(","):
        part = part.strip()
        try:
            return float(part)
        except Exception:
            continue
    return default


@dataclass
class RedditCreds:
    client_id: str
    client_secret: str
    user_agent: str = "moonwire/1.0 (by u/moonwire)"

    @staticmethod
    def from_env() -> "RedditCreds":
        cid = os.getenv("REDDIT_CLIENT_ID", "").strip()
        csec = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
        ua = os.getenv("REDDIT_USER_AGENT", "moonwire/1.0 (by u/moonwire)").strip()
        if not cid or not csec:
            raise RuntimeError("Missing REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET")
        return RedditCreds(cid, csec, ua)


class RedditApiClient:
    """
    Minimal first-party Reddit API client using client-credentials OAuth2.

    - Auth scope: 'read'
    - Primary endpoint: /r/{sub}/new (Listing API)
    - Pagination: 'after' fullname
    - Ratelimit: X-Ratelimit-Used/Remaining/Reset headers
    - Retries: jittered exponential backoff on transient HTTP/net errors
    """

    def __init__(self, creds: RedditCreds, session: Optional[requests.Session] = None):
        self.creds = creds
        self.sess = session or requests.Session()
        self.token: Optional[str] = None
        self._token_expiry = 0.0  # epoch seconds

    # --------------------
    # OAuth
    # --------------------
    def _ensure_token(self) -> None:
        now = time.time()
        if self.token and now < self._token_expiry - 30:
            return

        auth = requests.auth.HTTPBasicAuth(self.creds.client_id, self.creds.client_secret)
        # Be explicit about scope=read
        data = {"grant_type": "client_credentials", "duration": "temporary", "scope": "read"}
        headers = {"User-Agent": self.creds.user_agent}

        resp = self.sess.post(REDDIT_OAUTH_URL, data=data, headers=headers, auth=auth, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        self.token = j.get("access_token")
        expires_in = int(j.get("expires_in", 3600))
        self._token_expiry = now + expires_in
        log.info("Reddit OAuth token acquired; expires in %ss", expires_in)

    def _headers(self) -> Dict[str, str]:
        self._ensure_token()
        return {
            "Authorization": f"bearer {self.token}",
            "User-Agent": self.creds.user_agent,
        }

    # --------------------
    # Ratelimit & retries
    # --------------------
    def _respect_rate_limit(self, resp: requests.Response) -> None:
        remaining = _parse_float_header(resp.headers.get("x-ratelimit-remaining"), 1.0)
        reset_sec = _parse_float_header(resp.headers.get("x-ratelimit-reset"), 1.0)
        if remaining <= 0.2:
            sleep_s = max(1.0, reset_sec)
            log.warning("Rate limit near zero; sleeping %.1fs", sleep_s)
            time.sleep(sleep_s)

    def _request_with_retries(self, method: str, url: str, *, timeout: int = 30, **kwargs) -> requests.Response:
        max_tries = int(os.getenv("MW_HTTP_MAX_TRIES", "5"))
        base_sleep = float(os.getenv("MW_HTTP_BASE_SLEEP", "0.5"))
        tries = 0
        while True:
            try:
                self._ensure_token()
                resp = self.sess.request(method, url, headers=self._headers(), timeout=timeout, **kwargs)
                if resp.status_code == 429:
                    self._respect_rate_limit(resp)
                resp.raise_for_status()
                self._respect_rate_limit(resp)
                return resp
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (401, 403):
                    # Token expired or scope issue — refresh and retry
                    self.token = None
                # Only retry on typical transient codes
                if status not in (401, 403, 429, 500, 502, 503, 504):
                    raise
            except (requests.ConnectionError, requests.Timeout):
                pass

            tries += 1
            if tries >= max_tries:
                raise
            # jittered exponential backoff
            sleep_s = base_sleep * (2 ** (tries - 1)) * (1 + random.random() * 0.25)
            time.sleep(sleep_s)

    # --------------------
    # Listings
    # --------------------
    def fetch_new_page(
        self,
        subreddit: str,
        limit_per_page: int = 100,
        after: Optional[str] = None,
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Fetch a single page from /r/{subreddit}/new and map it to the JSONL schema
        expected by MoonWire’s social ingestion.

        Returns:
            (items, next_after)
            - items: List[Dict] with keys:
              {source, subreddit, created_utc, title, text, id, permalink}
            - next_after: fullname for the next page, or None if exhausted
        """
        params = {"limit": str(limit_per_page)}
        if after:
            params["after"] = after
        url = f"{REDDIT_API_BASE}/r/{subreddit}/new"

        resp = self._request_with_retries("GET", url, params=params)
        data = resp.json() or {}
        listing = (data.get("data") or {})
        children = listing.get("children") or []
        next_after = listing.get("after")

        items: List[Dict] = []
        for c in children:
            d = c.get("data") or {}
            # Strict schema: only the fields the pipeline expects
            items.append({
                "source": "reddit",
                "subreddit": d.get("subreddit") or subreddit,
                "created_utc": _utc_iso(d.get("created_utc", 0)),
                "title": d.get("title") or "",
                "text": d.get("selftext") or "",
                "id": d.get("name") or d.get("id"),  # fullname preferred
                "permalink": d.get("permalink") or "",
            })

        return items, next_after

# -*- coding: utf-8 -*-
"""
Lightweight Reddit client used by ingest tests.

Exports:
  - RedditLite (supports RSS by default, optional API with app-only OAuth)
  - requests (so tests can monkeypatch requests.Session.request)

The tests patch requests.Session.request in this module, so do not wrap/alias
requests anywhere else.
"""

from __future__ import annotations
import os
import time
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import requests  # tests patch requests.Session.request here


ISO = "%Y-%m-%dT%H:%M:%SZ"
def _iso(dt: datetime) -> str: return dt.strftime(ISO)

ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _find(el: ET.Element, tag_local: str) -> Optional[ET.Element]:
    """
    Find child by local tag name, tolerant to Atom namespace (or no ns).
    """
    x = el.find(f"{ATOM_NS}{tag_local}")
    if x is not None:
        return x
    return el.find(tag_local)


def _findall(el: ET.Element, tag_local: str) -> List[ET.Element]:
    xs = list(el.findall(f"{ATOM_NS}{tag_local}"))
    xs.extend(list(el.findall(tag_local)))
    return xs


def _findtext(el: ET.Element, tag_local: str, default: str = "") -> str:
    x = _find(el, tag_local)
    if x is None:
        return default
    return (x.text or "").strip()


def _parse_dt_fallback(txt: str) -> datetime:
    """
    Parse various timestamp formats we might see in feeds. Fallback to now(UTC).
    """
    if not txt:
        return datetime.now(timezone.utc)
    t = txt.strip()
    # Atom-ish: 2025-09-25T10:00:00Z
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        pass
    # RFC822-ish: Wed, 25 Sep 2025 10:00:00 +0000
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%d %b %Y %H:%M:%S %z"):
        try:
            return datetime.strptime(t, fmt).astimezone(timezone.utc)
        except Exception:
            continue
    return datetime.now(timezone.utc)


class _HTTP:
    """Minimal pacing + retry wrapper using THIS module's requests import."""
    def __init__(self, rate_per_min: int | str = 60):
        try:
            rpm = int(rate_per_min)  # env may pass string
        except Exception:
            rpm = 60
        self.session = requests.Session()
        self.rate_per_min = max(1, int(rpm))
        self._last = 0.0

    def _pace(self):
        gap = 60.0 / float(self.rate_per_min)
        now = time.time()
        if now - self._last < gap:
            time.sleep(gap - (now - self._last))
        self._last = time.time()

    def request(self, method: str, url: str, **kw) -> requests.Response:
        backoff = 0.5
        for attempt in range(5):
            self._pace()
            try:
                resp = self.session.request(method=method.upper(), url=url, timeout=(5, 10), **kw)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise RuntimeError(f"retryable:{resp.status_code}")
                resp.raise_for_status()
                return resp
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(backoff + random.uniform(0, 0.25))
                backoff *= 2.0
        raise RuntimeError("unreachable")


class RedditLite:
    """
    RSS mode (default): fetch https://www.reddit.com/r/{sub}/{sort}.rss
    API mode (optional): app-only OAuth then GET /r/{sub}/new.json
    """
    def __init__(self, mode: str = "rss"):
        self.mode = (mode or "rss").strip().lower()
        self.http = _HTTP(rate_per_min=os.getenv("MW_REDDIT_RATE_LIMIT_PER_MIN", "60"))
        self.client_id = os.getenv("MW_REDDIT_CLIENT_ID") or ""
        self.client_secret = os.getenv("MW_REDDIT_CLIENT_SECRET") or ""
        self._token: Optional[str] = None

    # ---------- RSS ----------
    def fetch_rss(self, sub: str, sort: str = "new") -> List[Dict[str, Any]]:
        """
        Parse Atom (namespaced or not). Tolerates classic RSS <item> as well.
        """
        url = f"https://www.reddit.com/r/{sub}/{sort}.rss"
        resp = self.http.request("GET", url, headers={"User-Agent": "moonwire/0.6.8 (+rss)"})
        data = resp.content if getattr(resp, "content", None) is not None else resp.text.encode("utf-8")
        root = ET.fromstring(data)
        out: List[Dict[str, Any]] = []

        # Prefer Atom <entry>; fallback to RSS <item>
        entries = _findall(root, "entry")
        if not entries:
            entries = root.findall(".//item")

        for entry in entries:
            # id/guid
            eid = _findtext(entry, "id") or _findtext(entry, "guid")

            # title
            title = _findtext(entry, "title")
            if not title:
                tnode = entry.find("title")
                title = (tnode.text or "").strip() if tnode is not None else ""

            # published/updated/pubDate
            published = _findtext(entry, "published") or _findtext(entry, "updated")
            if not published:
                pnode = entry.find("pubDate")
                published = (pnode.text or "").strip() if pnode is not None else ""

            dt = _parse_dt_fallback(published)

            # author: <author><name>… or <author>text or <dc:creator>…
            author = ""
            a_el = _find(entry, "author") or entry.find("author")  # non-ns
            if a_el is not None:
                name = _findtext(a_el, "name")
                author = (name or (a_el.text or "")).strip()
            if not author:
                dc = entry.find("{http://purl.org/dc/elements/1.1/}creator")
                if dc is not None:
                    author = (dc.text or "").strip()

            # permalink: atom:link rel=alternate OR <link> text/href
            permalink = ""
            for l in _findall(entry, "link"):
                rel = (l.get("rel") or "").lower()
                href = l.get("href")
                if rel == "alternate" and href:
                    permalink = href
                    break
            if not permalink:
                l = entry.find("link")
                if l is not None:
                    permalink = (l.text or l.get("href") or "").strip()

            out.append({
                "id": eid or "",
                "title": title,
                "created_utc": _iso(dt),
                "permalink": permalink,
                "author": author or None,
            })
        return out

    # ---------- API ----------
    def _ensure_token(self) -> Optional[str]:
        if not (self.client_id and self.client_secret):
            return None
        if self._token:
            return self._token
        resp = self.http.request(
            "POST",
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            headers={"User-Agent": "moonwire/0.6.8 (+api)"},
        )
        tok = resp.json().get("access_token")
        if tok:
            self._token = tok
        return self._token

    def fetch_api_listing(self, sub: str) -> List[Dict[str, Any]]:
        tok = self._ensure_token()
        if not tok:
            return []
        out: List[Dict[str, Any]] = []
        after = None
        for _ in range(3):
            params = {"limit": "100"}
            if after:
                params["after"] = after
            resp = self.http.request(
                "GET",
                f"https://oauth.reddit.com/r/{sub}/new.json",
                headers={"Authorization": f"Bearer {tok}", "User-Agent": "moonwire/0.6.8 (+api)"},
                params=params,
            )
            js = resp.json() or {}
            data = js.get("data", {})
            after = data.get("after")
            for ch in data.get("children", []):
                d = ch.get("data", {})
                # created_utc is epoch seconds
                ts = d.get("created_utc")
                dt = datetime.fromtimestamp(ts, tz=timezone.utc) if isinstance(ts, (int, float)) else datetime.now(timezone.utc)
                out.append({
                    "id": d.get("name") or f"t3_{d.get('id', '')}",
                    "title": d.get("title", ""),
                    "selftext": (d.get("selftext", "") or "")[:800],
                    "created_utc": _iso(dt),
                    "score": int(d.get("score", 0) or 0),
                    "num_comments": int(d.get("num_comments", 0) or 0),
                    "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
                    "author": d.get("author") or None,
                })
            if not after:
                break
        return out
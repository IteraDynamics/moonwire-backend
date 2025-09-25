# -*- coding: utf-8 -*-
"""
Reddit Lite Ingest (RSS by default; optional API mode)
Writes:
  - logs/social_reddit.jsonl (append-only, one per kept post)
  - models/social_reddit_context.json (summary)
  - artifacts/reddit_activity_<sub>.png
  - artifacts/reddit_bursts_<sub>.png
Env:
  MW_REDDIT_MODE=rss|api (default rss)
  MW_REDDIT_SUBS=CryptoCurrency,Bitcoin,ethtrader,Solana
  MW_REDDIT_SORT=new (rss)
  MW_REDDIT_LOOKBACK_H=72
  MW_REDDIT_RATE_LIMIT_PER_MIN=60 (api)
  MW_REDDIT_CLIENT_ID, MW_REDDIT_CLIENT_SECRET (api)
  MW_DEMO=true/false
"""

from __future__ import annotations
import os, re, json, time, math, random, string
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import requests
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")  # safemode
import matplotlib.pyplot as plt

# -------------------
# Helpers
# -------------------

ISO = "%Y-%m-%dT%H:%M:%SZ"
def _iso(dt: datetime) -> str: return dt.strftime(ISO)

STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","by","as","at","is","are",
    "be","from","this","that","it","its","you","your","we","our","they","their",
    "i","me","my","was","were","will","shall","has","have","had"
}
WORD_RE = re.compile(r"[a-z0-9]+")

def _now() -> datetime:
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)

def _bounded_sleep(min_s=0.8, max_s=1.6):
    time.sleep(random.uniform(min_s, max_s))

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))

def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _bucket_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

def _zscore(vals: List[float]) -> List[float]:
    if not vals:
        return []
    mu = sum(vals)/len(vals)
    var = sum((x-mu)**2 for x in vals)/len(vals)
    sd = math.sqrt(var) if var>0 else 0.0
    if sd == 0:
        return [0.0]*len(vals)
    return [(x-mu)/sd for x in vals]

# -------------------
# Client(s)
# -------------------

class _HTTP:
    def __init__(self, rate_per_min: int = 60):
        self.session = requests.Session()
        self.rate_per_min = max(1, rate_per_min)
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
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    timeout=(5, 10),
                    **kw,
                )
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise RuntimeError(f"retryable:{resp.status_code}:{resp.text[:120]}")
                resp.raise_for_status()
                return resp
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(backoff + random.uniform(0, 0.25))
                backoff *= 2.0
        raise RuntimeError("unreachable")

class RedditLite:
    def __init__(self, mode: str = "rss"):
        self.mode = (mode or "rss").strip().lower()
        self.http = _HTTP(rate_per_min=int(os.getenv("MW_REDDIT_RATE_LIMIT_PER_MIN", "60")))
        self.client_id = os.getenv("MW_REDDIT_CLIENT_ID") or ""
        self.client_secret = os.getenv("MW_REDDIT_CLIENT_SECRET") or ""
        self._token: Optional[str] = None

    # ---- RSS ----
    def fetch_rss(self, sub: str, sort: str="new") -> List[Dict[str, Any]]:
        url = f"https://www.reddit.com/r/{sub}/{sort}.rss"
        resp = self.http.request("GET", url, headers={"User-Agent":"moonwire/0.6.8 (+rss)"},
                                 params={},)
        root = ET.fromstring(resp.content)
        ns = {"atom":"http://www.w3.org/2005/Atom"}
        out: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            # prefer id/link/published/title/author
            eid = entry.findtext("atom:id", default="", namespaces=ns).strip()
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            published = entry.findtext("atom:published", default="", namespaces=ns).strip()
            author_el = entry.find("atom:author", ns)
            author = ""
            if author_el is not None:
                author = (author_el.findtext("atom:name", default="", namespaces=ns) or "").strip()
            link = ""
            for l in entry.findall("atom:link", ns):
                if l.get("rel") == "alternate":
                    link = l.get("href") or ""
                    break
            # normalize
            created_dt = None
            if published:
                try:
                    created_dt = datetime.fromisoformat(published.replace("Z","+00:00")).astimezone(timezone.utc)
                except Exception:
                    created_dt = _now()
            out.append({
                "id": eid or "",
                "title": title,
                "created_utc": _iso(created_dt or _now()),
                "permalink": link,
                "author": author or None,
            })
        return out

    # ---- API ----
    def _ensure_token(self) -> Optional[str]:
        if not (self.client_id and self.client_secret):
            return None
        if self._token:
            return self._token
        resp = self.http.request(
            "POST",
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type":"client_credentials"},
            auth=(self.client_id, self.client_secret),
            headers={"User-Agent":"moonwire/0.6.8 (+api)"},
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
        for _ in range(3):  # a few pages is enough for 72h window
            params = {"limit":"100"}
            if after: params["after"] = after
            resp = self.http.request(
                "GET",
                f"https://oauth.reddit.com/r/{sub}/new.json",
                headers={"Authorization": f"Bearer {tok}", "User-Agent":"moonwire/0.6.8 (+api)"},
                params=params,
            )
            js = resp.json()
            data = js.get("data", {})
            after = data.get("after")
            for ch in data.get("children", []):
                d = ch.get("data", {})
                out.append({
                    "id": d.get("name") or f"t3_{d.get('id','')}",
                    "title": d.get("title",""),
                    "selftext": (d.get("selftext","") or "")[:800],
                    "created_utc": _iso(datetime.fromtimestamp(d.get("created_utc", _now().timestamp()), tz=timezone.utc)),
                    "score": int(d.get("score", 0) or 0),
                    "num_comments": int(d.get("num_comments", 0) or 0),
                    "permalink": f"https://www.reddit.com{d.get('permalink','')}",
                    "author": d.get("author") or None,
                })
            if not after:
                break
        return out

# -------------------
# Ingest
# -------------------

def _terms_from_title(title: str) -> List[str]:
    toks = WORD_RE.findall((title or "").lower())
    return [t for t in toks if t not in STOP and len(t) >= 2]

def _summarize(posts: List[Dict[str, Any]], window_h: int, subs: List[str], mode: str) -> Dict[str, Any]:
    # hourly counts + bursts + top terms per sub
    by_sub: Dict[str, List[Dict[str, Any]]] = {s:[] for s in subs}
    for p in posts:
        by_sub.setdefault(p["subreddit"], []).append(p)

    summary: Dict[str, Any] = {
        "generated_at": _iso(_now()),
        "mode": mode,
        "window_hours": window_h,
        "subs": subs,
        "counts": {},
        "bursts": [],
        "top_terms": [],
        "demo": False,
    }

    for sub in subs:
        sub_posts = by_sub.get(sub, [])
        # Counts + authors
        counts: Dict[datetime, int] = {}
        authors = set()
        term_tf: Dict[str, int] = {}
        for p in sub_posts:
            dt = datetime.fromisoformat(p["created_utc"].replace("Z","+00:00"))
            b = _bucket_hour(dt)
            counts[b] = counts.get(b, 0) + 1
            if p.get("author"): authors.add(p["author"])
            for t in _terms_from_title(p.get("title","")):
                term_tf[t] = term_tf.get(t, 0) + 1

        # normalize 72-hour grid
        end = _bucket_hour(_now())
        start = end - timedelta(hours=window_h-1)
        grid, vals = [], []
        cur = start
        while cur <= end:
            grid.append(cur)
            vals.append(counts.get(cur, 0))
            cur += timedelta(hours=1)

        z = _zscore(vals)
        burst_idxs = [i for i, zc in enumerate(z) if zc >= 2.0]
        summary["counts"][sub] = {"posts": sum(vals), "unique_authors": len(authors)}
        for bi in burst_idxs[-3:]:
            summary["bursts"].append({
                "subreddit": sub,
                "bucket_start": _iso(grid[bi]),
                "posts": vals[bi],
                "z": z[bi],
            })
        # top terms (overall TF per sub)
        tops = sorted(term_tf.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        for term, tf in tops:
            summary["top_terms"].append({"subreddit": sub, "term": term, "tf": tf})

    return summary

def run_ingest(
    logs_dir: Path = Path("logs"),
    models_dir: Path = Path("models"),
    artifacts_dir: Path = Path("artifacts"),
) -> Dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    demo = (os.getenv("MW_DEMO","false").lower() == "true")
    mode = (os.getenv("MW_REDDIT_MODE","rss").strip().lower())
    subs = [s.strip() for s in os.getenv("MW_REDDIT_SUBS","CryptoCurrency,Bitcoin,ethtrader,Solana").split(",") if s.strip()]
    sort = os.getenv("MW_REDDIT_SORT","new")
    lookback_h = int(os.getenv("MW_REDDIT_LOOKBACK_H","72") or "72")

    client = RedditLite(mode=mode)
    now = _now()
    window_start = now - timedelta(hours=lookback_h)

    kept: List[Dict[str, Any]] = []

    if demo:
        random.seed(6068)
        for sub in subs:
            for i in range(60):
                t = window_start + timedelta(hours=i)
                kept.append({
                    "ts_ingested_utc": _iso(now),
                    "origin": "reddit",
                    "subreddit": sub,
                    "post_id": f"demo_{sub}_{i}",
                    "title": f"{sub} demo post #{i}",
                    "created_utc": _iso(t),
                    "permalink": f"/r/{sub}/comments/demo_{i}",
                    "mode": "demo",
                    "fields": {},
                    "demo": True,
                    "source": "reddit",
                })
    else:
        for sub in subs:
            try:
                posts: List[Dict[str, Any]] = []
                if mode == "api" and client._ensure_token():
                    posts = client.fetch_api_listing(sub)
                else:
                    # fallback to RSS
                    posts = client.fetch_rss(sub, sort=sort)

                for p in posts:
                    cdt = datetime.fromisoformat(p["created_utc"].replace("Z","+00:00"))
                    if cdt < window_start:  # only within lookback
                        continue
                    row = {
                        "ts_ingested_utc": _iso(now),
                        "origin": "reddit",
                        "subreddit": sub,
                        "post_id": p.get("id") or "",
                        "title": p.get("title",""),
                        "created_utc": _iso(cdt),
                        "permalink": p.get("permalink",""),
                        "mode": "api" if p.get("score") is not None else "rss",
                        "fields": {
                            **({"score": p.get("score")} if p.get("score") is not None else {}),
                            **({"num_comments": p.get("num_comments")} if p.get("num_comments") is not None else {}),
                        },
                        "author": p.get("author"),
                        "demo": False,
                        "source": "reddit",
                    }
                    kept.append(row)
                _bounded_sleep()
            except Exception:
                # soft-fail per sub
                continue

    # append log
    log_path = logs_dir / "social_reddit.jsonl"
    for row in kept:
        _append_jsonl(log_path, row)

    # summary + plots
    summary = _summarize(kept, lookback_h, subs, "demo" if demo else client.mode)
    (models_dir / "social_reddit_context.json").write_text(json.dumps(summary, ensure_ascii=False))

    # plots (hourly counts + bursts highlight)
    # We re-create hourly grid using summary
    end = _bucket_hour(now)
    start = end - timedelta(hours=lookback_h-1)
    grid = []
    cur = start
    while cur <= end:
        grid.append(cur)
        cur += timedelta(hours=1)

    counts_by_sub: Dict[str, List[int]] = {}
    bursts_by_sub_bucket: Dict[str, set] = {s:set() for s in summary["subs"]}
    for sub in summary["subs"]:
        counts_by_sub[sub] = [0]*len(grid)
    for sub, cinfo in summary.get("counts", {}).items():
        pass  # counts summary is aggregate only, so we reconstruct from log rows

    # reconstruct from kept rows
    by_sub_rows: Dict[str, Dict[datetime,int]] = {s:{} for s in summary["subs"]}
    for row in kept:
        if row["subreddit"] not in by_sub_rows:
            continue
        dt = _bucket_hour(datetime.fromisoformat(row["created_utc"].replace("Z","+00:00")))
        d = by_sub_rows[row["subreddit"]]
        d[dt] = d.get(dt, 0) + 1
    for sub in summary["subs"]:
        d = by_sub_rows[sub]
        counts_by_sub[sub] = [d.get(t,0) for t in grid]

    for b in summary.get("bursts", []):
        bursts_by_sub_bucket.setdefault(b["subreddit"], set()).add(b["bucket_start"])

    # draw
    for sub in summary["subs"]:
        xs = [t.strftime("%m-%d %H:%M") for t in grid]
        ys = counts_by_sub[sub]

        # activity
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{sub} hourly posts ({lookback_h}h)")
        plt.tight_layout()
        (artifacts_dir / f"reddit_activity_{sub}.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(artifacts_dir / f"reddit_activity_{sub}.png", dpi=120)
        plt.close()

        # bursts overlay (shade top 3 z buckets already in summary)
        plt.figure()
        plt.plot(xs, ys, marker="o")
        # shaded bands
        for i, t in enumerate(grid):
            if _iso(t) in bursts_by_sub_bucket.get(sub, set()):
                plt.axvspan(max(i-0.5,0), min(i+0.5, len(xs)-1), alpha=0.2)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{sub} bursts ({lookback_h}h)")
        plt.tight_layout()
        plt.savefig(artifacts_dir / f"reddit_bursts_{sub}.png", dpi=120)
        plt.close()

    return summary

if __name__ == "__main__":
    run_ingest()
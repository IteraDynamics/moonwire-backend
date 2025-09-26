# -*- coding: utf-8 -*-
"""
Reddit Lite Ingest

Provides:
  - IngestPaths dataclass (tests construct this)
  - run_ingest(paths: IngestPaths) -> Dict
  - __main__ entrypoint (runs with repo default folders)

Outputs:
  - logs/social_reddit.jsonl
  - models/social_reddit_context.json
  - artifacts/reddit_activity_<sub>.png
  - artifacts/reddit_bursts_<sub>.png
"""

from __future__ import annotations
import os, re, json, math, random
from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import reddit_client_lite as rc


ISO = "%Y-%m-%dT%H:%M:%SZ"
def _iso(dt: datetime) -> str: return dt.strftime(ISO)
def _now() -> datetime: return datetime.now(timezone.utc).replace(second=0, microsecond=0)
def _bucket_hour(dt: datetime) -> datetime: return dt.replace(minute=0, second=0, microsecond=0)

STOP = {"the","a","an","and","or","to","of","in","on","for","with","by","as","at","is","are","be","from","this","that","it","its","you","your","we","our","they","their","i","me","my","was","were","will","has","have","had"}
TOKEN = re.compile(r"[a-z0-9]+")

@dataclass
class IngestPaths:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _terms_from_title(title: str) -> List[str]:
    toks = TOKEN.findall((title or "").lower())
    return [t for t in toks if t not in STOP and len(t) >= 2]


def _z(vals: List[float]) -> List[float]:
    if not vals:
        return []
    mu = sum(vals)/len(vals)
    var = sum((x-mu)**2 for x in vals)/len(vals)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return [0.0]*len(vals)
    return [(x-mu)/sd for x in vals]


def _summarize(posts: List[Dict[str, Any]], window_h: int, subs: List[str], mode: str) -> Dict[str, Any]:
    by_sub: Dict[str, List[Dict[str, Any]]] = {s: [] for s in subs}
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
        "demo": (mode == "demo"),
    }

    for sub in subs:
        sub_posts = by_sub.get(sub, [])
        # counts + terms
        counts: Dict[datetime, int] = {}
        authors = set()
        tf: Dict[str, int] = {}
        for p in sub_posts:
            dt = datetime.fromisoformat(p["created_utc"].replace("Z","+00:00"))
            b = _bucket_hour(dt)
            counts[b] = counts.get(b, 0) + 1
            if p.get("author"):
                authors.add(p["author"])
            for t in _terms_from_title(p.get("title","")):
                tf[t] = tf.get(t, 0) + 1

        # 72h grid
        end = _bucket_hour(_now())
        start = end - timedelta(hours=window_h-1)
        grid, vals = [], []
        cur = start
        while cur <= end:
            grid.append(cur)
            vals.append(counts.get(cur, 0))
            cur += timedelta(hours=1)

        summary["counts"][sub] = {"posts": sum(vals), "unique_authors": len(authors)}
        z = _z(vals)
        for i, zv in enumerate(z):
            if zv >= 2.0:
                summary["bursts"].append({
                    "subreddit": sub,
                    "bucket_start": _iso(grid[i]),
                    "posts": vals[i],
                    "z": zv,
                })

        top = sorted(tf.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        for term, v in top:
            summary["top_terms"].append({"subreddit": sub, "term": term, "tf": v})

    return summary


def run_ingest(
    logs_dir: Path | None = None,
    models_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    paths: IngestPaths | None = None,
) -> Dict[str, Any]:
    """
    Tests may pass an IngestPaths; CI runs with default folders.
    """
    if paths is not None:
        logs_dir, models_dir, artifacts_dir = paths.logs_dir, paths.models_dir, paths.artifacts_dir

    logs_dir = Path(logs_dir or "logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir or "models"); models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(artifacts_dir or "artifacts"); artifacts_dir.mkdir(parents=True, exist_ok=True)

    demo = (os.getenv("MW_DEMO","false").lower() == "true")
    mode_env = (os.getenv("MW_REDDIT_MODE","rss").strip().lower())
    subs = [s.strip() for s in os.getenv("MW_REDDIT_SUBS","CryptoCurrency,Bitcoin,ethtrader,Solana").split(",") if s.strip()]
    sort = os.getenv("MW_REDDIT_SORT","new")
    lookback_h = int(os.getenv("MW_REDDIT_LOOKBACK_H","72") or "72")

    client = rc.RedditLite(mode=mode_env)
    now = _now()
    window_start = now - timedelta(hours=lookback_h)

    kept: List[Dict[str, Any]] = []

    if demo:
        random.seed(6068)
        for sub in subs:
            for i in range(36):
                t = window_start + timedelta(hours=i*2)
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
                if mode_env == "api" and client._ensure_token():
                    posts = client.fetch_api_listing(sub)
                else:
                    posts = client.fetch_rss(sub, sort=sort)

                for p in posts:
                    cdt = datetime.fromisoformat(p["created_utc"].replace("Z","+00:00"))
                    if cdt < window_start:
                        continue
                    kept.append({
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
                    })
            except Exception:
                # per-sub soft fail
                continue

    # append to log
    log_path = logs_dir / "social_reddit.jsonl"
    for row in kept:
        _append_jsonl(log_path, row)

    # write summary
    summary = _summarize(kept, lookback_h, subs, "demo" if demo else client.mode)
    (models_dir / "social_reddit_context.json").write_text(json.dumps(summary, ensure_ascii=False))

    # plots
    end = _bucket_hour(now)
    start = end - timedelta(hours=lookback_h-1)
    grid = []
    cur = start
    while cur <= end:
        grid.append(cur)
        cur += timedelta(hours=1)

    # reconstruct counts from kept rows
    per_sub_counts: Dict[str, Dict[datetime, int]] = {s: {} for s in subs}
    for r in kept:
        sub = r["subreddit"]
        dt = _bucket_hour(datetime.fromisoformat(r["created_utc"].replace("Z","+00:00")))
        d = per_sub_counts[sub]
        d[dt] = d.get(dt, 0) + 1

    burst_buckets = {(b["subreddit"], b["bucket_start"]) for b in summary.get("bursts", [])}

    for sub in subs:
        xs = [t.strftime("%m-%d %H:%M") for t in grid]
        ys = [per_sub_counts[sub].get(t, 0) for t in grid]

        # activity
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{sub} hourly posts ({lookback_h}h)")
        plt.tight_layout()
        plt.savefig(Path(artifacts_dir) / f"reddit_activity_{sub}.png", dpi=120)
        plt.close()

        # bursts (shade)
        plt.figure()
        plt.plot(xs, ys, marker="o")
        for i, t in enumerate(grid):
            if (sub, _iso(t)) in burst_buckets:
                plt.axvspan(max(i-0.5, 0), min(i+0.5, len(xs)-1), alpha=0.2)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{sub} bursts ({lookback_h}h)")
        plt.tight_layout()
        plt.savefig(Path(artifacts_dir) / f"reddit_bursts_{sub}.png", dpi=120)
        plt.close()

    return summary


if __name__ == "__main__":
    run_ingest()
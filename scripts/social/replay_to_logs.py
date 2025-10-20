# scripts/social/replay_to_logs.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Dict, Any, Iterable

ISO = "%Y-%m-%dT%H:%M:%SZ"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MW_SOCIAL_DATA_DIR", ROOT / "data" / "social"))
LOGS_DIR = Path(os.getenv("MW_LOGS_DIR", ROOT / "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_LOG = LOGS_DIR / "social_reddit.jsonl"
TWITTER_LOG = LOGS_DIR / "social_twitter.jsonl"

def _iter_jsonl(paths: Iterable[Path]):
    for p in sorted(paths):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _normalize_reddit(r: Dict[str, Any]) -> Dict[str, Any] | None:
    ciso = r.get("created_utc") or r.get("createdAt") or r.get("created_at")
    if not ciso:
        return None
    return {
        "ts_ingested_utc": r.get("ts_ingested_utc") or r.get("ingested_utc") or ciso,
        "origin": "reddit",
        "subreddit": r.get("subreddit") or r.get("sub") or "",
        "post_id": r.get("post_id") or r.get("id") or r.get("name") or "",
        "title": r.get("title") or r.get("text") or "",
        "created_utc": ciso,
        "permalink": r.get("permalink") or "",
        "author": r.get("author") or None,
        "mode": r.get("mode") or "replay",
        "fields": dict(r.get("fields") or {}),
        "demo": False,
        "source": r.get("source") or "reddit",
    }

def _normalize_twitter(t: Dict[str, Any]) -> Dict[str, Any] | None:
    ciso = t.get("created_utc") or t.get("created_at")
    if not ciso:
        return None
    metrics = t.get("metrics") or t.get("public_metrics") or {}
    return {
        "ts_ingested_utc": t.get("ts_ingested_utc") or ciso,
        "origin": "twitter",
        "tweet_id": t.get("tweet_id") or t.get("id") or "",
        "text": t.get("text") or "",
        "created_utc": ciso,
        "author_id": str(t.get("author_id") or t.get("user_id") or ""),
        "metrics": {
            "retweets": int(metrics.get("retweet_count") or metrics.get("retweets") or 0),
            "likes": int(metrics.get("like_count") or metrics.get("likes") or 0),
        },
        "lang": t.get("lang") or "en",
        "demo": False,
        "source": t.get("source") or "twitter_replay",
    }

def main():
    ap = argparse.ArgumentParser(description="Replay historical social JSONL into logs/")
    ap.add_argument("--clear", action="store_true", help="Truncate logs before appending")
    ap.add_argument("--only", choices=["reddit", "twitter"], default=None, help="Restrict to one origin")
    args = ap.parse_args()

    if args.clear:
        if REDDIT_LOG.exists(): REDDIT_LOG.unlink()
        if TWITTER_LOG.exists(): TWITTER_LOG.unlink()

    # Reddit
    if args.only in (None, "reddit"):
        reddit_files = list((DATA_DIR / "reddit").glob("*.jsonl"))
        seen_ids = set()
        out = []
        for row in _iter_jsonl(reddit_files):
            n = _normalize_reddit(row)
            if not n:
                continue
            key = ("reddit", n.get("post_id"))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            out.append(n)
        if out:
            _write_jsonl(REDDIT_LOG, out)

    # Twitter
    if args.only in (None, "twitter"):
        tw_files = list((DATA_DIR / "twitter").glob("*.jsonl"))
        seen_ids = set()
        out = []
        for row in _iter_jsonl(tw_files):
            n = _normalize_twitter(row)
            if not n:
                continue
            key = ("twitter", n.get("tweet_id"))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            out.append(n)
        if out:
            _write_jsonl(TWITTER_LOG, out)

if __name__ == "__main__":
    main()
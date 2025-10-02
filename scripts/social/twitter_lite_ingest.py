# scripts/social/twitter_lite_ingest.py
from __future__ import annotations

import os
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless CI
import matplotlib.pyplot as plt

from .twitter_client_lite import TwitterLiteConfig, TwitterClientLite, _iso


STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","is","are","was","were","be","with",
    "this","that","it","as","by","at","from","you","your","i","we","they","he","she","them",
    "rt","amp","http","https"
}


@dataclass
class IngestPaths:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path

    def ensure(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _bucket_hour(ts_iso: str) -> str:
    dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return _iso(dt)


def _tokenize(text: str) -> List[str]:
    # simple 1-gram tokenizer
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9_#]+", " ", text)
    toks = [t.strip("#") for t in text.split() if t and t not in STOPWORDS]
    return toks


def _z_scores(xs: List[int]) -> List[float]:
    if not xs:
        return []
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0.0:
        return [0.0] * len(xs)
    return [(x - mu) / sd for x in xs]


def _make_activity_plot(hour_buckets: List[str], counts: List[int], out_path: Path) -> None:
    plt.figure()
    plt.plot(hour_buckets, counts, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _make_bursts_plot(hour_buckets: List[str], counts: List[int], bursts_idx: List[int], out_path: Path) -> None:
    plt.figure()
    plt.plot(hour_buckets, counts, marker="o")
    # mark bursts with triangle markers
    yvals = [counts[i] for i in bursts_idx]
    xvals = [hour_buckets[i] for i in bursts_idx]
    if xvals:
        plt.scatter(xvals, yvals, marker="^")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def run_ingest(paths: IngestPaths | None = None) -> Dict[str, Any]:
    """
    Orchestrates Twitter Lite ingest (API or demo), writes:
      - logs/social_twitter.jsonl
      - models/social_twitter_context.json
      - artifacts/twitter_activity.png
      - artifacts/twitter_bursts.png
    Returns the context JSON as a dict.
    """
    cfg = TwitterLiteConfig.from_env()
    if paths is None:
        repo_root = Path(os.getcwd())
        paths = IngestPaths(
            logs_dir=repo_root / "logs",
            models_dir=repo_root / "models",
            artifacts_dir=repo_root / "artifacts",
        )
    paths.ensure()

    client = TwitterClientLite(cfg)
    rows = client.fetch_recent()

    # Normalize → log lines
    now_iso = _iso(_now_utc())
    log_lines: List[Dict[str, Any]] = []
    for r in rows:
        log_lines.append(
            {
                "ts_ingested_utc": now_iso,
                "origin": "twitter",
                "tweet_id": r.get("tweet_id"),
                "text": r.get("text") or "",
                "created_utc": r.get("created_utc"),
                "author_id": r.get("author_id") or "",
                "metrics": dict(r.get("metrics") or {}),
                "demo": cfg.demo,
                "source": r.get("source") or ("twitter_api" if cfg.mode == "api" else "demo"),
            }
        )
    _append_jsonl(paths.logs_dir / "social_twitter.jsonl", log_lines)

    # Window filter + buckets
    lookback_h = cfg.lookback_h
    window_start = _now_utc() - timedelta(hours=lookback_h)
    kept = [r for r in log_lines if datetime.fromisoformat(r["created_utc"].replace("Z","+00:00")) >= window_start]

    # Hourly counts
    buckets: Dict[str, int] = {}
    authors: set[str] = set()
    for r in kept:
        b = _bucket_hour(r["created_utc"])
        buckets[b] = buckets.get(b, 0) + 1
        aid = str(r.get("author_id") or "")
        if aid:
            authors.add(aid)

    hour_keys = sorted(buckets.keys())
    counts = [buckets[k] for k in hour_keys]

    # Burst detection
    zs = _z_scores(counts) if counts else []
    burst_idx = [i for i, z in enumerate(zs) if z >= 2.0]
    # top-3 bursts by z
    burst_idx_sorted = sorted(burst_idx, key=lambda i: zs[i], reverse=True)[:3]
    bursts = [
        {"bucket_start": hour_keys[i], "tweets": counts[i], "z": float(zs[i]), "keyword": None}
        for i in burst_idx_sorted
    ]

    # Top terms
    tf: Dict[str, int] = {}
    for r in kept:
        for tok in _tokenize(r.get("text", "")):
            tf[tok] = tf.get(tok, 0) + 1
    top_terms = [
        {"term": t, "tf": c} for (t, c) in sorted(tf.items(), key=lambda kv: kv[1], reverse=True)[:10]
    ]

    ctx_json = {
        "generated_at": now_iso,
        "window_hours": lookback_h,
        "keywords": list(cfg.keywords),
        "counts": {"total_tweets": len(kept), "unique_authors": len(authors)},
        "bursts": bursts,
        "top_terms": top_terms,
        "demo": cfg.demo,
    }

    out_json = paths.models_dir / "social_twitter_context.json"
    out_json.write_text(json.dumps(ctx_json, ensure_ascii=False, indent=2))

    # Plots
    _make_activity_plot(hour_keys, counts, paths.artifacts_dir / "twitter_activity.png")
    _make_bursts_plot(hour_keys, counts, burst_idx_sorted, paths.artifacts_dir / "twitter_bursts.png")

    return ctx_json
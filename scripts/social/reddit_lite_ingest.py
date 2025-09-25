# scripts/social/reddit_lite_ingest.py
from __future__ import annotations

import os
import re
import json
import math
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .reddit_client_lite import LiteConfig, RedditLiteClient, _iso, _now_utc

STOPWORDS = {
    "the","a","and","or","but","for","to","of","in","on","is","it","its","at","by","with",
    "this","that","be","as","are","from","an","was","were","if","else","we","you","i",
    "rt","amp","vs","via","about","has","have","had","will","just","they","he","she",
    "can","could","should","would","did","do","does","than","then","over","under","into",
    "out","up","down","not","no","yes",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t not in STOPWORDS and not t.isdigit() and len(t) > 1]


@dataclass
class IngestPaths:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path


def _load_env_paths() -> IngestPaths:
    root = Path(os.getcwd())
    logs = Path(os.getenv("LOGS_DIR", root / "logs"))
    models = Path(os.getenv("MODELS_DIR", root / "models"))
    arts = Path(os.getenv("ARTIFACTS_DIR", root / "artifacts"))
    logs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    arts.mkdir(parents=True, exist_ok=True)
    return IngestPaths(logs, models, arts)


def _within_lookback(dt: datetime, lookback_h: int) -> bool:
    return dt >= (_now_utc() - timedelta(hours=lookback_h))


def _z_scores(vals: List[int]) -> List[float]:
    if not vals:
        return []
    mu = sum(vals) / len(vals)
    var = sum((v - mu)**2 for v in vals) / max(1, len(vals)-1)
    sd = math.sqrt(var)
    if sd <= 1e-9:
        return [0.0] * len(vals)
    return [(v - mu) / sd for v in vals]


def _hour_floor(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def _plot_counts_with_bursts(hours: List[datetime], counts: List[int], burst_idx: List[int], out_path: Path, title: str) -> None:
    plt.figure()
    xs = hours
    ys = counts
    plt.plot(xs, ys)  # no color/style settings per guidelines
    # overlay shaded bursts
    for i in burst_idx:
        # shade the hour block
        start = xs[i]
        end = start + timedelta(hours=1)
        plt.axvspan(start, end, alpha=0.15)
    plt.title(title)
    plt.xlabel("UTC Hour")
    plt.ylabel("Posts")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_ingest(paths: Optional[IngestPaths] = None, cfg: Optional[LiteConfig] = None) -> Dict[str, Any]:
    """
    Fetch recent Reddit posts (RSS default; API optional), normalize, log, aggregate,
    plot activity + bursts, and emit models/social_reddit_context.json
    """
    paths = paths or _load_env_paths()
    cfg = cfg or LiteConfig.from_env()

    client = RedditLiteClient(cfg)

    # append-only log
    log_path = paths.logs_dir / "social_reddit.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # gather posts per subreddit
    lookback_h = cfg.lookback_h
    posts_by_sub: Dict[str, List[Dict[str, Any]]] = {}

    for idx, sub in enumerate(cfg.subs):
        posts = client.list_recent_posts(sub)
        kept: List[Dict[str, Any]] = []
        for p in posts:
            dt = p.get("created_utc")
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                except Exception:
                    continue
            if not isinstance(dt, datetime):
                continue
            if not _within_lookback(dt, lookback_h):
                continue
            kept.append(p)

            # write to append-only log (one per kept post)
            line = {
                "ts_ingested_utc": _iso(_now_utc()),
                "origin": "reddit",
                "subreddit": sub,
                "post_id": p.get("id"),
                "title": p.get("title") or "",
                "created_utc": _iso(dt),
                "permalink": p.get("permalink") or "",
                "mode": client.cfg.mode if not client.cfg.demo else "demo",
                "fields": p.get("fields") or {},
                "demo": bool(client.cfg.demo),
                "source": "reddit",
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")

        posts_by_sub[sub] = kept

        # politeness between subs
        if not client.cfg.demo:
            time_between = 1.25
            try:
                import time
                time.sleep(time_between)
            except Exception:
                pass

    # aggregate per hour
    window_hours = lookback_h
    now = _now_utc().replace(minute=0, second=0, microsecond=0)
    hours = [now - timedelta(hours=h) for h in range(window_hours)][::-1]  # ascending

    series = {}
    counts_summary = {}
    bursts_summary: List[Dict[str, Any]] = []
    top_terms_global: Counter[str] = Counter()

    for sub, plist in posts_by_sub.items():
        # bucket to hour
        count_by_hour: Dict[datetime, int] = {h: 0 for h in hours}
        authors: set = set()
        terms: Counter[str] = Counter()

        for p in plist:
            dt: datetime = p["created_utc"] if isinstance(p["created_utc"], datetime) else datetime.fromisoformat(str(p["created_utc"]).replace("Z", "+00:00"))
            hf = _hour_floor(dt)
            if hf in count_by_hour:
                count_by_hour[hf] += 1
            author = (p.get("author") or "").strip()
            if author:
                authors.add(author)
            # terms from title (and selftext if present in fields -> ignored in lite)
            terms.update(_tokenize(p.get("title") or ""))

        hours_list = hours
        counts = [count_by_hour[h] for h in hours_list]
        z = _z_scores(counts)
        burst_idx = [i for i, zval in enumerate(z) if zval >= 2.0]
        # plot activity
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        _plot_counts_with_bursts(
            hours_list,
            counts,
            burst_idx,
            paths.artifacts_dir / f"reddit_activity_{sub}.png",
            f"/r/{sub} activity (posts/hour)",
        )
        _plot_counts_with_bursts(
            hours_list,
            counts,
            burst_idx,
            paths.artifacts_dir / f"reddit_bursts_{sub}.png",
            f"/r/{sub} bursts (z≥2 highlighted)",
        )

        # summaries
        counts_summary[sub] = {
            "posts": int(sum(counts)),
            "unique_authors": int(len(authors)) if not cfg.demo else int(max(1, len(authors)))  # demo keeps count-ish
        }
        for i in burst_idx[:3]:  # top 3 highlights
            bursts_summary.append({
                "subreddit": sub,
                "bucket_start": _iso(hours_list[i]),
                "posts": counts[i],
                "z": round(float(z[i]), 3),
            })

        # top terms (per sub) – also aggregate globally
        top_terms = [{"term": t, "tf": int(c)} for t, c in Counter(terms).most_common(10)]
        series[sub] = {
            "hourly_counts": [{"t": _iso(h), "posts": count_by_hour[h]} for h in hours_list],
            "top_terms": top_terms,
        }
        top_terms_global.update(terms)

    top_terms_all = [{"term": t, "tf": int(c)} for t, c in top_terms_global.most_common(10)]

    artifact = {
        "generated_at": _iso(_now_utc()),
        "mode": "demo" if cfg.demo else cfg.mode,
        "window_hours": window_hours,
        "subs": cfg.subs,
        "counts": counts_summary,
        "bursts": bursts_summary,
        "top_terms": top_terms_all,
        "series": series,
        "demo": bool(cfg.demo),
    }

    out_json = paths.models_dir / "social_reddit_context.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(artifact, ensure_ascii=False))

    return artifact


if __name__ == "__main__":
    run_ingest()
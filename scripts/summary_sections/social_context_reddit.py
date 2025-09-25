# -*- coding: utf-8 -*-
"""
Social Context — Reddit (Lite)

This summary section reads the Reddit ingest artifact produced by
`scripts/social/reddit_lite_ingest.py` and renders a compact markdown block
for the CI summary. When the artifact is not present and we're in demo mode,
it synthesizes a deterministic demo artifact so the section is never empty.

Inputs:
  - models/social_reddit_context.json (if present)
  - Environment:
      MW_REDDIT_SUBS            (default: "CryptoCurrency,Bitcoin,ethtrader,Solana")
      MW_REDDIT_LOOKBACK_H      (default: "72")
      MW_DEMO                   ("true"/"false")

Outputs:
  - Appended markdown to the provided `md` list.
  - (In demo mode when missing) writes models/social_reddit_context.json

The markdown shape:

🗞️ Social Context — Reddit (72h) (rss|api|demo)
CryptoCurrency → 182 posts | 3 bursts | top: etf, sec, spot
Bitcoin        → 95 posts  | 1 burst  | top: halving, miner
...

We keep implementation intentionally lightweight here; heavy lifting (fetch,
counting, bursts, terms, plotting) belongs in the ingest script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta, timezone
import math
import random
import string

# SummaryContext protocol (duck-typed in tests)
# ctx.logs_dir, ctx.models_dir, ctx.artifacts_dir, ctx.is_demo


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_list(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # de-dup, preserve order
    seen = set()
    out: List[str] = []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def _synth_terms(seed: int, n: int = 10) -> List[Tuple[str, int]]:
    """
    Deterministic fake top terms list given a seed.
    """
    rnd = random.Random(seed)
    vocab = [
        "etf", "sec", "spot", "halving", "miner",
        "l2", "staking", "tps", "outage", "airdrop",
        "validator", "merge", "eip", "rollup", "ml",
        "macro", "fed", "rate", "inflation", "flows",
    ]
    terms = rnd.sample(vocab, k=min(n, len(vocab)))
    return [(t, rnd.randint(5, 40)) for t in terms]


def _synth_counts(seed: int, hours: int) -> Tuple[List[int], List[int]]:
    """
    Return (hourly_counts, bursts_mask) for the past `hours`.
    Simple deterministic wave + two spikes to produce bursts.
    """
    rnd = random.Random(seed)
    counts: List[int] = []
    for i in range(hours):
        base = 3 + int(2 * math.sin(i / 5.0))  # small wave
        jitter = rnd.randint(0, 3)
        counts.append(max(0, base + jitter))
    # add 1–2 spikes to ensure some bursts
    for k in range(rnd.randint(1, 2)):
        idx = hours - 1 - rnd.randint(0, min(6, hours - 1))
        counts[idx] += rnd.randint(8, 14)
    # naive z-score burst mask
    mu = sum(counts) / float(len(counts))
    var = sum((c - mu) ** 2 for c in counts) / float(len(counts))
    sd = math.sqrt(var) or 1.0
    bursts = [1 if (c - mu) / sd >= 2.0 else 0 for c in counts]
    return counts, bursts


def _ensure_demo_artifact(models_dir: Path, subs: List[str], hours: int) -> Dict[str, Any]:
    """
    Create a deterministic demo artifact if missing.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=hours)
    # deterministic seed based on subs + hours so CI is stable
    seed = hash(("|".join([s.lower() for s in subs]), hours)) & 0xFFFFFFFF
    rnd = random.Random(seed)

    counts: Dict[str, Dict[str, int]] = {}
    bursts_list: List[Dict[str, Any]] = []
    for idx, sub in enumerate(subs):
        sub_seed = seed + idx * 97
        hourly, burst_mask = _synth_counts(sub_seed, hours)
        total_posts = int(sum(hourly))
        uniq_authors = int(max(1, int(total_posts * 0.85)))
        counts[sub] = {"posts": total_posts, "unique_authors": uniq_authors}

        # collect top 3 burst buckets by z-score proxy (just the last hours where burst=1)
        top_buckets: List[int] = [i for i, b in enumerate(burst_mask) if b == 1]
        if len(top_buckets) > 3:
            # keep last three bursts
            top_buckets = top_buckets[-3:]
        for i in top_buckets:
            bucket_start = start + timedelta(hours=i)
            bursts_list.append({
                "subreddit": sub,
                "bucket_start": _iso(bucket_start),
                "posts": hourly[i],
                "z": 2.1,  # indicative
            })

    # top_terms: pick a few
    # build per-sub terms then consolidate simple frequency map
    term_map: Dict[str, int] = {}
    for idx, sub in enumerate(subs):
        for term, tf in _synth_terms(seed + idx * 131, n=6):
            term_map[term] = term_map.get(term, 0) + tf
    top_terms = [{"term": t, "tf": term_map[t]} for t in sorted(term_map, key=term_map.get, reverse=True)[:10]]

    artifact = {
        "generated_at": _iso(now),
        "mode": "demo",
        "window_hours": hours,
        "subs": subs,
        "counts": counts,
        "bursts": bursts_list,
        "top_terms": top_terms,
        "demo": True,
    }
    (models_dir / "social_reddit_context.json").write_text(json.dumps(artifact, ensure_ascii=False))
    return artifact


def _load_or_demo(models_dir: Path, subs: List[str], hours: int, is_demo: bool) -> Dict[str, Any] | None:
    jpath = models_dir / "social_reddit_context.json"
    if jpath.exists():
        try:
            return json.loads(jpath.read_text())
        except Exception:
            # fall through to demo if allowed
            pass
    if is_demo or _env_bool("MW_DEMO", False):
        models_dir.mkdir(parents=True, exist_ok=True)
        return _ensure_demo_artifact(models_dir, subs, hours)
    return None


def _fmt_line(sub: str, c: Dict[str, int], bursts_count: int, top_terms: List[str]) -> str:
    # left-align subreddit to 12 chars to make the summary scan-friendly
    left = f"{sub:<12}"
    terms = ", ".join(top_terms[:3]) if top_terms else "n/a"
    return f"{left} → {c.get('posts', 0)} posts | {bursts_count} bursts | top: {terms}"


def append(md: List[str], ctx) -> None:
    """
    Render the Social Context — Reddit section.
    Generates a demo artifact if missing and MW_DEMO=true.
    """
    models_dir = Path(ctx.models_dir)
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", models_dir.parent / "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    subs = _env_list("MW_REDDIT_SUBS", "CryptoCurrency,Bitcoin,ethtrader,Solana")
    hours = _env_int("MW_REDDIT_LOOKBACK_H", 72)

    data = _load_or_demo(models_dir, subs, hours, getattr(ctx, "is_demo", False))
    if not data:
        md.append("\n> ⚠️ Social Context — Reddit: no data (ingest step did not run).\n")
        return

    mode = data.get("mode") or ("api" if not data.get("demo") else "demo")
    counts_map: Dict[str, Dict[str, int]] = data.get("counts", {}) or {}
    bursts = data.get("bursts", []) or []
    top_terms_list = data.get("top_terms", []) or []
    # build quick per-sub burst count and top terms
    bursts_by_sub: Dict[str, int] = {}
    for b in bursts:
        s = b.get("subreddit")
        if s:
            bursts_by_sub[s] = bursts_by_sub.get(s, 0) + 1
    # a simple per-sub term pick using the global top_terms order (demo artifact),
    # real ingest can store per-sub terms later
    ordered_terms = [x.get("term", "") for x in top_terms_list]

    md.append(f"### 🗞️ Social Context — Reddit ({hours}h) ({mode})")
    any_line = False
    for sub in subs:
        c = counts_map.get(sub, {"posts": 0, "unique_authors": 0})
        bcnt = bursts_by_sub.get(sub, 0)
        line = _fmt_line(sub, c, bcnt, ordered_terms)
        md.append(line)
        any_line = True

    if not any_line:
        md.append("_no subreddit data_")

    md.append("\n_Footer: Data via Reddit RSS/API. Rate limits respected in API mode._\n")
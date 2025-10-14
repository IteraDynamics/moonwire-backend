# scripts/summary_sections/trigger_suppression_trend.py
from __future__ import annotations

import os, json, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import List, Dict, Any

# matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import SummaryContext, parse_ts, _iso, is_demo_mode

def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _bucket_start(ts: datetime, bucket_h: int) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    h = (ts.hour // bucket_h) * bucket_h
    return ts.replace(hour=h)

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _seed_demo_series(now: datetime, window_h: int, bucket_h: int) -> Dict[str, List[Dict[str, Any]]]:
    origins = ["twitter", "reddit", "rss_news"]
    buckets = []
    t = _bucket_start(now, bucket_h)
    cutoff = now - timedelta(hours=window_h)
    while t >= cutoff:
        buckets.append(t)
        t = t - timedelta(hours=bucket_h)
    buckets = list(reversed(buckets))

    series_by_origin: Dict[str, List[Dict[str, Any]]] = {}
    base = {"twitter": 0.65, "reddit": 0.82, "rss_news": 0.40}
    for o in origins:
        vals = []
        for i, bt in enumerate(buckets):
            wobble = ((i % 3) - 1) * 0.03  # -0.03, 0.00, +0.03
            y = min(0.99, max(0.0, base[o] + wobble))
            vals.append({"origin": o, "t": _iso(bt), "suppression_rate": y})
        series_by_origin[o] = vals
    return series_by_origin

def append(md: List[str], ctx: SummaryContext) -> None:
    window_h = int(os.getenv("MW_TRIGGER_SUPPRESSION_WINDOW_H", "48"))
    bucket_h = int(os.getenv("MW_TRIGGER_BUCKET_H", "3"))

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_h)

    candidates_path = (ctx.logs_dir or Path("logs")) / "candidates.jsonl"
    triggers_path = (ctx.models_dir or Path("models")) / "trigger_history.jsonl"

    cand = _load_jsonl(candidates_path)
    trig = _load_jsonl(triggers_path)

    cand_f = []
    for r in cand:
        ts = parse_ts(r.get("timestamp"))
        if ts and ts >= cutoff:
            o = r.get("origin") or "unknown"
            cand_f.append({"origin": o, "ts": ts})
    trig_f = []
    for r in trig:
        ts = parse_ts(r.get("timestamp"))
        if ts and ts >= cutoff:
            o = r.get("origin") or "unknown"
            trig_f.append({"origin": o, "ts": ts})

    buckets: List[datetime] = []
    t = _bucket_start(now, bucket_h)
    while t >= cutoff:
        buckets.append(t)
        t = t - timedelta(hours=bucket_h)
    buckets = list(reversed(buckets))

    def idx_for(ts: datetime) -> int | None:
        if ts < cutoff:
            return None
        bstart = _bucket_start(ts, bucket_h)
        try:
            return buckets.index(bstart)
        except ValueError:
            return None

    per_origin_cand: Dict[str, List[int]] = defaultdict(lambda: [0] * len(buckets))
    per_origin_trig: Dict[str, List[int]] = defaultdict(lambda: [0] * len(buckets))

    for r in cand_f:
        i = idx_for(r["ts"])
        if i is not None:
            per_origin_cand[r["origin"]][i] += 1
    for r in trig_f:
        i = idx_for(r["ts"])
        if i is not None:
            per_origin_trig[r["origin"]][i] += 1

    series: Dict[str, List[Dict[str, Any]]] = {}
    origins = sorted(set(list(per_origin_cand.keys()) + list(per_origin_trig.keys())))
    for o in origins:
        if o == "unknown":
            continue
        cand_counts = per_origin_cand.get(o, [0] * len(buckets))
        trig_counts = per_origin_trig.get(o, [0] * len(buckets))
        pts = []
        for i, bstart in enumerate(buckets):
            c = cand_counts[i]
            tcount = trig_counts[i]
            suppressed = max(0, c - tcount)
            rate = _safe_div(suppressed, max(c, 1))
            pts.append({
                "origin": o,
                "t": _iso(bstart),
                "suppression_rate": rate,
                "candidates": c,
                "triggers": tcount
            })
        series[o] = pts

    no_real_points = all(
        (sum(p["candidates"] for p in v) == 0 and sum(p["triggers"] for p in v) == 0)
        for v in series.values()
    ) if series else True
    is_demo = ctx.is_demo or is_demo_mode()
    if (not series or no_real_points) and is_demo:
        series = _seed_demo_series(now, window_h, bucket_h)
        demo_tag = " (demo)"
    else:
        demo_tag = ""

    # ---- artifacts dir: prefer <models_dir>/../artifacts (works in tests) ----
    artifacts_dir = (ctx.models_dir.parent / "artifacts") if getattr(ctx, "models_dir", None) else Path("artifacts")
    _ensure_dir(artifacts_dir)
    img_path = artifacts_dir / f"trigger_suppression_trend_{window_h}h.png"

    xs = [b for b in buckets]
    fig = plt.figure(figsize=(8, 3.5), dpi=140)
    ax = plt.gca()

    # Guides
    for y, style in [(0.5, "--"), (0.85, "--")]:
        ax.axhline(y, linestyle=style, linewidth=1)

    for o in sorted(series.keys()):
        pts = series[o]
        by_t = {parse_ts(p["t"]): p["suppression_rate"] for p in pts if parse_ts(p["t"]) is not None}
        ys = [by_t.get(b, math.nan) for b in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=o)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Suppression rate")
    ax.set_title(f"Trigger Suppression Trend ({window_h}h){demo_tag}")
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)

    md.append(f"### 📉 Trigger Suppression Trend ({window_h}h){demo_tag}")
    md.append(f"- saved: {img_path}")

# scripts/summary_sections/score_distribution.py
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import os
import random

def render(md: List[str]) -> None:
    """
    Renders:
      ### 📊 Score Distribution Snapshot
      - 24h stats (n, mean, median, std, min, max, >thr%)
      - 72h histogram (0.0–0.1, …, 0.9–1.0)
    Reads models/score_history.jsonl (append-only; optional).
    DEMO_MODE seeds ~10 plausible rows if the log is empty.
    """
    try:
        from src.paths import MODELS_DIR
    except Exception:
        md.append("\n### 📊 Score Distribution Snapshot")
        md.append("_paths not available_")
        return

    md.append("\n### 📊 Score Distribution Snapshot")

    log_path = MODELS_DIR / "score_history.jsonl"
    now = datetime.now(timezone.utc)
    cutoff_24 = now - timedelta(hours=24)
    cutoff_72 = now - timedelta(hours=72)

    # Threshold (global display): env override → default 0.5
    try:
        thr_env = os.getenv("TL_DECISION_THRESHOLD")
        threshold = float(thr_env) if thr_env is not None else 0.5
    except Exception:
        threshold = 0.5

    def _parse_ts(v) -> datetime | None:
        if v is None:
            return None
        try:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        except Exception:
            pass
        try:
            s = str(v)
            s = s[:-1] + "+00:00" if s.endswith("Z") else s
            dt = datetime.fromisoformat(s)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _load_jsonl(p: Path) -> list[dict]:
        if not p.exists():
            return []
        out: list[dict] = []
        try:
            for ln in p.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
        except Exception:
            return []
        return out

    rows = _load_jsonl(log_path)

    # DEMO: seed plausible scores in-memory (do NOT write file)
    demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
    if not rows and demo_mode:
        origins = ["reddit", "twitter", "rss_news"]
        versions = ["v0.5.2", "v0.5.1"]
        n = 10
        # mixture: mild bimodal around ~0.1 and ~0.6
        for i in range(n):
            t = now - timedelta(minutes=5 * i)
            if i % 3 == 0:
                s = max(0.0, min(1.0, random.gauss(0.60, 0.12)))
            else:
                s = max(0.0, min(1.0, random.gauss(0.15, 0.08)))
            rows.append({
                "timestamp": t.isoformat(),
                "origin": random.choice(origins),
                "adjusted_score": float(s),
                "model_version": random.choice(versions),
            })

    # Extract recent scores
    scores_24: list[float] = []
    scores_72: list[float] = []

    for r in rows:
        ts = _parse_ts(r.get("timestamp"))
        if ts is None:
            continue
        try:
            s = float(r.get("adjusted_score"))
        except Exception:
            continue
        if ts >= cutoff_72:
            scores_72.append(s)
            if ts >= cutoff_24:
                scores_24.append(s)

    def _safe_stats(vals: list[float]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "gt_thr_pct": 0.0}
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean = sum(vals_sorted) / n
        # median
        if n % 2 == 1:
            median = vals_sorted[n // 2]
        else:
            median = 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
        # population std (display; robust for small n)
        var = 0.0
        if n > 0:
            mu = mean
            var = sum((x - mu) ** 2 for x in vals_sorted) / n
        std = var ** 0.5
        vmin = vals_sorted[0]
        vmax = vals_sorted[-1]
        gt = sum(1 for x in vals_sorted if x > threshold)
        gt_pct = 100.0 * gt / n
        return {"n": n, "mean": mean, "median": median, "std": std, "min": vmin, "max": vmax, "gt_thr_pct": gt_pct}

    s24 = _safe_stats(scores_24)
    s72 = _safe_stats(scores_72)

    # Print 24h stats (compact)
    md.append(
        f"- **last 24h**: n={s24['n']}, mean={s24['mean']:.3f}, median={s24['median']:.3f}, "
        f"std={s24['std']:.3f}, min={s24['min']:.3f}, max={s24['max']:.3f}, "
        f">thr({threshold:.2f})={s24['gt_thr_pct']:.1f}%"
    )

    # Histogram over 72h (10 buckets: [0.0,0.1), …, [0.9,1.0])
    buckets = [{"lo": i / 10.0, "hi": (i + 1) / 10.0, "count": 0} for i in range(10)]
    for x in scores_72:
        idx = int(min(9, max(0, int(x * 10))))
        buckets[idx]["count"] += 1

    if s72["n"] == 0:
        md.append("- _no scores in last 72h_")
    else:
        # Show compact histogram string, then one per line for readability
        compact = ", ".join(f"{b['lo']:.1f}-{b['hi']:.1f}:{b['count']}" for b in buckets)
        md.append(f"- **72h histogram**: {compact}")
        # (optional) per-line view for quick scan
        for b in buckets:
            md.append(f"  - {b['lo']:.1f}–{b['hi']:.1f}: {b['count']}")
    # Done

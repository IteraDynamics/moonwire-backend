# scripts/summary_sections/source_precision_recall.py
from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime, timezone
import os
import random

from src.paths import LOGS_DIR
from src.analytics.source_metrics import compute_source_metrics


def _is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")


def _seed_demo_if_needed(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    If DEMO_MODE and no real rows (or only 'unknown'), inject plausible demo rows.
    """
    if not _is_demo_mode():
        return metrics or {}

    rows = (metrics or {}).get("origins") or []
    has_known = any(r.get("origin") and r.get("origin") != "unknown" for r in rows)
    if has_known:
        return metrics

    demo_rows = []
    for origin in ["twitter", "reddit", "rss_news"]:
        precision = round(random.uniform(0.25, 0.9), 2)
        recall    = round(random.uniform(0.10, 0.6), 2)
        demo_rows.append({"origin": origin, "precision": precision, "recall": recall})

    return {"window_days": 7, "origins": demo_rows, "notes": ["_demo mode: metrics seeded_"]}


def render(md: List[str]) -> None:
    """
    Writes a Source Precision & Recall section to the CI summary.
    Reads from retraining logs; seeds in demo mode if empty.
    Env knobs:
      METRICS_DAYS (default 7)
      METRICS_MIN_COUNT (default 1)
    """
    try:
        days = int(os.getenv("METRICS_DAYS", "7"))
    except Exception:
        days = 7

    try:
        min_count = int(os.getenv("METRICS_MIN_COUNT", "1"))
    except Exception:
        min_count = 1

    try:
        metrics = compute_source_metrics(
            flags_path=LOGS_DIR / "retraining_log.jsonl",
            triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
            days=days,
            min_count=min_count,
        )
        metrics = _seed_demo_if_needed(metrics)
        rows = (metrics or {}).get("origins", [])

        md.append(f"\n### 📉 Source Precision & Recall ({days}d)")
        if not rows:
            md.append("_No eligible origins to display._")
            return

        # deterministic order (origin asc)
        for row in sorted(rows, key=lambda r: r.get("origin", "unknown")):
            o = row.get("origin", "unknown")
            p = row.get("precision", 0.0)
            r = row.get("recall", 0.0)
            try:
                md.append(f"- `{o}`: precision={float(p):.2f} | recall={float(r):.2f}")
            except Exception:
                md.append(f"- `{o}`: precision={p} | recall={r}")
    except Exception as e:
        md.append(f"\n_⚠️ Source metrics failed: {e}_")

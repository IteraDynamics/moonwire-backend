# scripts/summary_sections/burst_detection_section.py
from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime, timezone
import os
import random

from src.paths import LOGS_DIR
from src.analytics.burst_detection import compute_bursts


def _is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")


def _known_only(bundle: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """
    Return only origins with a known name and a non-empty 'bursts' list.
    """
    if not bundle:
        return []
    origins = bundle.get("origins", []) or []
    out: List[Dict[str, Any]] = []
    for o in origins:
        if not isinstance(o, dict):
            continue
        if o.get("origin") and o.get("origin") != "unknown" and o.get("bursts"):
            out.append(o)
    return out


def _seed_demo_bursts(days: int, interval: str, z_thresh: float) -> List[Dict[str, Any]]:
    """
    Minimal, self-contained demo seeding (no cross-file deps).
    Creates one burst per origin at 'now' bucket.
    """
    now = datetime.now(timezone.utc)
    if interval == "hour":
        ts = now.replace(minute=0, second=0, microsecond=0)
    else:
        ts = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ts_iso = ts.isoformat().replace("+00:00", "Z")

    demo_origins = ["twitter", "reddit", "rss_news"]
    seeded: List[Dict[str, Any]] = []
    for o in demo_origins:
        seeded.append({
            "origin": o,
            "bursts": [{
                "timestamp_bucket": ts_iso,
                "count": random.randint(20, 60),
                "z_score": round(random.uniform(max(2.0, z_thresh), 3.8), 1),
            }],
        })
    return seeded


def render(md: List[str]) -> None:
    """
    Renders the 'Burst Detection' section into the markdown list.
    Env knobs:
      BURST_DAYS (default 7)
      BURST_INTERVAL (default "hour")
      BURST_Z (default 2.0)
      BURST_TOP (default 3)
    """
    try:
        days = int(os.getenv("BURST_DAYS", "7"))
    except Exception:
        days = 7

    interval = os.getenv("BURST_INTERVAL", "hour")
    try:
        z_thresh = float(os.getenv("BURST_Z", "2.0"))
    except Exception:
        z_thresh = 2.0
    try:
        top_n = int(os.getenv("BURST_TOP", "3"))
    except Exception:
        top_n = 3

    try:
        raw_bursts = compute_bursts(
            flags_path=LOGS_DIR / "retraining_log.jsonl",
            triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
            days=days,
            interval=interval,
            z_thresh=z_thresh,
        )

        display_origins = _known_only(raw_bursts)

        # Demo fallback if empty
        if not display_origins and _is_demo_mode():
            seeded = {"origins": _seed_demo_bursts(days, interval, z_thresh)}
            display_origins = _known_only(seeded) or seeded.get("origins", [])

        md.append(f"\n### 🚨 Burst Detection ({days}d, {interval})")

        if not display_origins:
            md.append("_No bursts detected._")
            return

        # Flatten and show top-N by z-score
        items: List[tuple[str, Dict[str, Any]]] = []
        for o in display_origins:
            origin = o.get("origin", "unknown")
            for b in (o.get("bursts") or []):
                if isinstance(b, dict):
                    items.append((origin, b))

        def _z(b: Dict[str, Any]) -> float:
            try:
                return float(b.get("z_score", 0.0) or 0.0)
            except Exception:
                return 0.0

        items.sort(key=lambda t: _z(t[1]), reverse=True)

        for origin, b in items[:max(1, top_n)]:
            ts = b.get("timestamp_bucket", "n/a")
            try:
                cnt = int(b.get("count", 0) or 0)
            except Exception:
                cnt = 0
            z = _z(b)
            md.append(f"- {origin}: {ts} (count={cnt}, z={z:.1f})")

    except Exception as e:
        md.append(f"\n_⚠️ Burst detection failed: {e}_")

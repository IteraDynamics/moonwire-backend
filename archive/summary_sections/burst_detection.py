# scripts/summary_sections/burst_detection.py
from __future__ import annotations
from datetime import datetime, timezone
from scripts.summary_sections.common import SummaryContext, is_demo_mode

def _generate_demo_bursts_if_needed(data, days=7, interval="hour", z_thresh=2.0):
    if not is_demo_mode():
        return data
    origins = (data or {}).get("origins", [])
    has_known = any(o.get("origin") != "unknown" and o.get("bursts") for o in origins)
    if has_known:
        return data
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    demo_origins = ["twitter", "reddit", "rss_news"]
    demo = []
    for o in demo_origins:
        ts = now.isoformat().replace("+00:00", "Z")
        demo.append({"origin": o, "bursts": [{"timestamp_bucket": ts, "count": 42, "z_score": 3.1}]})
    return {"window_days": days, "interval": interval, "origins": demo, "notes": ["demo bursts seeded"]}

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 🚨 Burst Detection (7d, hour)")
    try:
        from src.analytics.burst_detection import compute_bursts
    except Exception as e:
        md.append(f"_unavailable: {type(e).__name__}_")
        return

    try:
        raw = compute_bursts(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7,
            interval="hour",
            z_thresh=2.0,
        )
    except Exception as e:
        md.append(f"_compute failed: {type(e).__name__}_")
        return

    def _known_only(bundle):
        return [
            o for o in (bundle or {}).get("origins", [])
            if o.get("origin") != "unknown" and o.get("bursts")
        ]

    display = _known_only(raw)
    if not display:
        seeded = _generate_demo_bursts_if_needed(raw, days=7, interval="hour", z_thresh=2.0)
        display = _known_only(seeded) or seeded.get("origins", [])

    if not display:
        md.append("_No bursts detected._")
        return

    items = []
    for o in display:
        for b in o.get("bursts", []):
            items.append((o["origin"], b))
    items.sort(key=lambda t: t[1].get("z_score", 0), reverse=True)
    for origin, b in items[:3]:
        md.append(f"- {origin}: {b['timestamp_bucket']} (count={b['count']}, z={b['z_score']})")
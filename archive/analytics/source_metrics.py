import os
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any
from src.analytics.origin_utils import parse_ts, extract_origin

def compute_source_metrics(
    flags_path: Path,
    triggers_path: Path,
    days: int = 7,
    min_count: int = 1
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    flags_by_origin = defaultdict(int)
    triggers_by_origin = defaultdict(int)

    # Load flags
    if flags_path.exists():
        for line in flags_path.read_text().splitlines():
            try:
                data = json.loads(line)
                ts = parse_ts(data.get("timestamp"))
                if ts and ts >= cutoff:
                    origin = extract_origin(data.get("origin") or data.get("source") or data.get("meta", {}).get("origin") or data.get("metadata", {}).get("source"))
                    flags_by_origin[origin] += 1
            except Exception:
                continue

    # Load triggers
    if triggers_path.exists():
        for line in triggers_path.read_text().splitlines():
            try:
                data = json.loads(line)
                ts = parse_ts(data.get("timestamp"))
                if ts and ts >= cutoff:
                    origin = extract_origin(data.get("origin") or data.get("source") or data.get("meta", {}).get("origin") or data.get("metadata", {}).get("source"))
                    triggers_by_origin[origin] += 1
            except Exception:
                continue

    all_origins = set(flags_by_origin) | set(triggers_by_origin)
    total_triggers = sum(triggers_by_origin.values())
    origins = []

    # DEMO mode seeding
    if not all_origins and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        demo_origins = ["reddit", "twitter", "rss_news"]
        for o in demo_origins:
            flags = random.randint(8, 15)
            triggers = random.randint(1, min(flags, 7))
            flags_by_origin[o] = flags
            triggers_by_origin[o] = triggers
        all_origins = set(demo_origins)
        total_triggers = sum(triggers_by_origin.values())

    for origin in sorted(all_origins):
        flags = flags_by_origin.get(origin, 0)
        triggers = triggers_by_origin.get(origin, 0)
        total = flags + triggers
        if total < min_count:
            continue
        precision = round(triggers / flags, 2) if flags > 0 else 0.0
        recall = round(triggers / total_triggers, 2) if total_triggers > 0 else 0.0
        origins.append({
            "origin": origin,
            "flags": flags,
            "triggers": triggers,
            "total": total,
            "precision": precision,
            "recall": recall
        })

    # Sort: total desc, then origin asc
    origins.sort(key=lambda x: (-x["total"], x["origin"]))

    return {
        "window_days": days,
        "total_triggers": total_triggers,
        "origins": origins
    }

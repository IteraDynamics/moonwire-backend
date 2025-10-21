# scripts/social/backfill_social.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any

# Local imports
from .reddit_lite_ingest import run_ingest as reddit_ingest, IngestPaths as RedditPaths
from .twitter_lite_ingest import run_ingest as twitter_ingest, IngestPaths as TwitterPaths

ISO = "%Y-%m-%dT%H:%M:%SZ"
def _iso(dt: datetime) -> str: return dt.astimezone(timezone.utc).strftime(ISO)

@dataclass
class Paths:
    root: Path
    logs: Path
    models: Path
    arts: Path

def _ensure_paths() -> Paths:
    root = Path.cwd()
    p = Paths(
        root=root,
        logs=root / "logs",
        models=root / "models",
        arts=root / "artifacts" / "social_backfill_plots",
    )
    p.logs.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    p.arts.mkdir(parents=True, exist_ok=True)
    return p

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(str(v).strip()) if v is not None else default
    except Exception:
        return default

def main() -> None:
    paths = _ensure_paths()

    # Window
    backfill_days = _env_int("MW_BACKFILL_DAYS", 72 // 24)  # default 3d if not set
    lookback_h = max(1, backfill_days * 24)

    # Force REAL mode if caller asked for non-demo
    offline_demo = str(os.getenv("MW_OFFLINE_DEMO", "1")).strip().lower() in ("1", "true", "yes", "on")
    demo_flag = offline_demo  # we pass this through to ingesters via env they already read

    # Set per-ingest env knobs those scripts respect
    os.environ["MW_REDDIT_LOOKBACK_H"] = str(lookback_h)
    os.environ["MW_TWITTER_LOOKBACK_H"] = str(lookback_h)

    # Run ingests (each script already supports demo/api modes from env)
    r_ctx = reddit_ingest(paths=RedditPaths(paths.logs, paths.models, paths.arts))
    t_ctx = twitter_ingest(paths=TwitterPaths(paths.logs, paths.models, paths.arts))

    # Summarize
    now = datetime.now(timezone.utc)
    summary: Dict[str, Any] = {
        "generated_at": _iso(now),
        "backfill_days": backfill_days,
        "lookback_hours": lookback_h,
        "env": {
            "MW_OFFLINE_DEMO": os.getenv("MW_OFFLINE_DEMO", ""),
            "MW_REDDIT_MODE": os.getenv("MW_REDDIT_MODE", ""),
            "MW_TWITTER_MODE": os.getenv("MW_TWITTER_MODE", ""),
        },
        "reddit": {
            "demo": bool(r_ctx.get("demo", False)),
            "window_hours": r_ctx.get("window_hours"),
            "subs": r_ctx.get("subs"),
            "counts": r_ctx.get("counts"),
            "bursts": r_ctx.get("bursts", []),
            "top_terms": r_ctx.get("top_terms", []),
        },
        "twitter": {
            "demo": bool(t_ctx.get("demo", False)),
            "window_hours": t_ctx.get("window_hours"),
            "keywords": t_ctx.get("keywords"),
            "counts": t_ctx.get("counts"),
            "bursts": t_ctx.get("bursts", []),
            "top_terms": t_ctx.get("top_terms", []),
        },
    }

    out = paths.models / "social_backfill_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[backfill_social] wrote {out}")

if __name__ == "__main__":
    main()
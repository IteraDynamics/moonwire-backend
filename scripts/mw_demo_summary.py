#!/usr/bin/env python3
"""
MoonWire CI Demo Summary builder.

- Orchestrates summary sections via scripts.summary_sections.build_all
- Ensures artifacts/models/logs dirs exist
- Writes artifacts/demo_summary.md
- Exposes generate_demo_data_if_needed(seeds) used by tests
"""

from __future__ import annotations
import os
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Summary sections registry
from scripts.summary_sections import build_all, SummaryContext


# ---------------------------
# Utilities
# ---------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_artifacts(art_dir: Path) -> List[str]:
    # Only list PNGs we generate commonly; sorted for stable CI diffs
    files = sorted([f.name for f in art_dir.glob("*.png")])
    return files


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------
# Demo data seeding (kept for tests)
# ---------------------------

def generate_demo_data_if_needed(seeds: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Test-facing helper:

    - If DEMO_MODE=false: pass-through (return seeds, [] if provided; or ([], []) if not)
    - If DEMO_MODE=true and seeds is non-empty: pass-through reviewers; no events (tests expect [])
    - If DEMO_MODE=true and seeds is empty:
        * Generate 3-5 deterministic demo reviewers
        * Generate 1 event PER reviewer (tests expect len(events) == len(reviewers))

    Returns:
        reviewers, events
    """
    demo = _bool_env("DEMO_MODE", False) or _bool_env("MW_DEMO", False)

    # Non-demo: do nothing
    if not demo:
        return (seeds or [], [])

    # Demo with pre-supplied reviewers: pass-through, no events
    if seeds:
        return (seeds, [])

    # Demo and no seeds: deterministically create 3-5 reviewers + 1 event per reviewer
    rng = random.Random(42)  # stable across CI runs
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    n = rng.randint(3, 5)
    origins = ["reddit", "twitter", "rss_news"]

    reviewers: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for i in range(1, n + 1):
        origin = origins[(i - 1) % len(origins)]
        score = round(0.55 + 0.1 * rng.random(), 2)
        rid = f"rev_demo_{i}"
        reviewers.append({
            "id": rid,
            "origin": origin,
            "score": score,
            "timestamp": _iso(now - timedelta(minutes=60 - i))  # spread slightly
        })
        events.append({
            "type": "demo_review_created",
            "review_id": rid,
            "at": _iso(now + timedelta(minutes=i)),
            "meta": {
                "note": "seeded in demo mode",
                "version": os.getenv("MW_DEMO_VERSION", "v0.6.9"),
            },
        })

    return reviewers, events


# ---------------------------
# CI Summary builder
# ---------------------------

def main() -> None:
    # Resolve directories (defaults to repo root folders)
    repo_root = Path(os.getcwd())
    models_dir = ensure_dir(Path(os.getenv("MODELS_DIR", repo_root / "models")))
    logs_dir = ensure_dir(Path(os.getenv("LOGS_DIR", repo_root / "logs")))
    artifacts_dir = ensure_dir(Path(os.getenv("ARTIFACTS_DIR", repo_root / "artifacts")))

    # Demo toggle (both legacy DEMO_MODE and MW_DEMO supported)
    is_demo = _bool_env("MW_DEMO", _bool_env("DEMO_MODE", False))

    # Construct context expected by sections
    # SummaryContext in this repo accepts at least (logs_dir, models_dir, is_demo)
    ctx = SummaryContext(logs_dir=logs_dir, models_dir=models_dir, is_demo=is_demo)
    # Some newer sections also look for artifacts_dir on ctx; set it if missing.
    if not hasattr(ctx, "artifacts_dir"):
        setattr(ctx, "artifacts_dir", artifacts_dir)

    # Build all sections (market context, calibration trend, drift response, etc.)
    md_lines: List[str] = []
    md_lines.append("MoonWire CI Demo Summary")
    md_lines.extend(build_all(ctx))

    # Append plots list for convenience
    pngs = list_artifacts(artifacts_dir)
    if pngs:
        md_lines.append("\n🖼️ Plots available")
        for name in pngs:
            md_lines.append(f"\t•\t{name}")

    # Write to file for the workflow to publish
    out_md = artifacts_dir / "demo_summary.md"
    out_md.write_text("\n".join(md_lines))

    # Print a short notice (the workflow step concatenates this file to GITHUB_STEP_SUMMARY)
    print("Job summary generated at run-time")


if __name__ == "__main__":
    main()
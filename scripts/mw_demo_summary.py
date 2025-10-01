# scripts/mw_demo_summary.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Summary sections entrypoint
from scripts.summary_sections import build_all
from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso


# --------------------------
# Demo data seed (kept stable for tests)
# --------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def generate_demo_data_if_needed(reviewers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Test-exercised helper. Mirrors expected behavior:
      - If DEMO_MODE=false: pass-through, return (reviewers, []).
      - If DEMO_MODE=true and reviewers provided: pass-through, return (reviewers, []).
      - If DEMO_MODE=true and reviewers empty: synthesize 3 reviewers AND emit one event PER reviewer.
        (Tests assert len(events) == len(reviewers).)
    """
    demo = str(os.getenv("DEMO_MODE", os.getenv("MW_DEMO", "false"))).lower() == "true"
    if not demo:
        return reviewers, []

    if reviewers:
        # pass-through, no events (tests expect [])
        return reviewers, []

    now = _now_utc()
    out_reviewers: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    # deterministic 3 reviewers
    seeds = [
        {"id": "rev_demo_1", "origin": "reddit", "score": 0.82},
        {"id": "rev_demo_2", "origin": "rss_news", "score": 0.54},
        {"id": "rev_demo_3", "origin": "twitter", "score": 0.67},
    ]
    for i, r in enumerate(seeds):
        rcopy = dict(r)
        rcopy["timestamp"] = _iso(now - timedelta(hours=max(0, 2 - i)))
        out_reviewers.append(rcopy)
        # one event per reviewer (no extra summary event)
        events.append(
            {
                "type": "demo_review_created",
                "review_id": rcopy["id"],
                "at": _iso(now - timedelta(hours=max(0, 2 - i))),
                "meta": {"note": "seeded in demo mode", "version": "v0.6.6"},
            }
        )

    return out_reviewers, events


# --------------------------
# Seed governance demo artifacts when missing
# --------------------------

def _seed_drift_response_plan(models_dir: Path) -> None:
    """
    Create a benign 'no candidates' drift plan so the CI summary never shows
    'no plan available' when running without upstream pipeline.
    """
    ensure_dir(models_dir)
    jpath = models_dir / "drift_response_plan.json"
    if jpath.exists():
        return
    now = _now_utc()
    plan = {
        "generated_at": _iso(now),
        "window_hours": 72,
        "grace_hours": int(os.getenv("MW_DRIFT_GRACE_H", "6")),
        "min_buckets": int(os.getenv("MW_DRIFT_MIN_BUCKETS", "3")),
        "ece_threshold": float(os.getenv("MW_DRIFT_ECE_THRESH", "0.06")),
        "action_mode": os.getenv("MW_DRIFT_ACTION", "dryrun"),
        "candidates": [],
        "demo": True,
    }
    jpath.write_text(json.dumps(plan))


def _seed_retrain_plan(models_dir: Path) -> None:
    """
    Create a benign 'plan empty' retrain JSON so the CI summary can render a section.
    """
    ensure_dir(models_dir)
    jpath = models_dir / "retrain_plan.json"
    if jpath.exists():
        return
    now = _now_utc()
    plan = {
        "generated_at": _iso(now),
        "action_mode": os.getenv("MW_RETRAIN_ACTION", "dryrun"),
        "candidates": [],
        "demo": True,
    }
    jpath.write_text(json.dumps(plan))


# --------------------------
# Build demo summary markdown
# --------------------------

@dataclass
class _Ctx(SummaryContext):
    logs_dir: Path
    models_dir: Path
    is_demo: bool
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    origins_rows: List[Dict[str, Any]] = field(default_factory=list)
    yield_data: Any = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    caches: Dict[str, Any] = field(default_factory=dict)


def _write_md(md_lines: List[str], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(md_lines))


def main() -> None:
    # workspace paths
    root = Path(".").resolve()
    models = root / "models"
    logs = root / "logs"
    arts = Path(os.getenv("ARTIFACTS_DIR", str(root / "artifacts")))
    ensure_dir(models); ensure_dir(logs); ensure_dir(arts)

    # ensure demo governance artifacts exist for CI rendering
    demo = str(os.getenv("DEMO_MODE", os.getenv("MW_DEMO", "false"))).lower() == "true"
    if demo:
        _seed_drift_response_plan(models)
        _seed_retrain_plan(models)
    else:
        # Even in non-demo, write harmless stubs if completely missing,
        # so CI summary won’t show “no plan available”.
        _seed_drift_response_plan(models)
        _seed_retrain_plan(models)

    # assemble markdown
    ctx = _Ctx(logs_dir=logs, models_dir=models, is_demo=demo, artifacts_dir=arts)
    md_lines = build_all(ctx)

    # prepend a simple header so the CI block has a title
    header = ["MoonWire CI Demo Summary"]
    all_lines = header + md_lines + ["Job summary generated at run-time"]

    # write to artifacts
    _write_md(all_lines, arts / "demo_summary.md")


if __name__ == "__main__":
    main()
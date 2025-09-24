#!/usr/bin/env python
"""
MoonWire CI Demo Summary builder.

Order of operations:
  1) Ensure Market Context artifacts exist (live or demo) by calling ingest.
  2) Build summary sections (Market Context, then Calibration Trend overlays, then any optional sections).
  3) Write artifacts/demo_summary.md

This script is intentionally resilient:
- Respects DEMO_MODE / MW_DEMO envs.
- Never crashes CI on a single section failure — it prints inline error notes in the markdown.
- Works even if some sections are absent in the branch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Make repo root importable when executed as a module or a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imports from our package
from scripts.summary_sections import SummaryContext, build_all

# Ingest is optional here; if missing we degrade gracefully
try:
    from scripts.market.ingest_market import run_ingest  # ensures logs/models/artifacts & market_context.json
except Exception:
    run_ingest = None  # type: ignore


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def main() -> int:
    repo = REPO_ROOT
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(repo / "artifacts")))
    models_dir = Path(os.getenv("MODELS_DIR", str(repo / "models")))
    logs_dir = Path(os.getenv("LOGS_DIR", str(repo / "logs")))

    # Ensure dirs exist
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Demo flags
    is_demo = _env_bool("DEMO_MODE", False) or _env_bool("MW_DEMO", False)

    # 1) Ingest market context first — required for calibration overlays to have market regimes
    ingest_note = ""
    if callable(run_ingest):
        try:
            out = run_ingest(logs_dir, models_dir, artifacts_dir)
            ingest_note = f"✅ Market ingest completed ({out})."
        except Exception as e:
            ingest_note = f"⚠️ Market ingest failed: {type(e).__name__}: {e}"
    else:
        ingest_note = "⚠️ Market ingest unavailable in this branch (no run_ingest)."

    # 2) Build summary sections
    ctx = SummaryContext(
        logs_dir=logs_dir,
        models_dir=models_dir,
        is_demo=is_demo,
    )
    md_blocks = build_all(ctx)

    # 3) Write summary
    out_md = artifacts_dir / "demo_summary.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = [
        f"# MoonWire CI Demo Summary",
        "",
        f"_Generated: {now}_",
        "",
        ingest_note,
        "",
        "---",
        "",
    ]
    out_md.write_text("\n".join(header + md_blocks), encoding="utf-8")

    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
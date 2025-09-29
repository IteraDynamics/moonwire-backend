#!/usr/bin/env python3
"""
MoonWire CI Demo Summary builder.

- Orchestrates summary sections via scripts.summary_sections.build_all
- Ensures artifacts/models/logs dirs exist
- Writes artifacts/demo_summary.md
- Prints a compact tail to stdout (useful for local runs)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List

# Summary sections registry
from scripts.summary_sections import build_all, SummaryContext

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

    # Build all sections (market context, calibration trend, AND the new drift response)
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
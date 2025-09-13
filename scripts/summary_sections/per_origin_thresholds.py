# scripts/summary_sections/per_origin_thresholds.py
from __future__ import annotations
from scripts.summary_sections.common import SummaryContext

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 🎯 Per-Origin Thresholds")
    try:
        from src.ml.thresholds import load_per_origin_thresholds
        thresholds = load_per_origin_thresholds()
    except Exception as e:
        md.append(f"- [demo] fallback thresholds in use ({type(e).__name__})")
        return

    example_count = 0
    for origin, vals in thresholds.items():
        try:
            if "p70" in vals and "p80" in vals:
                md.append(f"- {origin}: p70={float(vals['p70']):.2f}, p80={float(vals['p80']):.2f}")
                example_count += 1
        except Exception:
            continue
        if example_count >= 2:
            break

    if example_count == 0:
        md.append("- [demo] fallback thresholds in use")
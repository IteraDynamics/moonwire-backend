# scripts/summary_sections/calibration.py
from __future__ import annotations
from scripts.summary_sections.common import SummaryContext

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 📏 Calibration")
    try:
        from src.ml.infer import model_metadata
        meta = model_metadata() or {}
        calib = meta.get("calibration", {})
    except Exception:
        calib = {}

    if "brier_pre" in calib and "brier_post" in calib:
        md.append(f"post-calibration Brier={float(calib['brier_post']):.4f} (vs pre={float(calib['brier_pre']):.4f})")
    elif calib:
        md.append(f"Available metrics: {list(calib.keys())}")
    else:
        md.append("[demo] calibration not available")
# scripts/summary_sections/calibration_section.py
from __future__ import annotations

from typing import List, Optional, Dict, Any


def _fmt_float(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def render(md: List[str], meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Render the '📏 Calibration' section into the running markdown list.
    If `meta` is not provided, this lazily loads via src.ml.infer.model_metadata().
    """
    md.append("\n### 📏 Calibration")

    # Load metadata if not supplied
    if meta is None:
        try:
            from src.ml.infer import model_metadata  # type: ignore
            meta = model_metadata() or {}
        except Exception:
            meta = {}

    calib = {}
    if isinstance(meta, dict):
        calib = meta.get("calibration") or {}

    # Preferred concise line if Brier pre/post available
    if "brier_pre" in calib and "brier_post" in calib:
        md.append(
            f"post-calibration Brier={_fmt_float(calib.get('brier_post'))} "
            f"(vs pre={_fmt_float(calib.get('brier_pre'))})"
        )
        return

    # Otherwise, list available keys (helps debugging)
    if calib:
        keys = list(calib.keys())
        preferred = [k for k in ("method", "on", "roc_auc_pre", "roc_auc_post", "brier_pre", "brier_post") if k in calib]
        rest = [k for k in keys if k not in preferred]
        show = preferred + rest
        md.append("Available metrics: " + ", ".join(show[:8]) + ("…" if len(show) > 8 else ""))
    else:
        md.append("[demo] calibration not available")
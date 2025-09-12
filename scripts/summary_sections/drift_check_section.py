# scripts/summary_sections/drift_check_section.py
from __future__ import annotations

from typing import List, Dict, Any, Optional


def _coerce_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _round2(v) -> float:
    try:
        return round(float(v), 2)
    except Exception:
        return 0.0


def render(md: List[str], drift: Optional[Dict[str, Any]] = None, score_min: float = 0.6) -> None:
    """
    Render the 'Drift Check (features)' section.

    Parameters
    ----------
    md : list[str]
        Markdown buffer to append to.
    drift : dict | None
        Expected shapes (any of these):
          { "features": [ { "feature": "...", "score": 0.7, "delta_mean": ..., "nz_train": ..., "nz_live": ... }, ... ] }
          { "items":    [ { "...": ... }, ... ] }
        Each item is normalized defensively with fallbacks:
          - name:  feature | name
          - score: score | drift_score | 0.0
          - Δmean: delta_mean | delta | 0.0
          - nz%:   nz_train | nz_pct_train | nz_train_pct | nz_train_percent | train_nonzero_pct
                   nz_live  | nz_pct_live  | nz_live_pct  | nz_live_percent  | live_nonzero_pct
    score_min : float
        Threshold for "material drift". Defaults to 0.6 (previously 1.0).
    """
    md.append("\n### 🔎 Drift Check (features)")

    items = []
    try:
        src = drift or {}
        items = (src.get("features") or src.get("items") or []) if isinstance(src, dict) else []
    except Exception:
        items = []

    # Normalize rows
    norm = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            feat = it.get("feature") or it.get("name") or "feature"
            score = _coerce_float(it.get("score", it.get("drift_score", 0.0)), 0.0)
            dmean = _coerce_float(it.get("delta_mean", it.get("delta", 0.0)), 0.0)
            nz_tr = _coerce_float(
                it.get("nz_train")
                or it.get("nz_pct_train")
                or it.get("nz_train_pct")
                or it.get("nz_train_percent")
                or it.get("train_nonzero_pct")
                or 0.0,
                0.0,
            )
            nz_lv = _coerce_float(
                it.get("nz_live")
                or it.get("nz_pct_live")
                or it.get("nz_live_pct")
                or it.get("nz_live_percent")
                or it.get("live_nonzero_pct")
                or 0.0,
                0.0,
            )
        except Exception:
            continue

        norm.append(
            {
                "feature": str(feat),
                "score": float(score),
                "delta_mean": float(dmean),
                "nz_train": float(nz_tr),
                "nz_live": float(nz_lv),
            }
        )

    # Filter + top-3 by score
    top = [x for x in norm if x["score"] >= float(score_min)]
    top.sort(key=lambda x: x["score"], reverse=True)
    top = top[:3]

    if not top:
        md.append("No material drift detected.")
        return

    for x in top:
        md.append(
            f"- {x['feature']}: Δmean={_round2(x['delta_mean'])}, "
            f"nz% {round(x['nz_train'])}→{round(x['nz_live'])}, "
            f"score={_round2(x['score'])}"
        )

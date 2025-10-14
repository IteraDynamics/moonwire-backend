# scripts/governance/bluegreen_promotion.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso

# -----------------------
# Helpers
# -----------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text() or "{}")
    except Exception:
        pass
    return {}

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))

# Minimal 1x1 PNG if matplotlib not present
_PNG_1x1_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

def _write_png_placeholder(path: Path, title: str = "") -> None:
    if path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa
        ensure_dir(path.parent)
        fig = plt.figure(figsize=(3, 2), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, title or "MoonWire", ha="center", va="center", wrap=True)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(path))
        plt.close(fig)
        return
    except Exception:
        pass
    ensure_dir(path.parent)
    path.write_bytes(_PNG_1x1_BYTES)

# -----------------------
# Core logic
# -----------------------

@dataclass
class _Cfg:
    lookback_h: int
    delta_thresh: float
    conf_thresh: float

def _cfg_from_env() -> _Cfg:
    return _Cfg(
        lookback_h=int(os.getenv("MW_BG_LOOKBACK_H", "72")),
        delta_thresh=float(os.getenv("MW_BG_DELTA_THRESH", "0.02")),
        conf_thresh=float(os.getenv("MW_BG_CONF_THRESH", "0.8")),
    )

def _get_current_model(models_dir: Path) -> Optional[str]:
    p = models_dir / "current_model.txt"
    try:
        if p.exists():
            val = p.read_text().strip()
            return val or None
    except Exception:
        pass
    return None

def _pick_candidate(plan: Dict[str, Any]) -> Optional[Tuple[str, float, List[str]]]:
    """
    Choose first 'promote' action with the highest confidence.
    Returns (version, confidence, reasons).
    """
    acts = plan.get("actions", []) or []
    best = None
    for a in acts:
        if str(a.get("action")) != "promote":
            continue
        ver = a.get("version")
        conf = float(a.get("confidence", 0.0) or 0.0)
        reasons = a.get("reasons") or a.get("reason") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        if ver:
            if not best or conf > best[1]:
                best = (ver, conf, reasons)
    return best

def _metrics_for_version(trend: Dict[str, Any], calib: Dict[str, Any], ver: str) -> Dict[str, Optional[float]]:
    # From model_performance_trend.json: search versions list
    mt = {"precision": None, "recall": None, "F1": None, "ECE": None}
    try:
        for v in trend.get("versions", []) or []:
            if v.get("version") == ver:
                # accept any subset; tests don't require full set
                for k in ("precision", "recall", "F1", "f1", "ece", "ECE"):
                    if k in v:
                        key = "F1" if k.lower() == "f1" else ("ECE" if k.lower() == "ece" else k)
                        mt[key] = float(v.get(k))
                break
    except Exception:
        pass

    # From calibration_trend.json: ECE override if present
    try:
        for v in calib.get("versions", []) or []:
            if v.get("version") == ver and "ECE" in v:
                mt["ECE"] = float(v["ECE"])
                break
    except Exception:
        pass
    return mt

def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except Exception:
        return None

def _classify(dF1: Optional[float], dECE: Optional[float], conf: float, cfg: _Cfg) -> str:
    # Safe defaults when metrics missing
    if dF1 is None and dECE is None:
        return "observe"
    # Risk first
    if (dF1 is not None and dF1 < -0.02) or (dECE is not None and dECE > 0.01):
        return "rollback_risk"
    # Promote readiness
    imp_ok = (dF1 is not None and dF1 >= cfg.delta_thresh) or (dECE is not None and -dECE >= cfg.delta_thresh)
    if imp_ok and conf >= cfg.conf_thresh:
        return "promote_ready"
    return "observe"

def _fmt_delta(x: Optional[float]) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    s = f"{x:+.2f}"
    return s

# -----------------------
# Public API
# -----------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Blue-Green Model Promotion Simulation (read-only).
    - Reads current (Blue) and a candidate (Green) from governance actions.
    - Compares metrics and writes JSON + artifacts + CI block lines.
    """
    models = Path(ctx.models_dir or "models")
    arts = Path(getattr(ctx, "artifacts_dir", "artifacts") or "artifacts")
    ensure_dir(models); ensure_dir(arts)

    cfg = _cfg_from_env()

    # Inputs
    gov_plan = _read_json(models / "model_governance_actions.json")
    trend = _read_json(models / "model_performance_trend.json")
    calib = _read_json(models / "calibration_trend.json")  # optional, tolerate missing

    current = _get_current_model(models) or "v0.7.7"  # default for demo realism
    cand_pick = _pick_candidate(gov_plan)

    # Demo safeguard: ensure a candidate exists in demo mode
    demo = bool(getattr(ctx, "is_demo", False))
    if not cand_pick and demo:
        cand_pick = ("v0.7.9", 0.85, ["demo_seed", "meets_delta_threshold"])

    if not cand_pick:
        # Nothing to simulate; still create a small JSON + line for CI
        out = {
            "generated_at": _iso(_now_utc()),
            "current_model": current,
            "candidate": None,
            "delta": {},
            "classification": "observe",
            "confidence": 0.0,
            "notes": ["no_candidate"],
        }
        _write_json(models / "bluegreen_promotion.json", out)
        md.append("🧪 Blue-Green Promotion Simulation ({} h)".format(cfg.lookback_h))
        md.append(f"no candidate available | current: {current}")
        return

    candidate, conf, reasons = cand_pick

    # Metrics lookup (tolerant of missing keys)
    m_cur = _metrics_for_version(trend, calib, current)
    m_cand = _metrics_for_version(trend, calib, candidate)

    # Demo seeding: if nothing present, synthesize a realistic improvement
    if demo and all(v is None for v in m_cur.values()) and all(v is None for v in m_cand.values()):
        # seed: +0.02 F1, -0.01 ECE for candidate
        m_cur = {"precision": 0.76, "recall": 0.71, "F1": 0.74, "ECE": 0.06}
        m_cand = {"precision": 0.78, "recall": 0.73, "F1": 0.76, "ECE": 0.05}
        if conf <= 0.0:
            conf = 0.85
        if not reasons:
            reasons = ["demo_seed", "calibration_improved"]

    dP = _delta(m_cur.get("precision"), m_cand.get("precision"))
    dR = _delta(m_cur.get("recall"), m_cand.get("recall"))
    dF1 = _delta(m_cur.get("F1"), m_cand.get("F1"))
    dECE = None
    if m_cur.get("ECE") is not None or m_cand.get("ECE") is not None:
        try:
            # Note: improvement is negative delta for ECE
            dECE = float(m_cand.get("ECE") or 0.0) - float(m_cur.get("ECE") or 0.0)
        except Exception:
            dECE = None

    classification = _classify(dF1, dECE, conf, cfg)

    out = {
        "generated_at": _iso(_now_utc()),  # <<< FIX: pass dt to _iso
        "current_model": current,
        "candidate": candidate,
        "delta": {
            "precision": dP,
            "recall": dR,
            "F1": dF1,
            "ECE": dECE,
        },
        "classification": classification,
        "confidence": round(float(conf), 2),
        "notes": list(reasons or []),
        "window_hours": int(cfg.lookback_h),
    }
    _write_json(models / "bluegreen_promotion.json", out)

    # Artifacts (comparison + timeline placeholders unless matplotlib available)
    _write_png_placeholder(arts / f"bluegreen_comparison_{candidate}.png", f"Blue vs Green ({current} → {candidate})")
    _write_png_placeholder(arts / "bluegreen_timeline.png", "Blue-Green Timeline")

    # CI Markdown block
    md.append(f"🧪 Blue-Green Promotion Simulation ({cfg.lookback_h} h)")
    md.append(f"current → candidate: {current} → {candidate}")
    md.append(
        f"ΔF1 {_fmt_delta(dF1)} | ΔECE {_fmt_delta(-dECE if dECE is not None else None)} | "
        f"conf {out['confidence']:.2f} → {classification.upper()}"
    )
    md.append("visuals: comparison + timeline")

#!/usr/bin/env python3
"""
Retrain Automation — minimal v1
- Decides whether to propose retrain (based on drift plan + calibration trend, if present)
- Writes models/retrain_plan.json
- Always safe: falls back to empty (demo-aware) plan when inputs absent
"""

from __future__ import annotations
import json, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

UTC = timezone.utc


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00","Z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _get_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","y","on")


def _choose_candidates(cal: Dict[str,Any], drift_plan: Dict[str,Any]) -> List[Dict[str,Any]]:
    """
    Minimal heuristic:
    - If drift response has any 'proceed' candidates or reasons matching 'high_ece_persistent'
      AND calibration still shows last point high, propose a retrain plan entry (dry-run).
    """
    cands: List[Dict[str,Any]] = []
    dcs = drift_plan.get("candidates") or []
    series = cal.get("series") or []
    last_high = False
    for s in series:
        pts = s.get("points") or []
        if pts:
            last = pts[-1]
            try:
                if float(last.get("ece",0.0)) > float(drift_plan.get("ece_threshold", 0.06)):
                    last_high = True
            except Exception:
                pass

    for c in dcs:
        reasons = [str(r) for r in c.get("reasons") or []]
        if "high_ece_persistent" in reasons or c.get("decision") == "proceed":
            cands.append({
                "origin": c.get("origin","unknown"),
                "current_version": c.get("model_version","v0"),
                "reason": list(set(["high_ece_persistent"] + reasons + (["still_high"] if last_high else []))),
                "window_days": int(os.getenv("MW_RETRAIN_LOOKBACK_DAYS", "30")),
                "labels": 0,
                "datasets": {"path": ""},
                "eval": {"precision_delta": 0.0, "ece_delta": -0.01, "f1_delta": 0.0},
                "decision": "hold" if not last_high else "plan",
                "new_version": None,
            })
    return cands


def build_and_write_plan(models_dir: Path, demo: bool) -> Path:
    cal = _read_json(models_dir / "calibration_reliability_trend.json") or {}
    drift_plan = _read_json(models_dir / "drift_response_plan.json") or {}

    action_mode = os.getenv("MW_RETRAIN_ACTION", "dryrun")

    candidates = _choose_candidates(cal, drift_plan) if (cal and drift_plan) else []

    plan = {
        "generated_at": _now_iso(),
        "action_mode": action_mode,
        "candidates": candidates,
        "demo": demo,
    }

    out = models_dir / "retrain_plan.json"
    _ensure_dir(models_dir)
    out.write_text(json.dumps(plan, indent=2))
    print(f"[retrain_automation] wrote {out}")
    return out


def main() -> int:
    cwd = Path(os.getcwd())
    repo = cwd
    models = Path(os.getenv("MODELS_DIR", repo / "models"))
    demo = _get_env_bool("MW_DEMO", False) or _get_env_bool("DEMO_MODE", False)
    build_and_write_plan(models, demo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
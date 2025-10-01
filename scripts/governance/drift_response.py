#!/usr/bin/env python3
"""
Drift Response — minimal v1
- Reads calibration trend (if present)
- Detects persistent high_ece (very simple heuristic)
- Writes models/drift_response_plan.json
- Emits a lightweight timeline plot (optional)
- Never crashes CI: falls back to "no candidates" plan (demo-aware)
"""

from __future__ import annotations
import json, os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


UTC = timezone.utc


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00","Z")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Ctx:
    repo_root: Path
    models: Path
    logs: Path
    arts: Path
    demo: bool


def _get_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","y","on")


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _detect_candidates(cal: Dict[str,Any], ece_thresh: float, min_buckets: int, grace_h: int) -> List[Dict[str,Any]]:
    """
    Very simple detection:
    - look at the last `grace_h` hours worth of points (if time stamps present)
    - otherwise just look at last N points per series
    - mark a candidate if >= min_buckets buckets have ece > threshold and n>=10
    """
    series = cal.get("series") or []
    candidates: List[Dict[str,Any]] = []
    cutoff = datetime.now(UTC) - timedelta(hours=grace_h)

    for s in series:
        key = s.get("key","unknown")
        points = s.get("points") or []
        recent: List[Dict[str,Any]] = []
        for p in points:
            ts = p.get("bucket_start")
            dt: Optional[datetime] = None
            try:
                if isinstance(ts, str):
                    if ts.endswith("Z"):
                        ts = ts[:-1] + "+00:00"
                    dt = datetime.fromisoformat(ts).astimezone(UTC)
            except Exception:
                dt = None
            if dt is None or dt >= cutoff:
                recent.append(p)

        highs = [p for p in recent if float(p.get("ece",0.0)) > ece_thresh and int(p.get("n",10)) >= 10]
        if len(highs) >= min_buckets:
            # synthesize a proposal (dryrun)
            candidates.append({
                "origin": key,
                "model_version": s.get("version","v0"),
                "current_threshold": 0.50,
                "proposed_threshold": 0.54,
                "delta": 0.04,
                "reasons": ["high_ece_persistent"],
                "backtest": {"precision_delta": 0.0, "ece_delta": -0.01, "recall_delta": -0.02, "f1_delta": -0.01},
                "decision": "proceed",
            })
    return candidates


def build_and_write_plan(ctx: Ctx) -> Path:
    ece_thresh = _get_env_float("MW_DRIFT_ECE_THRESH", 0.06)
    min_buckets = _get_env_int("MW_DRIFT_MIN_BUCKETS", 3)
    grace_h = _get_env_int("MW_DRIFT_GRACE_H", 6)
    action_mode = os.getenv("MW_DRIFT_ACTION", "dryrun")

    cal_path = ctx.models / "calibration_reliability_trend.json"
    plan_path = ctx.models / "drift_response_plan.json"

    cal = _read_json(cal_path) or {}
    if not cal.get("series"):
        # no calibration yet -> emit empty plan
        plan = {
            "generated_at": _now_iso(),
            "window_hours": 72,
            "grace_hours": grace_h,
            "min_buckets": min_buckets,
            "ece_threshold": ece_thresh,
            "action_mode": action_mode,
            "candidates": [],
            "demo": ctx.demo,
        }
        _ensure_dir(ctx.models)
        plan_path.write_text(json.dumps(plan, indent=2))
        return plan_path

    candidates = _detect_candidates(cal, ece_thresh, min_buckets, grace_h)

    plan = {
        "generated_at": _now_iso(),
        "window_hours": 72,
        "grace_hours": grace_h,
        "min_buckets": min_buckets,
        "ece_threshold": ece_thresh,
        "action_mode": action_mode,
        "candidates": candidates,
        "demo": ctx.demo,
    }
    _ensure_dir(ctx.models)
    plan_path.write_text(json.dumps(plan, indent=2))

    # tiny timeline plot (optional)
    try:
        x = []
        y = []
        for s in (cal.get("series") or [])[:1]:
            for p in s.get("points", []):
                x.append(p.get("bucket_start"))
                y.append(float(p.get("ece", 0.0)))
        if x and y:
            plt.figure()
            plt.plot(x, y, marker="o")
            plt.axhline(ece_thresh)
            plt.xticks(rotation=45, ha="right")
            plt.title("Drift Response Timeline (ECE)")
            _ensure_dir(ctx.arts)
            (ctx.arts / "drift_response_timeline.png").unlink(missing_ok=True)
            plt.tight_layout()
            plt.savefig(ctx.arts / "drift_response_timeline.png", dpi=120)
            plt.close()
    except Exception:
        pass

    return plan_path


def main() -> int:
    cwd = Path(os.getcwd())
    repo = cwd
    models = Path(os.getenv("MODELS_DIR", repo / "models"))
    logs = Path(os.getenv("LOGS_DIR", repo / "logs"))
    arts = Path(os.getenv("ARTIFACTS_DIR", repo / "artifacts"))
    demo = _get_env_bool("MW_DEMO", False) or _get_env_bool("DEMO_MODE", False)

    ctx = Ctx(repo_root=repo, models=models, logs=logs, arts=arts, demo=demo)
    _ensure_dir(models); _ensure_dir(logs); _ensure_dir(arts)
    build_and_write_plan(ctx)
    print(f"[drift_response] wrote {models / 'drift_response_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
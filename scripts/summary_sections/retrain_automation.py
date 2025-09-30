# scripts/governance/retrain_automation.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Matplotlib (Agg only, single-plot, no custom colors per repo rules)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ------------------------
# Small filesystem helpers
# ------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(p: Path, default=None):
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default


def _write_json(p: Path, obj: Any) -> None:
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2))


def _append_jsonl(p: Path, line: Dict[str, Any]) -> None:
    _ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ------------------------
# Public entrypoint
# ------------------------

def run(ctx) -> Dict[str, Any]:
    """
    Orchestrates retrain automation:
      - builds a plan (possibly demo)
      - writes models/retrain_plan.json
      - (optionally) writes artifacts and governance logs

    Returns the plan dict for summary rendering.
    """
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    logs_dir = Path(getattr(ctx, "logs_dir", "logs"))
    arts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    _ensure_dir(models_dir); _ensure_dir(logs_dir); _ensure_dir(arts_dir)

    action_mode = (os.getenv("MW_RETRAIN_ACTION", "dryrun") or "dryrun").lower()
    demo_env = os.getenv("MW_DEMO") or os.getenv("DEMO_MODE")
    is_demo = bool(getattr(ctx, "is_demo", False) or (str(demo_env).lower() == "true"))

    # Try to build a real plan from inputs; if not enough signal, provide a demo/empty plan.
    plan = _build_plan(models_dir=models_dir, logs_dir=logs_dir, action_mode=action_mode)

    if not plan.get("candidates") and is_demo:
        # Seed a friendly demo candidate so the section shows up informatively.
        plan = _build_demo_plan(action_mode=action_mode)

    # Persist plan artifact
    plan["generated_at"] = _iso(datetime.now(timezone.utc))
    plan["action_mode"] = action_mode
    _write_json(models_dir / "retrain_plan.json", plan)

    # Create simple eval plots for each candidate that has eval deltas (demo or real)
    for c in plan.get("candidates", []) or []:
        evald = c.get("eval") or {}
        if not evald:
            continue
        origin = c.get("origin", "unknown")
        curr = c.get("current_version", "v0")
        newv = c.get("new_version", "vX")
        _plot_eval_bars(
            arts_dir / f"retrain_eval_{origin}_{curr}_to_{newv}.png",
            precision_delta=evald.get("precision_delta", 0.0),
            ece_delta=evald.get("ece_delta", 0.0),
            f1_delta=evald.get("f1_delta", 0.0),
        )

    # If apply & decision == promote, append governance log (we do not actually move models here;
    # that belongs in a fuller pipeline; this keeps CI stable).
    if action_mode == "apply":
        for c in plan.get("candidates", []) or []:
            if c.get("decision") == "promote":
                _append_jsonl(
                    logs_dir / "governance_actions.jsonl",
                    {
                        "ts_utc": _iso(datetime.now(timezone.utc)),
                        "action": "retrain_promoted",
                        "mode": "apply",
                        "origin": c.get("origin"),
                        "current_version": c.get("current_version"),
                        "new_version": c.get("new_version"),
                        "policy": "retrain_automation_v1",
                        "context": {"eval": c.get("eval", {})},
                    },
                )

    return plan


# ------------------------
# Internal: planning logic
# ------------------------

def _build_plan(models_dir: Path, logs_dir: Path, action_mode: str) -> Dict[str, Any]:
    """
    Minimal non-demo plan:
      - Look at drift_response_plan.json; if it claims max shift reached recently and
        calibration_trend shows persistent high_ece, propose a candidate in 'plan' or 'hold' state.
      - Otherwise return empty plan.
    This is intentionally conservative to avoid heavy training in CI.
    """
    drift = _read_json(models_dir / "drift_response_plan.json", default={}) or {}
    cal = _read_json(models_dir / "calibration_reliability_trend.json", default={}) or {}
    candidates: List[Dict[str, Any]] = []

    # Heuristic: if any point in the last ~6h has "high_ece" AND drift plan mentions "max_shift"
    # for a known origin/version, surface a planning line (no eval deltas).
    recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
    recent_iso = _iso(recent_cutoff)

    max_shift_origins = set()
    for c in drift.get("candidates", []) or []:
        reasons = set((c.get("reasons") or []))
        if any("max_shift" in r for r in reasons):
            max_shift_origins.add((c.get("origin", "reddit"), c.get("model_version", "v0")))

    if max_shift_origins and cal.get("series"):
        for s in cal["series"]:
            key = s.get("key") or s.get("origin") or "reddit"
            pts = s.get("points", [])
            if not pts:
                continue
            # any high_ece in the last few buckets?
            recent_high = any(
                ("high_ece" in (p.get("alerts") or [])) and (p.get("bucket_start", "") >= recent_iso)
                for p in pts
            )
            if recent_high:
                # match to a model_version when available; default v0
                mv = "v0"
                # if drift plan had an entry for this origin, reuse its version
                for (o, ver) in max_shift_origins:
                    if o == key:
                        mv = ver or mv
                        break
                candidates.append({
                    "origin": key,
                    "current_version": mv,
                    "reasons": ["high_ece_persistent", "max_shift_reached"],
                    "decision": "plan",
                })

    return {
        "generated_at": _iso(datetime.now(timezone.utc)),
        "action_mode": action_mode,
        "candidates": candidates,
        "demo": False,
    }


def _build_demo_plan(action_mode: str) -> Dict[str, Any]:
    """
    Deterministic demo plan with one promotable candidate.
    """
    return {
        "generated_at": _iso(datetime.now(timezone.utc)),
        "action_mode": action_mode,
        "candidates": [
            {
                "origin": "reddit",
                "current_version": "v17",
                "new_version": "v18",
                "reasons": ["high_ece_persistent", "max_shift_reached"],
                "eval": {
                    "precision_delta": 0.018,  # +1.8pp
                    "ece_delta": -0.014,       # -0.014 absolute
                    "f1_delta": 0.006,         # +0.6pp
                },
                "decision": "promote" if action_mode == "apply" else "hold",
            }
        ],
        "demo": True,
    }


# ------------------------
# Internal: plotting
# ------------------------

def _plot_eval_bars(out_path: Path, precision_delta: float, ece_delta: float, f1_delta: float) -> None:
    """Single bar chart of delta metrics (no custom colors)."""
    _ensure_dir(out_path.parent)
    labels = ["ΔPrecision (pp)", "ΔECE", "ΔF1 (pp)"]
    values = [precision_delta * 100.0, ece_delta, f1_delta * 100.0]

    plt.figure(figsize=(5, 3))
    plt.bar(labels, values)  # no color styling
    plt.title("Retrain Evaluation Deltas")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
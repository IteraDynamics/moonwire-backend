# scripts/governance/bluegreen_promotion.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# We avoid importing seaborn; use matplotlib with a non-interactive backend
import matplotlib
matplotlib.use("Agg")  # headless CI-safe
import matplotlib.pyplot as plt

# Try to use shared helpers if present, but keep safe fallbacks
try:
    from scripts.summary_sections.common import SummaryContext, ensure_dir  # type: ignore
    from scripts.summary_sections.common import _iso as _iso_common  # type: ignore
except Exception:
    SummaryContext = Any  # type: ignore

    def ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def _iso_common(dt=None) -> str:
        from datetime import datetime, timezone
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


# ---------- Env helpers ----------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


# ---------- IO helpers ----------
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


# ---------- Metric plumbing ----------
def _version_metrics(
    trend: Dict[str, Any],
    ver: str
) -> Dict[str, Optional[float]]:
    """
    Extract metrics for a version from model_performance_trend / calibration_trend.
    Accepts flexible keys; falls back to None if missing.
    """
    lookup = {}
    for v in (trend or {}).get("versions", []):
        if v.get("version") == ver:
            lookup = v
            break
    # Common metric key shorthands
    precision = lookup.get("precision")
    recall = lookup.get("recall")
    f1 = lookup.get("f1") or lookup.get("F1")
    ece = lookup.get("ece") or lookup.get("ECE")

    # Some repos only store deltas; ensure floats
    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    return {
        "precision": _f(precision),
        "recall": _f(recall),
        "f1": _f(f1),
        "ece": _f(ece),
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


# ---------- Classification ----------
def _classify(
    dF1: Optional[float],
    dECE: Optional[float],
    conf: float,
    delta_thresh: float,
    conf_thresh: float
) -> str:
    # rollback risk first: clear degradations
    if (dF1 is not None and dF1 < -0.02) or (dECE is not None and dECE > 0.01):
        return "rollback_risk"

    # promote ready if improvements and confidence good
    imp_ok = (dF1 is not None and dF1 >= delta_thresh) or (dECE is not None and dECE <= -0.01)
    if imp_ok and conf >= conf_thresh:
        return "promote_ready"

    return "observe"


# ---------- Plotting ----------
def _plot_comparison_png(
    out_path: Path,
    blue: Dict[str, Optional[float]],
    green: Dict[str, Optional[float]],
    blue_label: str,
    green_label: str
) -> None:
    ensure_dir(out_path.parent)

    metrics = ["precision", "recall", "f1", "ece"]
    x = list(range(len(metrics)))
    blue_vals = [(blue[m] if blue[m] is not None else 0.0) for m in metrics]
    green_vals = [(green[m] if green[m] is not None else 0.0) for m in metrics]

    fig = plt.figure(figsize=(6, 3.5), dpi=140)
    width = 0.35
    plt.bar([i - width/2 for i in x], blue_vals, width, label=blue_label)
    plt.bar([i + width/2 for i in x], green_vals, width, label=green_label)
    plt.xticks(x, ["Precision", "Recall", "F1", "ECE"])
    plt.ylabel("Value")
    plt.title("Blue vs Green")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_timeline_placeholder(out_path: Path) -> None:
    """Simple placeholder timeline so CI artifacts exist even if no series present."""
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(6, 2.4), dpi=140)
    plt.plot([0, 1, 2, 3], [0.5, 0.6, 0.62, 0.67])
    plt.title("Metrics Timeline (demo placeholder)")
    plt.xlabel("t")
    plt.ylabel("score")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------- Main API ----------
def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Blue-Green Model Promotion Simulation
    - Reads current and candidate versions
    - Compares metrics and emits a classification + visuals
    - Writes models/bluegreen_promotion.json
    - Appends a CI markdown block to `md`
    - No file mutations
    """
    models = Path(ctx.models_dir or "models")
    artifacts = Path(ctx.artifacts_dir or "artifacts")
    ensure_dir(models); ensure_dir(artifacts)

    lookback_h = _env_int("MW_BG_LOOKBACK_H", 72)
    delta_thresh = _env_float("MW_BG_DELTA_THRESH", 0.02)
    conf_thresh = _env_float("MW_BG_CONF_THRESH", 0.80)

    # Inputs
    plan = _read_json(models / "model_governance_actions.json") or {}
    trend = _read_json(models / "model_performance_trend.json") or {}
    calib = _read_json(models / "calibration_trend.json") or {}  # optional

    # Determine Blue (current) version
    blue_ver = None
    cur_file = models / "current_model.txt"
    if cur_file.exists():
        try:
            blue_ver = cur_file.read_text().strip() or None
        except Exception:
            blue_ver = None
    if not blue_ver:
        # fall back to first trend version
        vs = [v.get("version") for v in trend.get("versions", []) if v.get("version")]
        blue_ver = vs[0] if vs else "v0.7.7"

    # Determine Green (candidate) version
    cand_ver = None
    for a in (plan.get("actions") or []):
        if a.get("action") in ("promote", "rollback"):  # prefer explicit governance targets
            v = a.get("version")
            if v and v != blue_ver:
                cand_ver = v
                break
    if not cand_ver:
        vs = [v.get("version") for v in trend.get("versions", []) if v.get("version")]
        cand_ver = next((v for v in vs if v != blue_ver), None) or "v0.7.9"

    # Confidence (from plan if available, else demo default)
    conf = None
    for a in (plan.get("actions") or []):
        if a.get("version") == cand_ver:
            conf = a.get("confidence")
            break
    if conf is None:
        conf = 0.85  # demo/readable default

    # Merge performance + calibration if calibration has richer metrics
    def _merge_perf_cal(ver: str) -> Dict[str, Optional[float]]:
        m = _version_metrics(trend, ver)
        c = _version_metrics(calib, ver) if calib else {}
        out = dict(m)
        # prefer explicit calibration ECE if present
        if c.get("ece") is not None:
            out["ece"] = c["ece"]
        return out

    blue = _merge_perf_cal(blue_ver)
    green = _merge_perf_cal(cand_ver)

    d_precision = _delta(green.get("precision"), blue.get("precision"))
    d_recall = _delta(green.get("recall"), blue.get("recall"))
    d_f1 = _delta(green.get("f1"), blue.get("f1"))
    d_ece = _delta(green.get("ece"), blue.get("ece"))

    classification = _classify(d_f1, d_ece, float(conf), delta_thresh, conf_thresh)

    # Build JSON summary
    delta_obj = {}
    def _fmt_delta(v: Optional[float]) -> Optional[float]:
        return None if v is None else float(f"{v:.4f}")

    delta_obj["precision"] = _fmt_delta(d_precision)
    delta_obj["recall"] = _fmt_delta(d_recall)
    delta_obj["F1"] = _fmt_delta(d_f1)
    delta_obj["ECE"] = _fmt_delta(d_ece)

    notes: List[str] = []
    if d_f1 is not None and d_f1 >= delta_thresh:
        notes.append("meets_delta_threshold")
    if d_ece is not None and d_ece <= -0.01:
        notes.append("calibration_improved")
    if not notes:
        notes.append("mixed_signals")

    result = {
        "generated_at": _iso_common(),
        "lookback_hours": lookback_h,
        "current_model": blue_ver,
        "candidate": cand_ver,
        "delta": delta_obj,
        "classification": classification,
        "confidence": float(f"{float(conf):.2f}"),
        "notes": notes,
    }
    _write_json(models / "bluegreen_promotion.json", result)

    # Visuals
    comp_png = artifacts / f"bluegreen_comparison_{cand_ver}.png"
    _plot_comparison_png(comp_png, blue, green, blue_ver, cand_ver)

    timeline_png = artifacts / "bluegreen_timeline.png"
    _plot_timeline_placeholder(timeline_png)

    # CI markdown block
    def _sym(v: Optional[float], pos="+"):
        if v is None:
            return "n/a"
        s = "−" if v < 0 else "+"
        if v == 0:
            s = "±"
        return f"{s}{abs(v):.2f}"

    md.append(f"🧪 Blue-Green Promotion Simulation ({lookback_h} h)")
    md.append(f"current → candidate: {blue_ver} → {cand_ver}")
    md.append(
        f"ΔF1 {_sym(d_f1)} | ΔECE {_sym(d_ece)} | conf {float(conf):.2f} → {classification.upper()}"
    )
    md.append("visuals: comparison + timeline (demo)")

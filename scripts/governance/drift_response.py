from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

# Matplotlib used only for exporting simple plots; Agg backend expected upstream
import matplotlib
matplotlib.use(os.getenv("MPLBACKEND", "Agg"))  # don't force if caller set it
import matplotlib.pyplot as plt  # noqa: E402


# ---------- Helpers ----------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(p: Path, default: Any) -> Any:
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default


def _write_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=False))


def _append_jsonl(p: Path, row: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


# ---------- Data structures ----------

@dataclass
class Paths:
    models_dir: Path
    logs_dir: Path
    artifacts_dir: Path


@dataclass
class Candidate:
    origin: str
    model_version: str
    current_threshold: float
    proposed_threshold: float
    delta: float
    reasons: List[str]
    buckets_used: List[str]  # ISO bucket_start strings
    backtest: Dict[str, float]
    decision: str  # "proceed" | "hold" | "observe" | "noop"
    demo: bool = False


# ---------- Policy Engine ----------

def _load_calibration_series(models_dir: Path) -> Dict[str, Any]:
    """
    We consume the existing artifact built by earlier tasks:
    models/calibration_reliability_trend.json
    Shape:
      { "series": [ { "key": "<origin or origin/version>", "points": [
            {"bucket_start": ISO, "ece": float, "brier": float, "n": int,
             "alerts": [...], "market": {...}, "social_bursts": [...] }
        ]}], "meta": {...} }
    """
    path = models_dir / "calibration_reliability_trend.json"
    return _read_json(path, {"series": [], "meta": {"demo": False}})


def _load_thresholds(models_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Threshold store: models/per_origin_thresholds.json
    Format:
      { "<origin>": {"default": 0.50, "v17": 0.62, ...}, ... }
    """
    p = models_dir / "per_origin_thresholds.json"
    return _read_json(p, {})


def _save_thresholds(models_dir: Path, store: Dict[str, Dict[str, float]]) -> None:
    _write_json(models_dir / "per_origin_thresholds.json", store)


def _key_origin_version(series_key: str) -> Tuple[str, str]:
    """
    Allow keys in series as either 'reddit' or 'reddit/v17'.
    """
    if "/" in series_key:
        origin, ver = series_key.split("/", 1)
        return origin, ver
    return series_key, "v0"


def _should_require_overlap() -> bool:
    return _env_bool("MW_DRIFT_REQUIRE_OVERLAP", False)


def _proposed_delta_step() -> float:
    # modest tighten step
    return 0.03


def _cooldown_hours() -> int:
    return _env_int("MW_DRIFT_COOLDOWN_H", 12)


def _max_shift() -> float:
    return _env_float("MW_THRESHOLD_MAX_SHIFT", 0.10)


def _min_precision() -> float:
    return _env_float("MW_THRESHOLD_MIN_PRECISION", 0.75)


def _min_labels() -> int:
    return _env_int("MW_THRESHOLD_MIN_LABELS", 10)


def _ece_high_thresh() -> float:
    return _env_float("MW_DRIFT_ECE_THRESH", 0.06)


def _grace_hours() -> int:
    return _env_int("MW_DRIFT_GRACE_H", 6)


def _min_buckets() -> int:
    return _env_int("MW_DRIFT_MIN_BUCKETS", 3)


def _action_mode() -> str:
    # "dryrun" | "apply"
    v = os.getenv("MW_DRIFT_ACTION", "dryrun").lower()
    if v not in ("dryrun", "apply"):
        return "dryrun"
    return v


def _window_hours() -> int:
    # used in plan metadata; backtests use last 72h by default
    return 72


def _demo_mode() -> bool:
    # Align with earlier sections
    return _env_bool("MW_DEMO", False)


def _passes_overlap(point: Dict[str, Any]) -> bool:
    if not _should_require_overlap():
        return True
    # overlap present if either volatility regime OR social burst overlap flagged
    alerts = point.get("alerts", [])
    if any(a in alerts for a in ("volatility_regime", "social_burst_overlap")):
        return True
    # or market object indicates high bucket
    mk = point.get("market") or {}
    if (mk.get("btc_vol_bucket") or "").lower() == "high":
        return True
    # or social_bursts non-empty
    if point.get("social_bursts"):
        return True
    return False


def _last_actions_map(logs_dir: Path) -> Dict[Tuple[str, str], datetime]:
    """
    Parse governance log to enforce cooldown per (origin, version).
    """
    path = logs_dir / "governance_actions.jsonl"
    out: Dict[Tuple[str, str], datetime] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("action") != "threshold_update":
                continue
            try:
                ts = datetime.fromisoformat(row.get("ts_utc", "").replace("Z", "+00:00"))
            except Exception:
                ts = _utcnow()
            k = (row.get("origin", "unknown"), row.get("model_version", "v0"))
            if k not in out or ts > out[k]:
                out[k] = ts
    except Exception:
        pass
    return out


def _cooldown_ok(last_ts: Optional[datetime]) -> bool:
    if not last_ts:
        return True
    return _utcnow() - last_ts >= timedelta(hours=_cooldown_hours())


def _heuristic_backtest(current_t: float, proposed_t: float, ece_values: List[float]) -> Dict[str, float]:
    """
    We don’t have a full trigger replay here; approximate a conservative outcome:

      - tighter threshold tends to reduce recall, may slightly raise precision
      - we expect small ECE improvement if drift is high, proportional to tighten step

    Return deltas (proposed - current).
    Units: deltas in absolute (not %), we convert to pp in UI.
    """
    step = max(0.0, proposed_t - current_t)
    # Clamp step for sanity
    step = min(step, 0.10)

    # Precision delta: + up to ~2.5pp per 0.03 step, scaled by recent mean ECE.
    mean_ece = sum(ece_values) / max(1, len(ece_values))
    prec_delta = 0.025 * (step / 0.03) * (0.5 + 5 * mean_ece)  # 0.5–0.65ish multiplier

    # Recall delta: negative, larger penalty than precision gain.
    rec_delta = -0.035 * (step / 0.03)

    # ECE delta: reduce by ~0.01 per 0.03 step when drift present (cap by mean_ece).
    ece_delta = -min(mean_ece, 0.01 * (step / 0.03) * 1.2)

    # F1 approx (very rough): 2 * p * r / (p + r), using small deltas assuming base ~0.75
    base_p, base_r = 0.75, 0.75
    p2, r2 = max(0.0, min(1.0, base_p + prec_delta)), max(0.0, min(1.0, base_r + rec_delta))
    f1_old = 2 * base_p * base_r / (base_p + base_r)
    f1_new = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else f1_old
    f1_delta = f1_new - f1_old

    return {
        "precision_delta": float(prec_delta),
        "recall_delta": float(rec_delta),
        "ece_delta": float(ece_delta),
        "f1_delta": float(f1_delta),
    }


def _gate_from_backtest(bt: Dict[str, float]) -> bool:
    """
    Proceed if precision improves OR stays within 0.5pp while ECE drops ≥ 0.01.
    """
    if bt.get("precision_delta", 0.0) > 0.0:
        return True
    if abs(bt.get("precision_delta", 0.0)) <= 0.005 and bt.get("ece_delta", 0.0) <= -0.01:
        return True
    return False


def _clamp_threshold(baseline: float, proposed: float) -> float:
    # Keep within max shift from baseline (±)
    max_shift = _max_shift()
    return float(max(baseline - max_shift, min(baseline + max_shift, proposed)))


def _resolve_current_threshold(store: Dict[str, Dict[str, float]], origin: str, version: str) -> Tuple[float, float]:
    """
    Returns (baseline, current), where baseline is the 'default' for origin
    (used for max shift envelope) and current is the effective threshold for version.
    """
    entry = store.get(origin, {})
    baseline = float(entry.get("default", 0.50))
    current = float(entry.get(version, entry.get("default", 0.50)))
    return baseline, current


def _detect_candidates(models_dir: Path, logs_dir: Path) -> Tuple[List[Candidate], Dict[str, Any]]:
    cal = _load_calibration_series(models_dir)
    series = cal.get("series", [])
    demo = bool(cal.get("meta", {}).get("demo", False)) or _demo_mode()

    min_b = _min_buckets()
    ece_thr = _ece_high_thresh()
    min_n = _min_labels()
    grace = _grace_hours()

    # Gather last-grace-hours window from buckets by ISO (assume hourly, already bucketed)
    now = _utcnow()
    win_start = now - timedelta(hours=grace)

    cands: List[Candidate] = []
    thresholds = _load_thresholds(models_dir)
    last_map = _last_actions_map(logs_dir)

    for s in series:
        key = s.get("key") or "unknown"
        origin, version = _key_origin_version(key)
        pts: List[Dict[str, Any]] = s.get("points") or []

        # Only consider buckets within grace horizon for “consecutive” definition
        recent = []
        for p in pts:
            try:
                bt = datetime.fromisoformat(p.get("bucket_start", "").replace("Z", "+00:00"))
            except Exception:
                continue
            if bt >= win_start:
                recent.append((bt, p))
        recent.sort(key=lambda x: x[0])

        if not recent:
            continue

        # Count high_ece buckets, requiring overlap if configured
        high_pts = []
        eces = []
        for bt, p in recent:
            ece = float(p.get("ece", 0.0))
            eces.append(ece)
            alerts = set(p.get("alerts", []))
            if ece > ece_thr and p.get("n", 0) >= min_n and _passes_overlap(p):
                high_pts.append((bt, p))

        if len(high_pts) < min_b:
            continue

        # Cooldown check
        last_ts = last_map.get((origin, version))
        if not _cooldown_ok(last_ts):
            # Skip candidate in cooldown window
            continue

        baseline, current = _resolve_current_threshold(thresholds, origin, version)
        step = _proposed_delta_step()
        proposed = _clamp_threshold(baseline, current + step)
        delta = proposed - current
        reasons = ["high_ece_persistent"]
        if _should_require_overlap():
            reasons.append("requires_overlap")
        # Add explicit overlap reason if any of the used buckets had overlap flags
        if any(("volatility_regime" in (p.get("alerts") or []) or "social_burst_overlap" in (p.get("alerts") or []))
               for _, p in high_pts):
            reasons.append("volatility_or_social_overlap")

        bt = _heuristic_backtest(current, proposed, eces)
        decision = "proceed" if _gate_from_backtest(bt) else "hold"

        cands.append(Candidate(
            origin=origin,
            model_version=version,
            current_threshold=current,
            proposed_threshold=proposed,
            delta=delta,
            reasons=reasons,
            buckets_used=[_iso(bt) for bt, _ in high_pts],
            backtest=bt,
            decision=decision,
            demo=demo,
        ))

    meta = {
        "generated_at": _iso(now),
        "window_hours": _window_hours(),
        "grace_hours": _grace_hours(),
        "min_buckets": _min_buckets(),
        "ece_threshold": _ece_high_thresh(),
        "action_mode": _action_mode(),
        "demo": demo,
    }
    return cands, meta


def _apply_if_allowed(paths: Paths, c: Candidate, thresholds: Dict[str, Dict[str, float]]) -> None:
    """
    Apply update and append governance log line if MW_DRIFT_ACTION=apply and decision=proceed.
    Always log a row (with decision).
    """
    mode = _action_mode()
    # ensure container
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)

    old_val = c.current_threshold
    new_val = c.current_threshold

    if mode == "apply" and c.decision == "proceed":
        entry = thresholds.setdefault(c.origin, {})
        baseline = float(entry.get("default", 0.50))
        # Respect max shift envelope versus baseline
        candidate_val = _clamp_threshold(baseline, c.proposed_threshold)
        entry[c.model_version] = candidate_val
        thresholds[c.origin] = entry
        _save_thresholds(paths.models_dir, thresholds)
        new_val = candidate_val

    # governance log row
    log_row = {
        "ts_utc": _iso(_utcnow()),
        "action": "threshold_update",
        "mode": mode,
        "origin": c.origin,
        "model_version": c.model_version,
        "old_threshold": round(old_val, 6),
        "new_threshold": round(new_val, 6),
        "delta": round(new_val - old_val, 6),
        "policy": "drift_response_v1",
        "decision": c.decision,
        "context": {
            "reasons": c.reasons,
            "buckets_used": c.buckets_used,
            "backtest_precision_delta": round(c.backtest.get("precision_delta", 0.0), 6),
            "backtest_ece_delta": round(c.backtest.get("ece_delta", 0.0), 6),
            "backtest_recall_delta": round(c.backtest.get("recall_delta", 0.0), 6),
        },
        "demo": c.demo,
    }
    _append_jsonl(paths.logs_dir / "governance_actions.jsonl", log_row)


def _plot_timeline(paths: Paths, cal: Dict[str, Any], cands: List[Candidate]) -> None:
    """
    Simple ECE timeline with action markers.
    """
    try:
        series = cal.get("series", [])
        fig, ax = plt.subplots(figsize=(8, 3))
        # Plot each series ECE as a line
        for s in series:
            key = s.get("key", "series")
            xs, ys = [], []
            for p in (s.get("points") or []):
                try:
                    xs.append(datetime.fromisoformat(p["bucket_start"].replace("Z", "+00:00")))
                    ys.append(float(p.get("ece", 0.0)))
                except Exception:
                    continue
            if xs and ys:
                ax.plot(xs, ys, label=key)
        # Action markers
        for c in cands:
            for iso_ts in c.buckets_used:
                try:
                    t = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
                except Exception:
                    continue
                ax.scatter([t], [0.0], marker="^")  # bottom markers
        ax.set_title("Drift Response — ECE timeline (72h)")
        ax.set_xlabel("time (UTC)")
        ax.set_ylabel("ECE")
        if series:
            ax.legend(loc="upper right", fontsize="small")
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        out = paths.artifacts_dir / "drift_response_timeline.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
    except Exception:
        # best-effort; do not break pipeline
        pass


def _plot_backtest_bar(paths: Paths, c: Candidate) -> None:
    """
    Bar chart of before/after deltas (precision, recall, ECE, F1).
    """
    try:
        metrics = ["precision", "recall", "ece", "f1"]
        deltas = [
            c.backtest.get("precision_delta", 0.0),
            c.backtest.get("recall_delta", 0.0),
            c.backtest.get("ece_delta", 0.0),
            c.backtest.get("f1_delta", 0.0),
        ]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(metrics, deltas)
        ax.set_title(f"Backtest Δ — {c.origin}/{c.model_version}")
        ax.axhline(0.0, linewidth=1)
        for i, v in enumerate(deltas):
            ax.text(i, v, f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        out = paths.artifacts_dir / f"drift_response_backtest_{c.origin}_{c.model_version}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
    except Exception:
        pass


def run_policy(paths: Paths) -> Path:
    """
    Entry-point: detect candidates, backtest, (optionally) apply, export plan + plots.

    Returns the path to the plan JSON.
    """
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    cal = _load_calibration_series(paths.models_dir)
    cands, meta = _detect_candidates(paths.models_dir, paths.logs_dir)

    # Apply decisions and log; generate per-candidate backtest plots
    thresholds = _load_thresholds(paths.models_dir)
    for c in cands:
        _apply_if_allowed(paths, c, thresholds)
        _plot_backtest_bar(paths, c)

    # Save plan
    plan = {
        **meta,
        "candidates": [
            {
                "origin": c.origin,
                "model_version": c.model_version,
                "current_threshold": round(c.current_threshold, 6),
                "proposed_threshold": round(c.proposed_threshold, 6),
                "delta": round(c.delta, 6),
                "reasons": c.reasons,
                "buckets_used": c.buckets_used,
                "backtest": {
                    "precision_delta": round(c.backtest.get("precision_delta", 0.0), 6),
                    "ece_delta": round(c.backtest.get("ece_delta", 0.0), 6),
                    "recall_delta": round(c.backtest.get("recall_delta", 0.0), 6),
                    "f1_delta": round(c.backtest.get("f1_delta", 0.0), 6),
                },
                "decision": c.decision,
            }
            for c in cands
        ],
    }
    plan_path = paths.models_dir / "drift_response_plan.json"
    _write_json(plan_path, plan)

    # Export a consolidated timeline plot (best-effort)
    _plot_timeline(paths, cal, cands)

    return plan_path

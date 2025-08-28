# src/ml/thresholds.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.paths import LOGS_DIR, MODELS_DIR
from src.analytics.origin_utils import normalize_origin as _norm
from src.ml.infer import infer_score


# ---------- helpers ----------
def _load_jsonl(path: Path) -> List[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except Exception:
        return []


def _parse_ts(v) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        try:
            s = str(v)
            s = s[:-1] + "+00:00" if s.endswith("Z") else s
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            return None


def _label_has_trigger_between(triggers: List[dict], origin: str, t0: datetime, t1: datetime) -> int:
    o = _norm(origin)
    for r in triggers:
        if _norm(r.get("origin", "")) != o:
            continue
        ts = _parse_ts(r.get("timestamp"))
        if ts and t0 < ts <= t1:
            return 1
    return 0


def _compute_pr_threshold(y_true: np.ndarray, p: np.ndarray, target_precision: float) -> float | None:
    """
    Return the *probability* cutoff that achieves >= target precision,
    preferring the smallest cutoff (higher recall) among qualifying points.
    """
    if y_true.size == 0 or p.size == 0:
        return None
    try:
        prec, rec, thr = precision_recall_curve(y_true, p)
        # precision_recall_curve returns len(thr) == len(prec) - 1
        best = None
        for pr, rc, th in zip(prec[:-1], rec[:-1], thr):
            if pr >= target_precision:
                if best is None or th < best[0] or (th == best[0] and rc > best[1]):
                    best = (float(th), float(rc))
        return best[0] if best else None
    except Exception:
        return None


def _unique_origins(flags: List[dict]) -> List[str]:
    out = sorted({ _norm(r.get("origin","unknown")) for r in flags })
    # keep only known sources if present
    known = [o for o in out if o in ("twitter","reddit","rss_news")]
    return known or out or ["twitter","reddit","rss_news"]


def _demo_seed_thresholds() -> Dict[str, Dict[str, float]]:
    # Demo/fallback values (match what you showed in CI while no data)
    return {
        "twitter": {"p70": 2.50, "p80": 3.00},
        "reddit":  {"p70": 2.60, "p80": 3.20},
        "rss_news":{"p70": 2.40, "p80": 2.90},
    }


# ---------- public API ----------
def fit_and_write_thresholds(
    *,
    days: int = 7,
    interval: str = "hour",
    horizon_hours: int = 6,
    min_samples: int = 40,
    targets: Tuple[float, float] = (0.70, 0.80),
    models_dir: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Build per-origin *probability* thresholds that hit the requested precision targets.

    Writes: models/per_origin_thresholds.json

    Returns a simple mapping: {origin: {"p70": thr1, "p80": thr2}}
    (If not enough data, we fall back to seeded demo values.)
    """
    models_dir = models_dir or MODELS_DIR
    out_path = models_dir / "per_origin_thresholds.json"

    # Load logs
    flags = _load_jsonl(LOGS_DIR / "retraining_log.jsonl")
    triggers = _load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")

    # Build hourly buckets for the past `days`
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(days * 24, 0, -1)]

    origins = _unique_origins(flags)
    per_origin: Dict[str, Dict[str, float]] = {}

    for origin in origins:
        ps: List[float] = []
        ys: List[int] = []

        for t in buckets:
            # Model probability at the bucket
            try:
                p = float(infer_score({"origin": origin, "timestamp": t.isoformat()}).get("prob_trigger_next_6h", 0.0))
            except Exception:
                p = 0.0
            # Label: did a trigger occur in the next `horizon_hours`?
            y = _label_has_trigger_between(triggers, origin, t, t + timedelta(hours=horizon_hours))
            ps.append(p); ys.append(y)

        p_arr = np.asarray(ps, dtype=float)
        y_arr = np.asarray(ys, dtype=int)

        # Require some positives and negatives to learn a useful cutoff
        if y_arr.sum() < 3 or (y_arr.size - y_arr.sum()) < 3 or y_arr.size < min_samples:
            continue

        p70 = _compute_pr_threshold(y_arr, p_arr, targets[0]) or float("nan")
        p80 = _compute_pr_threshold(y_arr, p_arr, targets[1]) or float("nan")
        per_origin[origin] = {"p70": p70, "p80": p80}

    # Fallback to demo if nothing computed
    if not per_origin:
        per_origin = _demo_seed_thresholds()

    # Persist a compact file (plus a few useful headers)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "horizon_hours": horizon_hours,
        "targets": {"p70": targets[0], "p80": targets[1]},
        "per_origin": per_origin,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    # Return the simple mapping shape callers expect
    return per_origin


def load_per_origin_thresholds(models_dir: Path | None = None) -> Dict[str, Dict[str, float]]:
    """
    Read models/per_origin_thresholds.json and return {origin: {"p70": x, "p80": y}}.
    If missing or malformed, return demo-seeded values.
    """
    models_dir = models_dir or MODELS_DIR
    path = models_dir / "per_origin_thresholds.json"
    try:
        with path.open("r") as f:
            blob = json.load(f)
        per_origin = blob.get("per_origin") or {}
        # Accept both flattened shape and nested shape
        if per_origin:
            return {k: {"p70": float(v.get("p70", np.nan)), "p80": float(v.get("p80", np.nan))}
                    for k, v in per_origin.items()}
    except Exception:
        pass
    return _demo_seed_thresholds()


__all__ = ["fit_and_write_thresholds", "load_per_origin_thresholds"]


if __name__ == "__main__":
    # Convenience CLI: python -m src.ml.thresholds
    res = fit_and_write_thresholds()
    print(json.dumps(res, indent=2))
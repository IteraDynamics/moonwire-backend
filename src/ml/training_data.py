# src/ml/training_data.py
from __future__ import annotations

import json, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from src.paths import MODELS_DIR

# Files
TRIGGER_HISTORY_PATH = MODELS_DIR / "trigger_history.jsonl"
LABEL_FEEDBACK_PATH  = MODELS_DIR / "label_feedback.jsonl"
TRAINING_DATA_LOG    = MODELS_DIR / "training_data.jsonl"


# ----------- helpers -----------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            # ignore bad lines
            pass
    return out


def _parse_ts(v) -> Optional[datetime]:
    if v is None:
        return None
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        try:
            s = str(v)
            s = s[:-1] + "+00:00" if s.endswith("Z") else s
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            return None


def _iso_utc(dt: datetime) -> str:
    # ISO8601 Z
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _nearest_trigger(
    triggers: List[Dict[str, Any]],
    origin: str,
    ts_label: datetime,
    window_minutes: int,
) -> Optional[Dict[str, Any]]:
    """Find the closest trigger by time within +/- window for the same origin."""
    if not triggers:
        return None
    w = timedelta(minutes=max(1, int(window_minutes)))
    best: Tuple[timedelta, Dict[str, Any]] | None = None
    for t in triggers:
        if (t.get("origin") or "").lower() != (origin or "").lower():
            continue
        ts_t = _parse_ts(t.get("timestamp"))
        if not ts_t:
            continue
        delta = abs(ts_t - ts_label)
        if delta <= w:
            if best is None or delta < best[0]:
                best = (delta, t)
    return best[1] if best else None


def _existing_keys(path: Path) -> set[Tuple[str, str]]:
    keys: set[Tuple[str, str]] = set()
    if not path.exists():
        return keys
    for ln in path.read_text().splitlines():
        try:
            r = json.loads(ln)
            keys.add((r.get("origin"), r.get("timestamp")))
        except Exception:
            pass
    return keys


# ----------- public API -----------
def collect_training_rows(window_minutes: int = 5) -> List[Dict[str, Any]]:
    """
    Join label_feedback.jsonl with trigger_history.jsonl (same origin, closest timestamp within window).
    Append unique rows to training_data.jsonl.
    Returns the list of rows newly appended.
    """
    LABELS  = _load_jsonl(LABEL_FEEDBACK_PATH)
    TRIGS   = _load_jsonl(TRIGGER_HISTORY_PATH)
    if not LABELS or not TRIGS:
        return []

    already = _existing_keys(TRAINING_DATA_LOG)
    out_rows: List[Dict[str, Any]] = []

    for fb in LABELS:
        ts_fb = _parse_ts(fb.get("timestamp"))
        origin = (fb.get("origin") or "unknown").lower()
        if not ts_fb:
            continue
        match = _nearest_trigger(TRIGS, origin, ts_fb, window_minutes=window_minutes)
        if not match:
            continue

        # Prefer features from trigger history if present; fall back to feedback payload (optional)
        feats = (
            match.get("features")
            or fb.get("features")
            or {}
        )

        row = {
            "timestamp": _iso_utc(ts_fb),
            "origin": origin,
            "features": feats,
            "label": bool(fb.get("label", False)),
            # Optional context for future analysis/versioning
            "adjusted_score": match.get("adjusted_score"),
            "threshold": match.get("threshold_used") or match.get("threshold"),
            "decision": match.get("decision"),
            "volatility_regime": match.get("volatility_regime"),
            "drifted_features": match.get("drifted_features") or [],
            "top_contributors": match.get("top_contributors") or [],
            "model_version": (match.get("model_version") or None),
        }

        key = (row["origin"], row["timestamp"])
        if key not in already:
            out_rows.append(row)
            already.add(key)

    if out_rows:
        TRAINING_DATA_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TRAINING_DATA_LOG.open("a") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")

    return out_rows


# CLI entrypoint for CI/backfills
if __name__ == "__main__":
    try:
        w = int(os.getenv("TRAINING_JOIN_WINDOW_MIN", "5"))
    except Exception:
        w = 5
    appended = collect_training_rows(window_minutes=w)
    print(f"[training_data] appended {len(appended)} rows (window={w}m)")

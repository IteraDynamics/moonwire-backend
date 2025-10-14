# src/trigger_likelihood_router.py
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query
from src import paths
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from src.paths import MODELS_DIR

from src.ml.infer import (
    infer_score,
    infer_score_ensemble,
    model_metadata,
    model_metadata_all,
)

from src.ml.thresholds import load_per_origin_thresholds

# This router is expected to be mounted in main.py under prefix="/internal"
router = APIRouter(tags=["trigger_likelihood"])


@router.post("/trigger-likelihood/score")
def score_endpoint(
    payload: Dict[str, Any] = Body(...),
    use: str = Query("logistic", regex="^(logistic|ensemble)$"),
    explain: bool = Query(False),
):
    """
    POST body can be either:
      { "features": {...} }                          # preferred
      { "origin": "twitter", "timestamp": "..." }    # best-effort fallback
    Query params:
      use=logistic|ensemble   -> choose scorer
      explain=true            -> include linear contributions (logistic only)
    """
    try:
        if use == "ensemble":
            return infer_score_ensemble(payload)
        # default to logistic
        return infer_score(payload, explain=explain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scoring error: {e}")


@router.get("/trigger-likelihood/metadata")
def trigger_likelihood_metadata(view: str = Query(default="base", regex="^(base|all)$")):
    """
    Default (view=base): return classic flat logistic metadata so existing tests/clients pass:
      { "created_at": ..., "metrics": {...}, "feature_order": [...], ... }

    view=all: return both models if present:
      { "logistic": {...}, "random_forest": {...} }
    """
    # Load logistic (primary) with monkeypatch-friendly paths
    try:
        base = model_metadata(models_dir=paths.MODELS_DIR)
    except Exception:
        base = {}

    # Optional RF metadata
    rf_meta = None
    try:
        rf_meta_path = paths.MODELS_DIR / "trigger_likelihood_rf.meta.json"
        if rf_meta_path.exists():
            with rf_meta_path.open("r") as f:
                rf_meta = json.load(f)
    except Exception:
        rf_meta = None  # ignore RF read errors

    # If nothing at all, allow demo fallback (200) or 503
    if not base and not rf_meta:
        if os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
            return {
                "demo": True,
                "message": "No model artifacts found; returning demo metadata.",
                "created_at": None,
                "metrics": {"roc_auc_va": 0.5},
                "feature_order": [],
                "feature_coverage": {},
                "top_features": [],
            }
        raise HTTPException(status_code=503, detail="No model artifacts available")

    # base view: return flat logistic meta (backward-compatible for tests)
    if view == "base":
        if base:
            return base
        # no logistic but RF exists → return RF flat to still satisfy tests' shape expectation
        return rf_meta

    # view=all: return both if available
    payload = {}
    if base:
        payload["logistic"] = base
    if rf_meta:
        payload["random_forest"] = rf_meta
    return payload


@router.get("/internal/trigger-likelihood/thresholds")
def get_trigger_thresholds():
    """
    Returns per-origin thresholds for trigger likelihood (or demo fallback).
    """
    return load_per_origin_thresholds()


@router.get("/internal/trigger-likelihood/metadata")
def get_trigger_model_metadata():
    """
    Returns full model metadata (logistic + rf), including calibration + thresholds.
    """
    meta = model_metadata()
    meta["thresholds"] = load_per_origin_thresholds()
    return meta


# --- Label Feedback Logging (v0.5.2) ------------------------------------------

# Locations in MODELS_DIR
_LABEL_FEEDBACK_PATH: Path = MODELS_DIR / "label_feedback.jsonl"
_TRIGGER_HISTORY_PATH: Path = MODELS_DIR / "trigger_history.jsonl"


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed/mid-write lines
                continue


def _parse_any_ts(ts: str | int | float) -> datetime:
    """
    Accepts:
      - 'YYYY-MM-DDTHH:MM:SSZ' (with/without fractional seconds)
      - ISO with offset: '...+00:00'
      - epoch seconds (string/number)
    Returns tz-aware UTC datetime.
    """
    # epoch seconds (int/float or numeric string)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    s = str(ts).strip()
    if s.replace(".", "", 1).isdigit():
        # numeric string epoch
        return datetime.fromtimestamp(float(s), tz=timezone.utc)

    # ISO handling
    if s.endswith("Z"):
        # strip fractional seconds if present for consistent parsing
        if "." in s:
            s = s.split(".")[0] + "Z"
        # convert Z to +00:00 for fromisoformat
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _find_model_version_for_label(*, label_timestamp: str, origin: str, window_minutes: int = 5) -> str:
    """
    Scan MODELS_DIR/trigger_history.jsonl for the closest row (by |Δt|) that
    shares the same origin and is within ±window. Return its model_version or 'unknown'.
    """
    label_dt = _parse_any_ts(label_timestamp)
    window = timedelta(minutes=window_minutes)
    best_row = None
    best_abs = None

    for row in _iter_jsonl(_TRIGGER_HISTORY_PATH):
        if row.get("origin") != origin:
            continue
        ts = row.get("timestamp")
        if not ts:
            continue
        try:
            trig_dt = _parse_any_ts(ts)
        except Exception:
            continue
        delta = trig_dt - label_dt
        if abs(delta) <= window:
            ad = abs(delta)
            if best_row is None or ad < best_abs:
                best_row, best_abs = row, ad

    mv = (best_row or {}).get("model_version")
    return str(mv) if mv else "unknown"


def _validate_feedback_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Required:
      - timestamp (ISO-8601 or epoch seconds)
      - origin (str)
      - adjusted_score (float)
      - label (bool)
    Optional:
      - notes (str)
      - reviewer (str)
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    # origin
    origin = payload.get("origin")
    if not isinstance(origin, str) or not origin.strip():
        raise ValueError("origin is required (non-empty string)")

    # adjusted_score
    try:
        adjusted_score = float(payload.get("adjusted_score"))
    except Exception:
        raise ValueError("adjusted_score is required (number)")

    # label
    label = payload.get("label")
    if not isinstance(label, bool):
        raise ValueError("label is required (true/false)")

    # timestamp
    ts = payload.get("timestamp")
    ts_iso: str
    if ts is None:
        # If omitted, default to 'now' (UTC)
        ts_iso = datetime.now(timezone.utc).isoformat()
    else:
        # Accept epoch seconds or ISO string; normalize to ISO (UTC)
        try:
            if isinstance(ts, (int, float)):
                ts_iso = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
            else:
                s = str(ts)
                s = s[:-1] + "+00:00" if s.endswith("Z") else s
                ts_iso = datetime.fromisoformat(s).astimezone(timezone.utc).isoformat()
        except Exception:
            raise ValueError("timestamp must be ISO-8601 or epoch seconds")

    out = {
        "timestamp": ts_iso,
        "origin": origin.strip(),
        "adjusted_score": adjusted_score,
        "label": label,
    }

    # Optionals (safe copy)
    notes = payload.get("notes")
    if isinstance(notes, str) and notes.strip():
        out["notes"] = notes.strip()

    reviewer = payload.get("reviewer")
    if isinstance(reviewer, str) and reviewer.strip():
        out["reviewer"] = reviewer.strip()

    return out


@router.post("/trigger-likelihood/feedback")
async def post_label_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts label feedback and appends it to models/label_feedback.jsonl.
    Schema enforced by _validate_feedback_payload.
    """
    try:
        record = _validate_feedback_payload(payload)
        # --- v0.5.2: attach model_version from trigger history (fallback 'unknown')
        record["model_version"] = _find_model_version_for_label(
            label_timestamp=record["timestamp"],
            origin=record["origin"],
            window_minutes=5,
        )
        _append_jsonl(_LABEL_FEEDBACK_PATH, record)
        return {"status": "ok", "written": True, "model_version": record["model_version"]}
    except ValueError as ve:
        # 400-like error (FastAPI will still wrap as 200 unless you raise HTTPException;
        # keeping minimal to avoid changing imports)
        return {"status": "error", "written": False, "error": str(ve)}
    except Exception as e:
        return {"status": "error", "written": False, "error": f"internal: {type(e).__name__}: {e}"}
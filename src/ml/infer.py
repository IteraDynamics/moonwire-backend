from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import joblib
import numpy as np

from src.ml.feature_builder import build_feature_row_for, FEATURE_ORDER
import src.paths as paths  # use live module attributes so monkeypatch works

MODEL_NAME = "trigger_likelihood_v0"


def _load() -> tuple[object | None, Dict[str, Any]]:
    model_path = paths.MODELS_DIR / f"{MODEL_NAME}.joblib"
    meta_path = paths.MODELS_DIR / f"{MODEL_NAME}.meta.json"
    if not (model_path.exists() and meta_path.exists()):
        return None, {}
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return model, meta


def _align_features(feats: Dict[str, float], order: List[str]) -> np.ndarray:
    return np.array([float(feats.get(k, 0.0)) for k in order], dtype=float)


def score(body: Dict[str, Any]) -> Dict[str, Any]:
    model, meta = _load()
    if model is None:
        if os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
            return {"prob_trigger_next_6h": 0.42, "contributions": [], "demo": True}
        raise FileNotFoundError("Model artifacts not found.")

    order = meta.get("feature_order", FEATURE_ORDER)
    if "features" in body:
        feats = {k: float(v) for k, v in body["features"].items()}
    else:
        origin = body.get("origin")
        ts_raw = body.get("timestamp")
        if not origin or not ts_raw:
            raise ValueError("Provide either {'features': {...}} or {'origin','timestamp'}.")
        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).astimezone(timezone.utc)

        # Build paths from CURRENT LOGS_DIR (monkeypatch-friendly)
        flags_path = paths.LOGS_DIR / "retraining_log.jsonl"
        triggers_path = paths.LOGS_DIR / "retraining_triggered.jsonl"

        feats, _ = build_feature_row_for(
            flags_path,
            triggers_path,
            origin=origin,
            ts=ts,
            interval="hour",
        )

    x = _align_features(feats, order).reshape(1, -1)
    proba = float(model.predict_proba(x)[0, 1])

    # contributions = coef * value + intercept
    coef = getattr(model, "coef_", None)
    intercept = float(getattr(model, "intercept_", [0.0])[0])
    contribs = []
    if coef is not None:
        for name, val, c in zip(order, x.flatten().tolist(), coef[0].tolist()):
            contribs.append({"name": name, "value": float(val), "coef": float(c), "contrib": float(val * c)})
        contribs.append({"name": "intercept", "value": 1.0, "coef": intercept, "contrib": intercept})

    return {
        "prob_trigger_next_6h": proba,
        "contributions": contribs,
        "features_used": {k: float(feats.get(k, 0.0)) for k in order},
        "demo": bool(meta.get("demo", False)),
        "model_meta": {"model_name": meta.get("model_name"), "created_at": meta.get("created_at")},
    }


def metadata() -> Dict[str, Any]:
    _, meta = _load()
    if not meta and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        return {"model_name": MODEL_NAME, "feature_order": FEATURE_ORDER, "demo": True}
    if not meta:
        raise FileNotFoundError("Metadata not found.")
    return meta
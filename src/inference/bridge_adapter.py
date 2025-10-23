# src/inference/bridge_adapter.py
from __future__ import annotations
import os, json, time
from pathlib import Path
import numpy as np

# Prefer your *inference* helpers if they exist
# Fallback to the training runner if needed.
_MODEL_LOADER_TRIED = False
_PREDICT_TRIED = False

def _import_infer_primitives():
    """
    Best-effort import of model loader + predictor from your codebase.
    Tries src/ml/infer.py first (exists in your repo), then scripts/ml/model_runner.py,
    then scripts/ml/train_predict.py. Returns (load_model, predict_fn, FEATURES).
    """
    global _MODEL_LOADER_TRIED, _PREDICT_TRIED
    # 1) src/ml/infer.py
    try:
        from src.ml.infer import load_model as _lm, predict_signal as _ps, FEATURES as _F
        _MODEL_LOADER_TRIED = _PREDICT_TRIED = True
        return _lm, _ps, _F
    except Exception:
        pass
    # 2) scripts/ml/model_runner.py
    try:
        from scripts.ml.model_runner import load_model as _lm, predict_signal as _ps, FEATURES as _F
        _MODEL_LOADER_TRIED = _PREDICT_TRIED = True
        return _lm, _ps, _F
    except Exception:
        pass
    # 3) scripts/ml/train_predict.py (some repos expose helpers there)
    try:
        from scripts.ml.train_predict import load_model as _lm, predict_signal as _ps, FEATURES as _F
        _MODEL_LOADER_TRIED = _PREDICT_TRIED = True
        return _lm, _ps, _F
    except Exception:
        pass
    raise ImportError("Could not import load_model/predict_signal/FEATURES from known locations.")

ROOT = Path(".")
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
SHADOW_LOG = LOGS / "signal_inference_shadow.jsonl"

def _latest_feature_vector(symbol: str):
    """
    Build the most recent features row for `symbol` using your training feature pipeline.
    Matches the exact FEATURES list used by the model.
    """
    from scripts.ml.data_loader import load_prices
    from scripts.ml.feature_builder import build_features

    # Keep lookback modest for speed; builder computes rolling windows internally.
    prices = load_prices([symbol], lookback_days=int(os.getenv("MW_ML_LOOKBACK_DAYS", "30")))
    frames = build_features(prices)
    df = frames.get(symbol.upper())
    if df is None or df.empty:
        return None

    # Import FEATURES from the same place as the model
    _, _, FEATURES = _import_infer_primitives()

    # Ensure all features exist
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None
    row = df.iloc[-1][FEATURES].astype(float).values
    if not np.all(np.isfinite(row)):
        return None
    return row

def run_bridge_shadow(symbol: str):
    """
    When MW_INFERENCE_BRIDGE=shadow|on:
      - loads active model from models/current/
      - builds latest feature vector for symbol
      - predicts and logs to logs/signal_inference_shadow.jsonl
    Returns dict or None.
    """
    mode = os.getenv("MW_INFERENCE_BRIDGE", "off").lower()
    if mode not in {"shadow", "on"}:
        return None

    try:
        load_model, predict_signal, FEATURES = _import_infer_primitives()
    except Exception:
        return None

    model_dir = Path(os.getenv("MW_MODEL_DIR", "models/current"))
    try:
        model, meta = load_model(model_dir)
    except Exception:
        return None
    if model is None:
        return None

    x = _latest_feature_vector(symbol)
    if x is None:
        return None

    try:
        pred = predict_signal(model, x.reshape(1, -1))
    except Exception:
        return None

    rec = {
        "ts": int(time.time()),
        "symbol": symbol.upper(),
        # keep keys generic; adapt to whatever your predict_signal returns
        "y_pred": int(pred.get("y_pred", 1 if pred.get("p_long", 0.5) >= 0.5 else 0)),
        "p_long": float(pred.get("p_long", pred.get("confidence", 0.5))),
        "mode": mode,
        "features_used": FEATURES,
        "model_meta": meta if isinstance(meta, dict) else {"meta": str(meta)},
    }
    with SHADOW_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

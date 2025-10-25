# scripts/perf/backfill_ml_shadow.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import joblib

# Reuse your training pipeline pieces
from scripts.ml.data_loader import load_prices
from scripts.ml.feature_builder import build_features

BUNDLE_DIR = Path(os.getenv("MW_BUNDLE_DIR", "models/current"))
SHADOW_LOG = Path("logs/signal_inference_shadow.jsonl")
GOV_PATH   = Path("models/governance_params.json")

def _utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()

def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _load_bundle():
    man_p   = BUNDLE_DIR / "manifest.json"
    feats_p = BUNDLE_DIR / "features.json"
    model_p = BUNDLE_DIR / "model.joblib"

    if not (man_p.exists() and model_p.exists()):
        raise FileNotFoundError(f"bundle incomplete in {BUNDLE_DIR}")
    manifest = json.loads(man_p.read_text(encoding="utf-8"))

    # features.json may be a list or {"feature_order":[...]}
    feat_order = None
    if feats_p.exists():
        fj = json.loads(feats_p.read_text(encoding="utf-8"))
        if isinstance(fj, list):
            feat_order = fj
        elif isinstance(fj, dict) and "feature_order" in fj:
            feat_order = fj["feature_order"]

    # fallback to manifest
    if not feat_order:
        feat_order = manifest.get("feature_order") or manifest.get("features") or []
    if not isinstance(feat_order, list) or not feat_order:
        raise RuntimeError("feature list missing in bundle")

    model = joblib.load(model_p)
    return model, [str(f) for f in feat_order], manifest

def _governance_for(sym: str) -> Dict[str, Any]:
    data = _read_json(GOV_PATH, {})
    row = data.get(sym, {})
    return {
        "conf_min": float(row.get("conf_min", 0.60)),
        "debounce_min": int(row.get("debounce_min", 15)),
    }

def _should_include(p_long: float, gov: Dict[str, Any], include_all: bool, conf_override: float | None) -> bool:
    if include_all:
        return True
    thr = float(conf_override) if conf_override is not None else float(gov.get("conf_min", 0.60))
    return p_long >= thr

def main():
    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Inputs via env (so Actions can pass UI inputs)
    symbols_env = os.getenv("MW_BACKFILL_SYMBOLS", "BTC,ETH,SOL")
    symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
    lookback_days = int(os.getenv("MW_BACKFILL_DAYS", "180"))
    include_all = str(os.getenv("MW_BACKFILL_INCLUDE_ALL", "0")).lower() in {"1","true","yes"}
    conf_override = os.getenv("MW_BACKFILL_CONF_MIN_OVERRIDE")
    conf_override = float(conf_override) if conf_override not in (None, "",) else None

    model, feat_order, manifest = _load_bundle()

    written = 0
    for sym in symbols:
        # Build full historical feature frame for this symbol
        prices = load_prices([sym], lookback_days=lookback_days)
        feats_map = build_features(prices)
        df = feats_map.get(sym)
        if df is None or df.empty:
            print(f"[backfill] WARN: no features for {sym}")
            continue

        # Vectorize all rows at once in the bundle feature order
        X = np.array([[float(row.get(k, 0.0) or 0.0) for k in feat_order] for row in df[feat_order].to_dict("records")], dtype=float)

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            z = model.decision_function(X).astype(float).ravel()
            probs = 1.0 / (1.0 + np.exp(-z))
        else:
            raise RuntimeError("model has neither predict_proba nor decision_function")

        gov = _governance_for(sym)

        with SHADOW_LOG.open("a", encoding="utf-8") as f:
            for p_long, ts in zip(probs, df["ts"].tolist()):
                direction = "long" if float(p_long) >= 0.5 else "short"
                if not _should_include(float(p_long), gov, include_all, conf_override):
                    continue
                rec = {
                    "symbol": sym,
                    "reason": "ml_backfill",
                    "ml_ok": True,
                    "ml_dir": direction,
                    "ml_conf": float(p_long),
                    "gov": gov,
                    "ts": ts if isinstance(ts, str) else _utc(ts),
                }
                f.write(json.dumps(rec) + "\n")
                written += 1

        print(f"[backfill] {sym}: wrote {written} rows so far")

    print(f"[backfill] DONE. total_rows_written={written}, log={SHADOW_LOG}")

if __name__ == "__main__":
    main()
# scripts/ml/train_predict.py
from __future__ import annotations
"""
End-to-end training + prediction + threshold tuning + artifact write-out.

- Imports features from .features (because features.py lives in scripts/ml/).
- Falls back to absolute imports if needed.
- Writes the artifacts tests expect:
  * models/ml_model_manifest.json
  * models/backtest_summary.json
  * models/signal_thresholds.json
  * artifacts/ml_roc_pr_curve.png
  * artifacts/bt_equity_curve.png
- Optionally writes artifacts/data_provenance.json if _provenance is present.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------- resolve repo paths ----------
import sys
FALLBACK_ROOT = Path(__file__).resolve().parents[2]
try:
    from scripts.paths import ROOT, MODELS_DIR, ARTIFACTS_DIR  # type: ignore
except Exception:
    ROOT = FALLBACK_ROOT
    MODELS_DIR = ROOT / "models"
    ARTIFACTS_DIR = ROOT / "artifacts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- imports (relative first) ----------
try:
    # when invoked as 'scripts.ml.train_predict', __package__ == 'scripts.ml'
    from .data_loader import load_prices
    from .features import build_features, label_next_horizon, walk_forward_splits
    from .model_runner import train_model, predict_proba
    from .tuner import tune_thresholds
    try:
        from ._provenance import detect_provenance  # optional
    except Exception:
        detect_provenance = None
except Exception:
    # absolute fallback with a tiny sys.path shim
    if str(FALLBACK_ROOT) not in sys.path:
        sys.path.insert(0, str(FALLBACK_ROOT))
    from scripts.ml.data_loader import load_prices  # type: ignore
    from scripts.ml.features import (  # type: ignore
        build_features, label_next_horizon, walk_forward_splits
    )
    from scripts.ml.model_runner import train_model, predict_proba  # type: ignore
    from scripts.ml.tuner import tune_thresholds  # type: ignore
    try:
        from scripts.ml._provenance import detect_provenance  # type: ignore
    except Exception:
        detect_provenance = None

# ---------- helpers ----------
def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _ensure_placeholder_plot(path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title(title)
    plt.plot([0, 1], [0, 1])
    plt.savefig(path)
    plt.close()

def _manifest(model_name: str, symbols: List[str], horizon_h: int) -> Dict:
    return {
        "version": "v0.9.1",
        "model": model_name,
        "symbols": symbols,
        "horizon_h": horizon_h,
    }

# ---------- main ----------
def main() -> None:
    symbols = ["BTC", "ETH", "SOL"]
    lookback_days = int(os.getenv("MW_LOOKBACK_DAYS", "90"))
    horizon_h = int(os.getenv("MW_HORIZON_H", "1"))
    model_name = os.getenv("MW_MODEL", "logreg")

    # Load + features
    prices = load_prices(symbols, lookback_days=lookback_days)
    feats = build_features(prices)

    dfs: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = label_next_horizon(feats[s], horizon_h=horizon_h)
        X = df[[
            "r_1h","r_3h","r_6h","vol_6h","atr_14h","sma_gap","high_vol","social_score"
        ]].values
        y = df["y_long"].values

        # simple single split to keep runtime low
        for tr_ix, te_ix in walk_forward_splits(df, n_splits=1, train_days=30, test_days=7):
            if len(tr_ix) < 30 or len(te_ix) < 7:
                continue
            m = train_model(X[tr_ix], y[tr_ix], model_type=model_name)
            p = predict_proba(m, X)
            dfs[s] = pd.DataFrame({"ts": df["ts"], "p_long": p})
            break

    # Tune thresholds + backtest summary
    res = tune_thresholds(dfs, prices)

    # Artifacts required by tests
    _write_json(MODELS_DIR / "signal_thresholds.json", res.get("params", res))
    _write_json(MODELS_DIR / "backtest_summary.json", {
        "aggregate": res.get("aggregate", res.get("agg", {})),
        "per_symbol": res.get("per_symbol", {}),
    })
    _write_json(MODELS_DIR / "ml_model_manifest.json",
                _manifest(model_name, symbols, horizon_h))

    _ensure_placeholder_plot(ARTIFACTS_DIR / "ml_roc_pr_curve.png", "ROC/PR (placeholder)")
    _ensure_placeholder_plot(ARTIFACTS_DIR / "bt_equity_curve.png", "Equity Curve (placeholder)")

    # Optional provenance
    if callable(globals().get("detect_provenance", None)):
        prov = detect_provenance()  # type: ignore
        _write_json(ARTIFACTS_DIR / "data_provenance.json", prov)
        print(f"[train_predict] wrote data_provenance.json: {prov.get('mode', 'unknown')}")

if __name__ == "__main__":
    main()
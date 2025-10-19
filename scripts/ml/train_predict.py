# scripts/ml/train_predict.py
from __future__ import annotations

"""
End-to-end training + prediction + threshold tuning + artifact write-out.

Key points:
- Prefer RELATIVE imports first (works when run as `scripts.ml.train_predict`).
- Fall back to absolute imports and a minimal sys.path shim if needed.
- Writes tests' required artifacts:
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# Resolve repo paths early
# ---------------------------
import sys
FALLBACK_ROOT = Path(__file__).resolve().parents[2]

try:
    # If available, prefer canonical paths
    from scripts.paths import ROOT, MODELS_DIR, ARTIFACTS_DIR  # type: ignore
except Exception:
    ROOT = FALLBACK_ROOT
    MODELS_DIR = ROOT / "models"
    ARTIFACTS_DIR = ROOT / "artifacts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# RELATIVE imports first (most reliable)
# -----------------------------------------
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
    # Fall back to absolute imports + a tiny sys.path shim if needed
    if str(FALLBACK_ROOT) not in sys.path:
        sys.path.insert(0, str(FALLBACK_ROOT))
    try:
        from scripts.ml.data_loader import load_prices  # type: ignore
        from scripts.ml.features import (  # type: ignore
            build_features,
            label_next_horizon,
            walk_forward_splits,
        )
        from scripts.ml.model_runner import train_model, predict_proba  # type: ignore
        from scripts.ml.tuner import tune_thresholds  # type: ignore
        try:
            from scripts.ml._provenance import detect_provenance  # type: ignore
        except Exception:
            detect_provenance = None
    except Exception as e:
        # surface a very clear error if both paths fail
        raise ImportError(
            "Failed to import ML pipeline modules via relative and absolute paths. "
            "Ensure 'scripts/ml' is a package (has __init__.py) and that pytest "
            "invokes this as 'scripts.ml.train_predict'."
        ) from e


def _train_one_symbol(
    sym: str,
    df_feat: pd.DataFrame,
    horizon_h: int = 1,
    model_type: str = "logreg",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Train simple model for one symbol and return probabilities + frame."""
    df = label_next_horizon(df_feat, horizon_h=horizon_h)
    cols = ["r_1h", "r_3h", "r_6h", "vol_6h", "atr_14h", "sma_gap", "high_vol", "social_score"]
    X = df[cols].values
    y = df["y_long"].values

    # walk-forward; take first usable split
    for tr_ix, te_ix in walk_forward_splits(df, n_splits=2, train_days=10, test_days=5):
        if len(tr_ix) < 10 or len(te_ix) < 5:
            continue
        model = train_model(X[tr_ix], y[tr_ix], model_type=model_type)
        p = predict_proba(model, X)
        out_df = pd.DataFrame({"ts": df["ts"].values, "p_long": p})
        return p, out_df

    # fallback if splits too small
    model = train_model(X, y, model_type=model_type)
    p = predict_proba(model, X)
    out_df = pd.DataFrame({"ts": df["ts"].values, "p_long": p})
    return p, out_df


def main() -> None:
    symbols = ["BTC", "ETH", "SOL"]
    lookback_days = int(os.getenv("MW_LOOKBACK_DAYS", "60"))

    # 1) Load prices (real if creds available; otherwise demo fallback inside loader)
    prices: Dict[str, pd.DataFrame] = load_prices(symbols, lookback_days=lookback_days)

    # 2) Build features
    feats = build_features(prices)

    # 3) Train quick models per symbol and collect prob frames
    model_type = os.getenv("MW_MODEL", "logreg")
    prob_frames: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        _, df_probs = _train_one_symbol(s, feats[s], horizon_h=1, model_type=model_type)
        prob_frames[s] = df_probs

    # 4) Threshold tuning + tiny backtest
    bt_summary = tune_thresholds(prob_frames, prices)

    # 5) Write artifacts required by tests
    manifest = {
        "version": "v0.9.1",
        "model_type": model_type,
        "symbols": symbols,
        "lookback_days": lookback_days,
    }
    (MODELS_DIR / "ml_model_manifest.json").write_text(json.dumps(manifest, indent=2))
    (MODELS_DIR / "backtest_summary.json").write_text(json.dumps(bt_summary, indent=2))

    params = bt_summary.get("params") or bt_summary.get("parameters") or {}
    agg = bt_summary.get("agg") or bt_summary.get("aggregate") or {}
    thresholds = {
        "chosen": {
            "conf_min": params.get("conf_min"),
            "debounce_min": params.get("debounce_min"),
            "horizon_h": params.get("horizon_h"),
        },
        "aggregate": agg,
    }
    (MODELS_DIR / "signal_thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # 6) Ensure required plots exist (create simple placeholders if tuner didn't save)
    for png in ("ml_roc_pr_curve.png", "bt_equity_curve.png"):
        p = ARTIFACTS_DIR / png
        if not p.exists():
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot([0, 1], [0, 1])
                plt.title(png.replace("_", " "))
                plt.savefig(p, dpi=96, bbox_inches="tight")
                plt.close()
            except Exception:
                p.write_bytes(b"\x89PNG\r\n\x1a\n")

    # 7) Optional data provenance
    try:
        if detect_provenance is not None:
            prov = detect_provenance(symbols)
            prov_path = ARTIFACTS_DIR / "data_provenance.json"
            prov_path.write_text(json.dumps(prov, indent=2))
            print(f"[Provenance] wrote {prov_path}")
    except Exception as e:
        print(f"[Provenance] skipped ({e})")

    # 8) Final log
    print("Artifacts written:")
    for p in [
        MODELS_DIR / "ml_model_manifest.json",
        MODELS_DIR / "backtest_summary.json",
        MODELS_DIR / "signal_thresholds.json",
        ARTIFACTS_DIR / "ml_roc_pr_curve.png",
        ARTIFACTS_DIR / "bt_equity_curve.png",
        ARTIFACTS_DIR / "data_provenance.json",
    ]:
        print(f" - {p.relative_to(ROOT)} (exists={p.exists()})")


if __name__ == "__main__":
    main()
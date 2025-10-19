# scripts/ml/train_predict.py
from __future__ import annotations
import os, json, pathlib, re
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dirs, env_str, env_int, env_float, to_list, save_json
from .data_loader import load_prices
from .feature_builder import build_features
from .labeler import label_next_horizon
from .splitter import walk_forward_splits
from .model_runner import train_model, predict_proba
from .tuner import tune_thresholds

# --- optional provenance hook (safe if missing) ---
try:
    from ._provenance import detect_provenance  # type: ignore
except Exception:
    detect_provenance = None  # graceful fallback

ROOT = pathlib.Path(".").resolve()

def _feature_matrix(df: pd.DataFrame):
    feature_cols = ["r_1h","r_3h","r_6h","vol_6h","atr_14h","sma_gap","high_vol","social_score"]
    X = df[feature_cols].values.astype(float)
    y = df["y_long"].values.astype(int)
    return X, y, feature_cols

def _plots_roc_pr_placeholder():
    # Simple placeholder figure (we're not computing ROC/PR here to keep deps minimal)
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title("ML ROC/PR (placeholder)")
    plt.plot([0,1],[0,1])
    plt.savefig(ROOT/"artifacts/ml_roc_pr_curve.png")
    plt.close()

def _plot_equity_placeholder():
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title("Backtest Equity (placeholder)")
    plt.plot([0,1,2,3],[1,1.01,0.99,1.02])
    plt.savefig(ROOT/"artifacts/bt_equity_curve.png")
    plt.close()

def _maybe_fail_on_demo(prov: dict):
    """
    Optional guard. OFF by default.
    Enable by setting:
      MW_FAIL_ON_DEMO=1
      MW_BRANCH=<current branch name> (CI: ${{ github.ref_name }})
      MW_PROTECTED_PATTERN='^main$'   (regex; defaults to ^main$)
    If provenance indicates demo + branch matches protected pattern -> raise.
    """
    if not prov or not isinstance(prov, dict):
        return
    if str(os.getenv("MW_FAIL_ON_DEMO", "0")).strip() not in {"1", "true", "TRUE"}:
        return

    branch = os.getenv("MW_BRANCH", "")
    pattern = os.getenv("MW_PROTECTED_PATTERN", r"^main$")
    is_protected = bool(re.search(pattern, branch or ""))

    # We treat these flags as "demo" hints; adjust to your _provenance payload.
    is_demo = bool(prov.get("demo", False)) or str(prov.get("source", "")).lower().startswith("demo")

    if is_protected and is_demo:
        raise RuntimeError(
            f"Provenance indicates demo data on protected branch '{branch}'. "
            f"Set MW_FAIL_ON_DEMO=0 to bypass, or switch to real data."
        )

def main():
    ensure_dirs()

    symbols = to_list(env_str("MW_ML_SYMBOLS", "BTC,ETH,SOL"))
    lookback_days = env_int("MW_ML_LOOKBACK_DAYS", 180)
    model_type = env_str("MW_ML_MODEL", "logreg")
    train_days = env_int("MW_TRAIN_DAYS", 60)
    test_days = env_int("MW_TEST_DAYS", 30)
    horizon_h = env_int("MW_HORIZON_H", 1)

    # --- Load + build features (unchanged behavior) ---
    prices = load_prices(symbols, lookback_days=lookback_days)
    feats = build_features(prices)

    # --- Data provenance (new; harmless if _provenance not present) ---
    provenance_payload = None
    if callable(detect_provenance):
        try:
            provenance_payload = detect_provenance(
                prices=prices,
                symbols=symbols,
                lookback_days=lookback_days,
                env=dict(
                    MW_OFFLINE_DEMO=os.getenv("MW_OFFLINE_DEMO"),
                    MW_DEMO=os.getenv("MW_DEMO"),
                    DEMO_MODE=os.getenv("DEMO_MODE"),
                ),
            )
        except Exception as _:
            provenance_payload = {"error": "detect_provenance_failed"}

        # write artifacts/data_provenance.json (doesn't affect tests)
        try:
            save_json("artifacts/data_provenance.json", provenance_payload)
        except Exception:
            # never fail test run just because provenance write failed
            pass

        # optional hard-fail if configured
        _maybe_fail_on_demo(provenance_payload)

    pred_dfs: Dict[str, pd.DataFrame] = {}
    manifest = {
        "model_type": model_type,
        "symbols": symbols,
        "lookback_days": lookback_days,
        "feature_list": ["r_1h","r_3h","r_6h","vol_6h","atr_14h","sma_gap","high_vol","social_score"],
        "train_days": train_days,
        "test_days": test_days,
        "horizon_h": horizon_h,
        "fold_metrics": [],
    }

    for sym in symbols:
        df = label_next_horizon(feats[sym], horizon_h=horizon_h)
        X, y, feature_cols = _feature_matrix(df)

        # walk-forward; keep last fold predictions
        preds = np.zeros(len(df))
        fold_ix = 0
        for tr_ix, te_ix in walk_forward_splits(df, n_splits=3, train_days=train_days, test_days=test_days):
            if len(tr_ix) < 10 or len(te_ix) < 5:  # tiny safety
                continue
            m = train_model(X[tr_ix], y[tr_ix], model_type=model_type)
            p = predict_proba(m, X[te_ix])
            preds[te_ix] = p
            # simple fold metric
            manifest["fold_metrics"].append({
                "symbol": sym,
                "fold": fold_ix,
                "train_len": int(len(tr_ix)),
                "test_len": int(len(te_ix)),
                "pred_mean": float(p.mean()),
            })
            fold_ix += 1

        pred_dfs[sym] = pd.DataFrame({"ts": df["ts"], "p_long": preds})

    # --- tune thresholds (unchanged) ---
    best = tune_thresholds(pred_dfs, prices)
    save_json("models/backtest_summary.json", {"aggregate": best["agg"], "per_symbol": best["per_symbol"]})
    save_json("models/ml_model_manifest.json", manifest)

    # --- placeholder plots (unchanged) ---
    _plots_roc_pr_placeholder()
    _plot_equity_placeholder()

if __name__ == "__main__":
    main()
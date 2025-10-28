# scripts/ml/train_predict.py
from __future__ import annotations
import os, json, pathlib, re, shutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dirs, env_str, env_int, to_list, save_json
from .data_loader import load_prices
from .feature_builder import build_features
from .labeler import label_next_horizon
from .splitter import walk_forward_splits
from .model_runner import train_model, predict_proba

# Import per-regime training (optional)
try:
    from .per_regime_trainer import (
        per_regime_enabled,
        train_per_regime_models,
        predict_per_regime,
        save_per_regime_models
    )
    _PER_REGIME_AVAILABLE = True
except ImportError:
    _PER_REGIME_AVAILABLE = False
    per_regime_enabled = lambda: False

ROOT = pathlib.Path(".").resolve()

# --- optional provenance (no-op if module not present)
try:
    from ._provenance import detect_provenance  # optional
except Exception:
    detect_provenance = None

# =========================
# Feature list (now includes bursts + optional regime)
# =========================
_BASE_FEATURES = [
    "r_1h","r_3h","r_6h",
    "vol_6h","atr_14h","sma_gap","high_vol",
    "social_score",
    "price_burst",      # NEW
    "social_burst",     # NEW
]

# Dynamically add regime_trending if enabled
def _get_features():
    """Get feature list, conditionally including regime feature."""
    features = _BASE_FEATURES.copy()
    if str(os.getenv("MW_REGIME_ENABLED", "0")).lower() in {"1", "true", "yes"}:
        features.append("regime_trending")
    return features

FEATURES = _get_features()

# =========================
# Per-coin model overrides
# =========================
def _parse_model_map(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part: 
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().upper()] = v.strip()
        else:
            # allow single token to mean DEFAULT:<token>
            out["DEFAULT"] = part
    return out

MODEL_MAP = _parse_model_map(os.getenv("MW_ML_MODEL_MAP", ""))

def model_type_for(sym: str, fallback: str) -> str:
    return MODEL_MAP.get(sym.upper(), MODEL_MAP.get("DEFAULT", fallback))

PER_SYMBOL_MODELS = str(os.getenv("MW_PER_SYMBOL_MODELS", "1")).lower() in {"1","true","yes"}

# ---------- provenance ----------
def _write_provenance(prices_map, symbols, lookback_days):
    out = ROOT / "artifacts" / "data_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "symbols": symbols,
        "lookback_days": int(lookback_days),
        "env": {
            "MW_OFFLINE_DEMO": os.getenv("MW_OFFLINE_DEMO"),
            "MW_DEMO": os.getenv("MW_DEMO"),
            "DEMO_MODE": os.getenv("DEMO_MODE"),
            "MW_BRANCH": os.getenv("MW_BRANCH"),
        },
    }

    payload = None
    if callable(detect_provenance):
        try:
            try:
                payload = detect_provenance(prices_map, symbols)
            except TypeError:
                payload = detect_provenance(
                    prices=prices_map,
                    symbols=symbols,
                    lookback_days=lookback_days,
                    env=base["env"],
                )
        except Exception as e:
            base["error"] = f"{type(e).__name__}: {e}"
    else:
        base["error"] = "provenance_module_missing"

    to_write = {**base, **(payload or {})}
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_write, f, indent=2, sort_keys=True, default=str)
    print(f"[provenance] wrote {out}")

# ---------- helpers ----------
def _get_selected_features() -> List[str]:
    """
    Get feature list based on MW_FEATURE_SELECTION env var.

    Options:
    - 'top5': Use only top 5 most important features
    - 'all' or empty: Use all features (default)
    """
    feature_selection = env_str("MW_FEATURE_SELECTION", "all").lower()

    if feature_selection == "top5":
        # Top 5 features based on empirical importance
        return ["r_6h", "vol_6h", "social_score", "atr_14h", "sma_gap"]

    # Default: use all features
    return FEATURES


def _feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    selected_features = _get_selected_features()

    # Ensure all selected features exist in df
    available_features = [f for f in selected_features if f in df.columns]
    if len(available_features) != len(selected_features):
        missing = set(selected_features) - set(available_features)
        print(f"[WARN] Missing features: {missing}. Using {len(available_features)} available features.")

    X = df[available_features].values.astype(float)
    y = df["y_long"].values.astype(int)
    return X, y, available_features

def _plots_roc_pr_placeholder():
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
    if not prov or not isinstance(prov, dict):
        return
    if str(os.getenv("MW_FAIL_ON_DEMO", "0")).strip() not in {"1", "true", "TRUE"}:
        return
    branch = os.getenv("MW_BRANCH", "")
    pattern = os.getenv("MW_PROTECTED_PATTERN", r"^main$")
    is_protected = bool(re.search(pattern, branch or ""))
    is_demo = bool(prov.get("demo", False)) or str(prov.get("source", "")).lower().startswith("demo")
    if is_protected and is_demo:
        raise RuntimeError(
            f"Provenance indicates demo data on protected branch '{branch}'. "
            f"Set MW_FAIL_ON_DEMO=0 to bypass, or switch to real data."
        )

def _write_bundle(dirpath: pathlib.Path, model_obj, feature_list, manifest_dict):
    import joblib
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "manifest.json").write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    (dirpath / "features.json").write_text(json.dumps({"feature_order": feature_list}, indent=2), encoding="utf-8")
    joblib.dump(model_obj, dirpath / "model.joblib")

# ---------- main ----------
def main():
    ensure_dirs()

    symbols = to_list(env_str("MW_ML_SYMBOLS", "BTC,ETH,SOL"))
    lookback_days = env_int("MW_ML_LOOKBACK_DAYS", 180)
    model_type_global = env_str("MW_ML_MODEL", "hybrid")
    train_days = env_int("MW_TRAIN_DAYS", 60)
    test_days = env_int("MW_TEST_DAYS", 30)
    horizon_h = env_int("MW_HORIZON_H", 1)
    n_splits = env_int("MW_N_SPLITS", 3)  # Number of walk-forward folds

    # Load + build features
    prices = load_prices(symbols, lookback_days=lookback_days)
    feats = build_features(prices)
    _write_provenance(prices, symbols, lookback_days)

    pred_dfs: Dict[str, pd.DataFrame] = {}
    per_symbol_models: Dict[str, object] = {}

    manifest_base = {
        "symbols": symbols,
        "lookback_days": lookback_days,
        "features": FEATURES,
        "feature_list": FEATURES,
        "social_enabled": str(os.getenv("MW_SOCIAL_ENABLED","0")).lower() in {"1","true","yes"},
        "train_days": train_days,
        "test_days": test_days,
        "horizon_h": horizon_h,
        "fold_metrics": [],
        "model_type": model_type_global,  # global fallback
    }
    manifest_base["social_include"] = os.getenv("MW_SOCIAL_INCLUDE")
    manifest_base["fix_end_ts"] = os.getenv("MW_FIX_END_TS")

    # Choose a primary symbol for legacy single-bundle compatibility
    primary_symbol: Optional[str] = symbols[0] if symbols else None
    export_model = None

    # Check if regime filtering or per-regime training is enabled
    regime_filter_enabled = str(os.getenv("MW_REGIME_FILTER_ENABLED", "0")).lower() in {"1", "true", "yes"}
    per_regime_training = _PER_REGIME_AVAILABLE and per_regime_enabled()

    for sym in symbols:
        df = label_next_horizon(feats[sym], horizon_h=horizon_h)
        X, y, feature_cols = _feature_matrix(df)

        sym_model_type = model_type_for(sym, model_type_global)

        # walk-forward; keep last fold predictions
        preds = np.zeros(len(df))
        fold_ix = 0
        for tr_ix, te_ix in walk_forward_splits(df, n_splits=n_splits, train_days=train_days, test_days=test_days):
            if len(tr_ix) < 10 or len(te_ix) < 5:
                continue

            # ===== PER-REGIME TRAINING (Approach 4) =====
            if per_regime_training:
                try:
                    df_train = df.iloc[tr_ix]
                    df_test = df.iloc[te_ix]
                    # Train separate models for each regime
                    regime_models = train_per_regime_models(
                        X[tr_ix], y[tr_ix],
                        df_train, prices[sym],
                        model_type=sym_model_type,
                        symbol=sym
                    )
                    # Predict using regime-specific models
                    p = predict_per_regime(
                        regime_models,
                        X[te_ix],
                        df_test,
                        prices[sym],
                        symbol=sym
                    )
                    preds[te_ix] = p
                    manifest_base["fold_metrics"].append({
                        "symbol": sym,
                        "model_type": sym_model_type,
                        "fold": fold_ix,
                        "train_len": int(len(tr_ix)),
                        "test_len": int(len(te_ix)),
                        "pred_mean": float(p.mean()),
                        "per_regime_training": True,
                    })
                    fold_ix += 1
                    continue
                except Exception as e:
                    print(f"[per_regime] {sym} fold {fold_ix}: Per-regime training failed: {e}, falling back to standard training")

            # ===== STANDARD TRAINING WITH OPTIONAL REGIME FILTERING (Approach 1) =====
            # Apply regime filtering to training data only (if enabled)
            tr_ix_filtered = tr_ix
            if regime_filter_enabled and "regime_trending" in df.columns:
                # Filter training set to only include trending markets
                regime_mask = df["regime_trending"].iloc[tr_ix].values > 0.5
                tr_ix_filtered = tr_ix[regime_mask]
                print(f"[regime_filter] {sym} fold {fold_ix}: filtered {len(tr_ix)} -> {len(tr_ix_filtered)} training samples (trending only)")

                if len(tr_ix_filtered) < 10:
                    print(f"[regime_filter] {sym} fold {fold_ix}: insufficient trending samples, skipping fold")
                    continue

            m = train_model(X[tr_ix_filtered], y[tr_ix_filtered], model_type=sym_model_type)
            p = predict_proba(m, X[te_ix])
            preds[te_ix] = p
            manifest_base["fold_metrics"].append({
                "symbol": sym,
                "model_type": sym_model_type,
                "fold": fold_ix,
                "train_len": int(len(tr_ix)),
                "train_len_filtered": int(len(tr_ix_filtered)) if regime_filter_enabled else int(len(tr_ix)),
                "test_len": int(len(te_ix)),
                "pred_mean": float(p.mean()),
                "regime_filter_enabled": regime_filter_enabled,
                "per_regime_training": False,
            })
            fold_ix += 1

        pred_dfs[sym] = pd.DataFrame({"ts": df["ts"], "p_long": preds})

        # full-data export per symbol
        try:
            full_model = train_model(X, y, model_type=sym_model_type)
            per_symbol_models[sym] = full_model
            print(f"[bundle] trained full-data model for {sym} with {sym_model_type} (rows={len(y)})")
            if export_model is None and sym == primary_symbol:
                export_model = full_model
            # write a tiny per-symbol feature preview for CI summary
            preview = df[FEATURES].iloc[-1].to_dict()
            (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
            (ROOT / "artifacts" / f"features_preview_{sym}.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[bundle] WARN: failed export model for {sym}: {e}")

    # tune thresholds (optional)
    try:
        from .tuner import tune_thresholds
        best = tune_thresholds(pred_dfs, prices)
        save_json("models/backtest_summary.json", {"aggregate": best.get("agg", {}), "per_symbol": best.get("per_symbol", {})})
    except Exception as e:
        print(f"[tuner] WARN: tune_thresholds failed: {e}")
        save_json("models/backtest_summary.json", {"aggregate": {}, "per_symbol": {}})

    save_json("models/ml_model_manifest.json", manifest_base)

    # placeholder plots
    _plots_roc_pr_placeholder()
    _plot_equity_placeholder()

    # Export bundles
    try:
        BUNDLE_ROOT = ROOT / "models" / "current"
        BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

        if PER_SYMBOL_MODELS and per_symbol_models:
            index = {"symbols": [], "default": None}
            for s, m in per_symbol_models.items():
                d = BUNDLE_ROOT / s
                man = {**manifest_base, "symbol": s, "model_type": model_type_for(s, model_type_global)}
                _write_bundle(d, m, FEATURES, man)
                index["symbols"].append(s)
            # optional default fallback
            if export_model is not None:
                _write_bundle(BUNDLE_ROOT / "default", export_model, FEATURES, {**manifest_base, "symbol": None})
                index["default"] = "default"
            (BUNDLE_ROOT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
            print("[bundle] wrote per-symbol bundles:", index)
        else:
            # legacy flat single bundle
            target_model = export_model or (list(per_symbol_models.values())[0] if per_symbol_models else None)
            if target_model is None:
                raise RuntimeError("no trained model available to export")
            _write_bundle(BUNDLE_ROOT, target_model, FEATURES, manifest_base)
            print("[bundle] wrote single bundle at models/current")

    except Exception as e:
        print(f"[bundle] WARN: failed to export inference bundle: {e}")

if __name__ == "__main__":
    main()
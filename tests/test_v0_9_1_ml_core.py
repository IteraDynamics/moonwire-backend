# tests/test_v0.9.1_ml_core.py
import os, json, pathlib
import pandas as pd
from scripts.ml.data_loader import load_prices
from scripts.ml.feature_builder import build_features
from scripts.ml.labeler import label_next_horizon
from scripts.ml.splitter import walk_forward_splits
from scripts.ml.model_runner import train_model, predict_proba
from scripts.ml.tuner import tune_thresholds

ROOT = pathlib.Path(".").resolve()

def test_pipeline_runs_end_to_end():
    syms = ["BTC","ETH","SOL"]
    prices = load_prices(syms, lookback_days=30)  # keep quick for CI
    feats = build_features(prices)
    dfs = {}
    for s in syms:
        df = label_next_horizon(feats[s], horizon_h=1)
        X = df[["r_1h","r_3h","r_6h","vol_6h","atr_14h","sma_gap","high_vol","social_score"]].values
        y = df["y_long"].values
        for tr_ix, te_ix in walk_forward_splits(df, n_splits=2, train_days=10, test_days=5):
            if len(tr_ix) < 10 or len(te_ix) < 5:
                continue
            m = train_model(X[tr_ix], y[tr_ix], model_type="logreg")
            p = predict_proba(m, X)
            assert p.shape[0] == X.shape[0]
            dfs[s] = pd.DataFrame({"ts": df["ts"], "p_long": p})
            break

    res = tune_thresholds(dfs, prices)
    assert "params" in res and "agg" in res

def test_artifacts_exist_after_train_predict(tmp_path):
    # run the module to produce artifacts
    import runpy
    runpy.run_module("scripts.ml.train_predict", run_name="__main__")

    for p in [
        "models/ml_model_manifest.json",
        "models/backtest_summary.json",
        "models/signal_thresholds.json",
        "artifacts/ml_roc_pr_curve.png",
        "artifacts/bt_equity_curve.png",
    ]:
        assert (ROOT / p).exists(), f"missing artifact: {p}"
        assert (ROOT / p).stat().st_size > 0

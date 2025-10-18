# scripts/ml/train_predict.py
# MoonWire v0.9.1 — ML core: train → predict → tune → artifacts
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Use non-interactive backend for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# ──────────────────────────────────────────────────────────────────────────────
# Repo root detection (robust for local + GitHub Actions)
# ──────────────────────────────────────────────────────────────────────────────
def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for up in [p] + list(p.parents):
        if (up / ".git").exists() or (up / ".github").exists() or (up / "pyproject.toml").exists():
            return up
    # Fallback: scripts/ml/<here> -> repo root is parents[2]
    return start.resolve().parents[2]

ROOT = _find_repo_root(Path(__file__))
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ROOT / "logs"
for d in (MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Imports from the ML core modules
# ──────────────────────────────────────────────────────────────────────────────
from scripts.ml.data_loader import load_prices
from scripts.ml.feature_builder import build_features
from scripts.ml.labeler import label_next_horizon
from scripts.ml.splitter import walk_forward_splits
from scripts.ml.model_runner import train_model, predict_proba
from scripts.ml.backtest import run_backtest
from scripts.ml.tuner import tune_thresholds

# ──────────────────────────────────────────────────────────────────────────────
# Config from environment (with safe defaults for CI/demo)
# ──────────────────────────────────────────────────────────────────────────────
SYMBOLS = [s.strip() for s in os.getenv("MW_ML_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()]
LOOKBACK_DAYS = int(os.getenv("MW_ML_LOOKBACK_DAYS", "60"))
TRAIN_DAYS = int(os.getenv("MW_TRAIN_DAYS", "30"))
TEST_DAYS = int(os.getenv("MW_TEST_DAYS", "15"))
MODEL_TYPE = os.getenv("MW_ML_MODEL", "logreg")  # "logreg" | "gb"
HORIZON_H_DEFAULT = int(os.getenv("MW_HORIZON_H", "1"))

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FitBundle:
    symbol: str
    df_labeled: pd.DataFrame
    proba: np.ndarray  # p(long)
    ts: pd.Series

def _safe_series(v, default):
    try:
        return float(v)
    except Exception:
        return default

def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)

def _plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if len(y_true) == 0 or len(np.unique(y_true)) == 1:
        # Degenerate case — produce a placeholder figure so CI has a PNG.
        fig = plt.figure(figsize=(6, 4))
        plt.title("ROC/PR (insufficient labels)")
        plt.text(0.5, 0.5, "Not enough positives/negatives", ha="center", va="center")
        plt.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig = plt.figure(figsize=(10, 4))

    # Left: ROC
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(fpr, tpr, lw=1.5)
    ax1.plot([0, 1], [0, 1], linestyle="--", lw=1)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title(f"ROC (AUC={roc_auc:.3f})")

    # Right: PR
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(recall, precision, lw=1.5)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"PR (AUC={pr_auc:.3f})")

    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

def _plot_equity(equity: List[dict], out_png: Path, title: str = "Equity Curve") -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    if not equity:
        plt.title(title + " (no trades)")
        plt.text(0.5, 0.5, "No equity data", ha="center", va="center")
    else:
        ts = [e["ts"] for e in equity]
        eq = [e["equity"] for e in equity]
        plt.plot(ts, eq, lw=1.5)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Equity")
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1) Load hourly prices
    prices: Dict[str, pd.DataFrame] = load_prices(SYMBOLS, lookback_days=LOOKBACK_DAYS)

    # 2) Build features
    feat_map: Dict[str, pd.DataFrame] = build_features(prices)

    # 3) Label & train per symbol (simple walk-forward: first viable split)
    bundles: List[FitBundle] = []
    for sym in SYMBOLS:
        fdf = feat_map.get(sym)
        if fdf is None or fdf.empty:
            continue

        df = label_next_horizon(fdf, horizon_h=HORIZON_H_DEFAULT).dropna().reset_index(drop=True)
        if df.empty:
            continue

        X_cols = ["r_1h","r_3h","r_6h","vol_6h","atr_14h","sma_gap","high_vol","social_score"]
        X = df[X_cols].values
        y = df["y_long"].values

        # Walk-forward to get a simple train split
        trained = False
        for tr_ix, te_ix in walk_forward_splits(df, n_splits=2, train_days=TRAIN_DAYS, test_days=TEST_DAYS):
            if len(tr_ix) < 10 or len(te_ix) < 5:
                continue
            model = train_model(X[tr_ix], y[tr_ix], model_type=MODEL_TYPE)
            proba = predict_proba(model, X)  # p(long) for full series
            bundles.append(FitBundle(symbol=sym, df_labeled=df, proba=proba, ts=df["ts"]))
            trained = True
            break

        if not trained:
            # Fallback: trivial probabilities if we failed to get a split
            proba = np.full(len(df), 0.5, dtype=float)
            bundles.append(FitBundle(symbol=sym, df_labeled=df, proba=proba, ts=df["ts"]))

    # 4) Aggregate predictions dict for tuner
    pred_dfs: Dict[str, pd.DataFrame] = {}
    y_true_concat = []
    y_prob_concat = []
    for b in bundles:
        pred = pd.DataFrame({"ts": b.ts, "p_long": b.proba})
        pred_dfs[b.symbol] = pred
        # collect for ROC/PR
        if "y_long" in b.df_labeled.columns:
            y_true_concat.append(b.df_labeled["y_long"].astype(int).values)
            y_prob_concat.append(b.proba.astype(float))

    # 5) Tune thresholds (writes models/signal_thresholds.json inside)
    tune_res = tune_thresholds(pred_dfs, prices)
    # Guarantee a params block for downstream usage
    params = tune_res.get("params", {
        "conf_min": _safe_series(tune_res.get("conf_min", 0.55), 0.55),
        "debounce_min": int(tune_res.get("debounce_min", 15)),
        "horizon_h": int(tune_res.get("horizon_h", HORIZON_H_DEFAULT)),
    })

    # 6) Save backtest summary (normalize shape)
    agg = tune_res.get("agg", tune_res.get("aggregate", {}))
    per_symbol = tune_res.get("per_symbol", {})
    backtest_summary = {
        "aggregate": {
            "win_rate": _safe_series(agg.get("win_rate", 0.0), 0.0),
            "profit_factor": _safe_series(agg.get("profit_factor", 0.0), 0.0),
            "max_drawdown": _safe_series(agg.get("max_drawdown", 0.0), 0.0),
            "signals_per_day": _safe_series(agg.get("signals_per_day", 0.0), 0.0),
            "n_trades": int(agg.get("n_trades", 0)),
        },
        "per_symbol": {},
    }
    for s, m in per_symbol.items():
        backtest_summary["per_symbol"][s] = {
            "win_rate": _safe_series(m.get("win_rate", 0.0), 0.0),
            "profit_factor": _safe_series(m.get("profit_factor", 0.0), 0.0),
            "max_drawdown": _safe_series(m.get("max_drawdown", 0.0), 0.0),
            "signals_per_day": _safe_series(m.get("signals_per_day", 0.0), 0.0),
            "n_trades": int(m.get("n_trades", 0)),
        }
    _save_json(MODELS_DIR / "backtest_summary.json", backtest_summary)

    # 7) Save model manifest (simple, we retrain each run)
    manifest = {
        "model_type": MODEL_TYPE,
        "features": ["r_1h","r_3h","r_6h","vol_6h","atr_14h","sma_gap","high_vol","social_score"],
        "symbols": SYMBOLS,
        "train_window_days": TRAIN_DAYS,
        "test_window_days": TEST_DAYS,
        "horizon_h_default": HORIZON_H_DEFAULT,
        "train_time_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": backtest_summary["aggregate"],
    }
    _save_json(MODELS_DIR / "ml_model_manifest.json", manifest)

    # 8) Ensure signal_thresholds.json exists (the tuner usually writes it;
    #    if it didn't, write from resolved params)
    thresholds_path = MODELS_DIR / "signal_thresholds.json"
    if not thresholds_path.exists():
        _save_json(thresholds_path, {
            "conf_min": float(params["conf_min"]),
            "debounce_min": int(params["debounce_min"]),
            "horizon_h": int(params["horizon_h"]),
        })

    # 9) Produce ROC/PR chart (aggregate across symbols where possible)
    if y_true_concat and y_prob_concat:
        y_true = np.concatenate(y_true_concat)
        y_prob = np.concatenate(y_prob_concat)
    else:
        # Degenerate: no labels; make empty arrays to trigger placeholder chart
        y_true = np.array([], dtype=int)
        y_prob = np.array([], dtype=float)

    _plot_roc_pr(y_true, y_prob, ARTIFACTS_DIR / "ml_roc_pr_curve.png")

    # 10) Quick equity plot using the chosen params on BTC (or first available)
    equity_data: List[dict] = []
    try:
        chosen = {
            "conf_min": float(params["conf_min"]),
            "debounce_min": int(params["debounce_min"]),
            "horizon_h": int(params["horizon_h"]),
        }
        sym_for_eq = None
        for s in SYMBOLS:
            if s in pred_dfs and s in prices:
                sym_for_eq = s
                break
        if sym_for_eq is not None:
            bt = run_backtest(
                pred_df=pred_dfs[sym_for_eq],
                prices_df=prices[sym_for_eq],
                conf_min=chosen["conf_min"],
                debounce_min=chosen["debounce_min"],
                horizon_h=chosen["horizon_h"],
            )
            equity_data = bt.get("equity", [])
    except Exception:
        # Never fail CI because of plotting — fall back to empty curve
        equity_data = []

    _plot_equity(equity_data, ARTIFACTS_DIR / "bt_equity_curve.png", title="Backtest Equity (sample)")

    # (optional) dump a tiny trades log if present
    logs_path = LOGS_DIR / "trades.jsonl"
    if not logs_path.exists():
        try:
            if sym_for_eq is not None:
                bt = run_backtest(
                    pred_df=pred_dfs[sym_for_eq],
                    prices_df=prices[sym_for_eq],
                    conf_min=float(params["conf_min"]),
                    debounce_min=int(params["debounce_min"]),
                    horizon_h=int(params["horizon_h"]),
                )
                trades = bt.get("trades", [])
                with logs_path.open("w", encoding="utf-8") as f:
                    if isinstance(trades, list):
                        for t in trades:
                            # Support Trade dataclass or dict
                            if hasattr(t, "__dict__"):
                                f.write(json.dumps(t.__dict__, default=str) + "\n")
                            else:
                                f.write(json.dumps(t, default=str) + "\n")
                    else:
                        # Not a list — write a single line summary
                        f.write(json.dumps({"n_trades": int(trades)}) + "\n")
        except Exception:
            # best-effort only
            pass

if __name__ == "__main__":
    main()
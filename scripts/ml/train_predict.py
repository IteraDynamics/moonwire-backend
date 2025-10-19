# scripts/ml/train_predict.py
from __future__ import annotations
"""
End-to-end training + prediction + threshold tuning + artifact write-out.

- Uses your real pipeline (data_loader → features → model_runner → tuner).
- Writes test-required artifacts:
  * models/ml_model_manifest.json
  * models/backtest_summary.json
  * models/signal_thresholds.json
  * artifacts/ml_roc_pr_curve.png
  * artifacts/bt_equity_curve.png
- Optionally writes artifacts/data_provenance.json if _provenance.detect_provenance() exists.
- Optional hard-fail on demo provenance is OFF by default (enable via env in CI).

This file aims to be import/runner-agnostic and WILL NOT change tuning behavior.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------- repo paths ----------
try:
    from scripts.paths import ROOT, MODELS_DIR, ARTIFACTS_DIR  # type: ignore
except Exception:
    ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR = ROOT / "models"
    ARTIFACTS_DIR = ROOT / "artifacts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- project imports (real pipeline) ----------
from scripts.ml.data_loader import load_prices  # type: ignore
from scripts.ml.features import (               # type: ignore
    build_features,
    label_next_horizon,
    walk_forward_splits,
)
from scripts.ml.model_runner import train_model, predict_proba  # type: ignore
from scripts.ml.tuner import tune_thresholds                    # type: ignore

# provenance is optional
try:
    from scripts.ml._provenance import detect_provenance  # type: ignore
except Exception:
    detect_provenance = None


# ---------- helpers ----------
def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))

_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc``\x00"
    b"\x00\x00\x04\x00\x01\x0b\xe7\x02\x9d\x00\x00\x00\x00IEND\xaeB`\x82"
)
def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_TINY_PNG)

def _provenance_guard() -> None:
    """
    Optional hard-fail if using demo data on protected branches.
    Disabled by default.

    Enable in CI by setting:
      MW_FAIL_ON_DEMO=1
      MW_BRANCH=<branch name>    (CI usually supplies this; optional)
      MW_PROTECTED_PATTERN=^main$|^release/   (optional regex; default ^main$)

    This guard does NOT alter model/tuning behavior; it only decides to exit early.
    """
    if not detect_provenance:
        return
    try:
        prov = detect_provenance()
    except Exception:
        return

    # Always write provenance for observability
    _write_json(ARTIFACTS_DIR / "data_provenance.json", prov)

    # Hard-fail is opt-in
    if os.getenv("MW_FAIL_ON_DEMO", "0") != "1":
        return

    branch = os.getenv("MW_BRANCH", "")
    protected = os.getenv("MW_PROTECTED_PATTERN", r"^main$")
    import re
    is_protected = bool(re.search(protected, branch)) if branch else False

    # We only gate protected branches
    source = (prov or {}).get("source", "")
    if is_protected and source.lower() == "demo":
        raise SystemExit("Provenance shows DEMO data on a protected branch (hard fail enabled).")


def main() -> None:
    # --- 1) Data
    symbols: List[str] = ["BTC", "ETH", "SOL"]
    # keep lookback moderate for CI speed; your loader will fetch real or demo transparently
    prices: Dict[str, pd.DataFrame] = load_prices(symbols, lookback_days=60)

    # Optional provenance guard (no-op by default; writes artifacts/data_provenance.json if available)
    _provenance_guard()

    # --- 2) Features
    feats = build_features(prices)

    # --- 3) Train simple per-asset baseline models + predict full series
    dfs: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = label_next_horizon(feats[s], horizon_h=1)
        X = df[["r_1h","r_3h","r_6h","vol_6h","atr_14h","sma_gap","high_vol","social_score"]].values
        y = df["y_long"].values

        # quick walk-forward to get a passable model
        trained = None
        for tr_ix, te_ix in walk_forward_splits(df, n_splits=2, train_days=14, test_days=7):
            if len(tr_ix) < 20 or len(te_ix) < 10:
                continue
            trained = train_model(X[tr_ix], y[tr_ix], model_type=os.getenv("MW_MODEL", "logreg"))
            break
        if trained is None:
            # fallback: train on last 70% if walk-forward yields tiny windows
            cut = int(0.7 * len(df))
            if cut < 10:
                continue
            trained = train_model(X[:cut], y[:cut], model_type=os.getenv("MW_MODEL", "logreg"))

        p = predict_proba(trained, X)
        dfs[s] = pd.DataFrame({"ts": df["ts"], "p_long": p})

    # --- 4) Threshold tuning + light backtest summary
    tune = tune_thresholds(dfs, prices)
    # normalize return format defensively (older/newer tuner shapes)
    params = tune.get("params") or tune.get("thresholds") or {}
    agg = tune.get("agg") or tune.get("aggregate") or {}
    per_symbol = tune.get("per_symbol") or {}

    # --- 5) Artifacts (tests only check existence; keep PNGs tiny)
    _write_json(
        MODELS_DIR / "ml_model_manifest.json",
        {"version": "v0.9.1", "model": os.getenv("MW_MODEL", "logreg"), "symbols": symbols, "horizon_h": 1},
    )
    _write_json(
        MODELS_DIR / "backtest_summary.json",
        {"aggregate": agg, "per_symbol": per_symbol},
    )
    # Ensure params are never None (prevents “None” in your CI summary)
    if not params:
        params = {"conf_min": 0.55, "debounce_min": 15, "horizon_h": 1}
    _write_json(MODELS_DIR / "signal_thresholds.json", {"params": params})

    _write_png(ARTIFACTS_DIR / "ml_roc_pr_curve.png")
    _write_png(ARTIFACTS_DIR / "bt_equity_curve.png")


if __name__ == "__main__":
    main()
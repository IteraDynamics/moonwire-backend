# scripts/ml/cv_eval.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from scripts.ml.data_loader import load_prices
from scripts.ml.feature_builder import build_features
from scripts.ml.labeler import label_next_horizon
from scripts.ml.splitter import walk_forward_splits
from scripts.ml.model_runner import train_model, predict_proba

# ------------------ knobs (via env) ------------------
SYMBOLS   = [s.strip().upper() for s in os.getenv("MW_ML_SYMBOLS", "BTC,ETH").split(",") if s.strip()]
LOOKBACK  = int(os.getenv("MW_ML_LOOKBACK_DAYS", "180"))
MODEL     = os.getenv("MW_ML_MODEL", "gb").strip().lower()   # gb|rf|logreg|hybrid
HORIZON_H = int(os.getenv("MW_HORIZON_H", "1"))
FOLDS     = int(os.getenv("MW_CV_FOLDS", "5"))

# gating for making a trade from a probability (out-of-sample row)
# - if present in governance, we’ll prefer that below; otherwise use this fallback:
CONF_MIN_FALLBACK = float(os.getenv("CV_CONF_MIN", os.getenv("REPLAY_CONF_MIN", "0.58")))

OUT_DIR = Path("artifacts"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "cv_results.json"
OUT_MD   = OUT_DIR / "cv_summary.md"


def _future_return(close: pd.Series, horizon_h: int) -> pd.Series:
    """
    Computes forward return over the next H hours: (close[t+H] / close[t] - 1).
    Aligns on 't' (same index as features/labels).
    """
    return (close.shift(-horizon_h) / close - 1.0).astype(float)


def _gov_conf(symbol: str) -> float:
    """
    Try to read models/governance_params.json for per-coin conf_min. Fallback to CONF_MIN_FALLBACK.
    """
    try:
        gpath = Path("models/governance_params.json")
        if gpath.exists():
            j = json.loads(gpath.read_text(encoding="utf-8") or "{}")
            row = j.get(symbol) or {}
            c = row.get("conf_min", None)
            if c is not None:
                return float(c)
    except Exception:
        pass
    return CONF_MIN_FALLBACK


@dataclass
class FoldStats:
    trades: int
    win_rate: float | None
    profit_factor: float | None

def _stats_from_signed_returns(sr: np.ndarray) -> FoldStats:
    """
    sr: array of realized signed trade returns (long: r, short: -r), for the set of *taken* trades in the fold
    """
    sr = np.array([x for x in sr if np.isfinite(x)], dtype=float)
    if sr.size == 0:
        return FoldStats(trades=0, win_rate=None, profit_factor=None)
    wins = sr[sr > 0.0]
    losses = sr[sr < 0.0]
    wr = float((sr > 0.0).mean()) if sr.size else None
    pf = None
    if losses.size > 0:
        pf = float(wins.sum() / abs(losses.sum())) if wins.size > 0 else 0.0
    elif wins.size > 0:
        pf = float("inf")
    else:
        pf = 0.0
    return FoldStats(trades=int(sr.size), win_rate=wr, profit_factor=pf)


def _cv_for_symbol(sym: str, feats_df: pd.DataFrame) -> Dict:
    """
    Run FOLDS walk-forward splits on a single symbol.
    For each fold: train on train split, score test split, gate by conf_min, compute signed returns.
    """
    # ensure required columns
    need = {"ts","close"}
    if not need.issubset(set(feats_df.columns)):
        raise ValueError(f"[{sym}] feature frame missing base columns: {need - set(feats_df.columns)}")

    # label next-horizon (adds 'y_long')
    L = label_next_horizon(feats_df, horizon_h=HORIZON_H)
    # feature matrix
    feature_cols = [c for c in L.columns if c not in {"ts","close","y_long"} and np.issubdtype(L[c].dtype, np.number)]
    X = L[feature_cols].values.astype(float)
    y = L["y_long"].values.astype(int)
    fwd_ret = _future_return(L["close"], HORIZON_H).values.astype(float)

    # governance conf threshold
    conf_min = _gov_conf(sym)

    per_fold: List[Dict] = []
    # walk-forward splits (~time-series k-fold). Keep n_splits=FOLDS for even coverage.
    fold_id = 0
    for tr_ix, te_ix in walk_forward_splits(L, n_splits=FOLDS, train_days=max(LOOKBACK // FOLDS, 10), test_days=max(LOOKBACK // (FOLDS*2), 5)):
        if len(tr_ix) < 20 or len(te_ix) < 5:
            continue
        fold_id += 1
        model = train_model(X[tr_ix], y[tr_ix], model_type=MODEL)
        p = predict_proba(model, X[te_ix])  # probability of long
        # trade rule:
        #   take trade if max(p, 1-p) >= conf_min
        #   direction = long if p>=0.5 else short
        direction_long = (p >= 0.5)
        confidence = np.where(direction_long, p, 1.0 - p)
        take = (confidence >= conf_min)

        # realized signed returns for taken rows in test
        r = fwd_ret[te_ix]
        signed = np.where(direction_long, r, -r)
        signed_taken = signed[take]

        fs = _stats_from_signed_returns(signed_taken)
        per_fold.append({
            "fold": fold_id,
            "test_len": int(len(te_ix)),
            "trades": fs.trades,
            "win_rate": fs.win_rate,
            "profit_factor": fs.profit_factor,
        })

    # aggregate
    trades = [f["trades"] for f in per_fold]
    wrs    = [f["win_rate"] for f in per_fold if f["win_rate"] is not None]
    pfs    = [f["profit_factor"] for f in per_fold if f["profit_factor"] is not None]

    agg = {
        "folds": len(per_fold),
        "trades_sum": int(sum(trades)),
        "win_rate_mean": float(np.mean(wrs)) if wrs else None,
        "win_rate_std":  float(np.std(wrs, ddof=1)) if len(wrs) > 1 else None,
        "profit_factor_mean": float(np.mean(pfs)) if pfs else None,
        "profit_factor_std":  float(np.std(pfs, ddof=1)) if len(pfs) > 1 else None,
        "conf_min_used": conf_min,
        "model": MODEL,
        "horizon_h": HORIZON_H,
        "lookback_days": LOOKBACK,
    }
    return {"symbol": sym, "feature_cols": feature_cols, "per_fold": per_fold, "aggregate": agg}


def main() -> Dict:
    prices = load_prices(SYMBOLS, lookback_days=LOOKBACK)
    feats = build_features(prices)

    results: Dict[str, Dict] = {}
    for sym in SYMBOLS:
        df = feats[sym]
        res = _cv_for_symbol(sym, df)
        results[sym] = res

    # pretty summary → md
    lines = []
    lines.append(f"# Cross-Validation (Time-Series, {FOLDS} folds)")
    lines.append(f"- Symbols: `{','.join(SYMBOLS)}`")
    lines.append(f"- Model: `{MODEL}` | Horizon: `{HORIZON_H}h` | Lookback: `{LOOKBACK}d`")
    lines.append("")

    for sym, res in results.items():
        agg = res["aggregate"]
        lines.append(f"## {sym}")
        lines.append(f"- Trades (sum over folds): **{agg['trades_sum']}**")
        lines.append(f"- Win-rate mean: **{agg['win_rate_mean']}**  std: {agg['win_rate_std']}")
        lines.append(f"- Profit-factor mean: **{agg['profit_factor_mean']}**  std: {agg['profit_factor_std']}")
        lines.append(f"- conf_min used: **{agg['conf_min_used']}**")
        lines.append(f"")
        lines.append("| fold | test_len | trades | win_rate | profit_factor |")
        lines.append("|-----:|---------:|-------:|---------:|--------------:|")
        for f in res["per_fold"]:
            lines.append(f"| {f['fold']} | {f['test_len']} | {f['trades']} | {f['win_rate']} | {f['profit_factor']} |")
        lines.append("")

    OUT_JSON.write_text(json.dumps({"symbols": SYMBOLS, "results": results}, indent=2), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # also print a compact one-liner to logs
    print("[cv_eval] wrote:", OUT_JSON, OUT_MD)
    return {"symbols": SYMBOLS, "results": results}


if __name__ == "__main__":
    main()
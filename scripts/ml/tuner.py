# scripts/ml/tuner.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

# Local backtest
from .backtest import run_backtest

# ----------------------------
# Config / Grids (wider to avoid zero-trade cold starts)
# ----------------------------
CONF_GRID: Tuple[float, ...] = (0.50, 0.52, 0.55, 0.58, 0.60, 0.62)
DEBOUNCE_GRID: Tuple[int, ...] = (15, 30, 45, 60)           # minutes
HORIZON_GRID: Tuple[int, ...] = (1, 2)                       # hours

FEES_BPS_DEFAULT: float = float(os.environ.get("MW_FEES_BPS", 1))
SLIPPAGE_BPS_DEFAULT: float = float(os.environ.get("MW_SLIPPAGE_BPS", 2))

# Constraints / objective
MIN_WINRATE: float = 0.60
SIGS_PER_DAY_MIN: float = 5.0
SIGS_PER_DAY_MAX: float = 10.0

# Output paths
MODELS_DIR = "models"
ART_MODELS_THRESH = os.path.join(MODELS_DIR, "signal_thresholds.json")
ART_MODELS_BTSUM = os.path.join(MODELS_DIR, "backtest_summary.json")


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@dataclass
class BTMetrics:
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    signals_per_day: float = 0.0
    n_trades: int = 0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _extract_metrics(bt: Dict[str, Any]) -> BTMetrics:
    """
    Accepts results from run_backtest. Supports both:
      - flat keys: win_rate, profit_factor, max_drawdown, signals_per_day, trades (list or int)
      - nested metrics: metrics={win_rate, profit_factor, ...}, plus trades at top-level
    """
    m = bt.get("metrics", {})
    # Prefer nested metrics (if present), else top-level
    win_rate = _safe_float(m.get("win_rate", bt.get("win_rate", 0.0)))
    profit_factor = _safe_float(m.get("profit_factor", bt.get("profit_factor", 0.0)))
    max_drawdown = _safe_float(m.get("max_drawdown", bt.get("max_drawdown", 0.0)))
    signals_per_day = _safe_float(m.get("signals_per_day", bt.get("signals_per_day", 0.0)))

    trades_obj = bt.get("trades", m.get("trades", []))
    if isinstance(trades_obj, int):
        n_trades = trades_obj
    else:
        try:
            n_trades = len(trades_obj)
        except Exception:
            n_trades = _safe_int(bt.get("n_trades", m.get("n_trades", 0)))

    return BTMetrics(
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        signals_per_day=signals_per_day,
        n_trades=n_trades,
    )


def _score_tuple(metrics: BTMetrics) -> Tuple[float, float, float]:
    """
    Sorting preference:
      1) higher profit_factor
      2) higher win_rate
      3) lower absolute max_drawdown (i.e., less drawdown is better)
    """
    return (
        _safe_float(metrics.profit_factor, 0.0),
        _safe_float(metrics.win_rate, 0.0),
        -abs(_safe_float(metrics.max_drawdown, 0.0)),
    )


def _normalize_frames(preds: pd.DataFrame, px: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align schemas and timestamps (UTC). Ensure we have 'ts' and 'p_long'.
    Drop unparsable timestamps.
    """
    preds = preds.copy()
    px = px.copy()

    # prediction column alias
    if "p_long" not in preds.columns:
        if "p" in preds.columns:
            preds["p_long"] = preds["p"]
        else:
            # If neither exists, fabricate neutral predictions (won't trigger at high conf)
            preds["p_long"] = 0.5

    # ensure ts present and UTC for both
    if "ts" not in preds.columns and "ts" in px.columns:
        # fallback: align length if possible
        preds["ts"] = np.asarray(px["ts"])[: len(preds)]
    preds["ts"] = pd.to_datetime(preds["ts"], utc=True, errors="coerce")
    if "ts" in px.columns:
        px["ts"] = pd.to_datetime(px["ts"], utc=True, errors="coerce")

    # drop NA timestamps / predictions
    preds = preds.dropna(subset=["ts", "p_long"])
    if "ts" in px.columns:
        px = px.dropna(subset=["ts"])

    return preds, px


def _aggregate(per_symbol: Dict[str, BTMetrics]) -> BTMetrics:
    # Aggregate as averages where sensible; signals/day sums across symbols.
    if not per_symbol:
        return BTMetrics()

    n = max(len(per_symbol), 1)
    agg = BTMetrics()
    agg.signals_per_day = sum(v.signals_per_day for v in per_symbol.values())
    agg.win_rate = sum(v.win_rate for v in per_symbol.values()) / n
    agg.profit_factor = sum(v.profit_factor for v in per_symbol.values()) / n
    agg.max_drawdown = sum(v.max_drawdown for v in per_symbol.values()) / n
    agg.n_trades = sum(v.n_trades for v in per_symbol.values())
    return agg


def _meets_constraints(agg: BTMetrics) -> bool:
    if agg.win_rate < MIN_WINRATE:
        return False
    if not (SIGS_PER_DAY_MIN <= agg.signals_per_day <= SIGS_PER_DAY_MAX):
        return False
    return True


def _metrics_to_dict(m: BTMetrics) -> Dict[str, Any]:
    return {
        "win_rate": round(_safe_float(m.win_rate), 4),
        "profit_factor": round(_safe_float(m.profit_factor), 4),
        "max_drawdown": round(_safe_float(m.max_drawdown), 4),
        "signals_per_day": round(_safe_float(m.signals_per_day), 2),
        "n_trades": _safe_int(m.n_trades),
    }


def tune_thresholds(pred_dfs: Dict[str, pd.DataFrame],
                    prices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Grid-search thresholds across symbols, backtest, and choose the best set
    meeting constraints (win-rate & signals/day). If none meet constraints,
    fall back to the best by score. Writes:
      - models/signal_thresholds.json
      - models/backtest_summary.json
    Returns a dict with keys: params, agg, per_symbol, grid_evaluations.
    """
    fees = FEES_BPS_DEFAULT
    slip = SLIPPAGE_BPS_DEFAULT

    evaluations: List[Dict[str, Any]] = []
    best_choice: Optional[Dict[str, Any]] = None
    best_choice_score: Optional[Tuple[float, float, float]] = None

    # Evaluate each grid point
    for conf in CONF_GRID:
        for debounce in DEBOUNCE_GRID:
            for horizon in HORIZON_GRID:
                symbol_metrics: Dict[str, BTMetrics] = {}
                # Per-symbol backtests
                for sym, preds in pred_dfs.items():
                    px = prices.get(sym)
                    if px is None or preds is None:
                        continue

                    preds_norm, px_norm = _normalize_frames(preds, px)

                    # Skip if nothing left after normalization
                    if preds_norm.empty or px_norm.empty:
                        symbol_metrics[sym] = BTMetrics()
                        continue

                    bt = run_backtest(
                        pred_df=preds_norm,
                        prices_df=px_norm,
                        conf_min=float(conf),
                        debounce_min=int(debounce),
                        horizon_h=int(horizon),
                        fees_bps=fees,
                        slippage_bps=slip,
                    )
                    symbol_metrics[sym] = _extract_metrics(bt)

                # Aggregate
                agg = _aggregate(symbol_metrics)
                meets = _meets_constraints(agg)
                score = _score_tuple(agg)

                # record this evaluation
                evaluations.append({
                    "params": {"conf_min": conf, "debounce_min": debounce, "horizon_h": horizon},
                    "agg": _metrics_to_dict(agg),
                    "per_symbol": {k: _metrics_to_dict(v) for k, v in symbol_metrics.items()},
                    "meets_constraints": meets,
                    "score_tuple": score,
                })

                # choose best
                if meets:
                    if (best_choice is None) or (score > best_choice_score):
                        best_choice = evaluations[-1]
                        best_choice_score = score

    # Fallback: if none meet constraints, take the best by score anyway
    if best_choice is None and evaluations:
        best_choice = max(evaluations, key=lambda r: r["score_tuple"])
        best_choice_score = best_choice["score_tuple"]

    # Build result payload
    if best_choice is None:
        # total failure; emit safe defaults
        chosen_params = {"conf_min": 0.55, "debounce_min": 30, "horizon_h": 1}
        agg = BTMetrics()
        per_symbol = {}
    else:
        chosen_params = best_choice["params"]
        # In case of any rounding / safety
        agg_dict = best_choice["agg"]
        per_symbol = best_choice["per_symbol"]
        agg = BTMetrics(
            win_rate=_safe_float(agg_dict.get("win_rate", 0.0)),
            profit_factor=_safe_float(agg_dict.get("profit_factor", 0.0)),
            max_drawdown=_safe_float(agg_dict.get("max_drawdown", 0.0)),
            signals_per_day=_safe_float(agg_dict.get("signals_per_day", 0.0)),
            n_trades=_safe_int(agg_dict.get("n_trades", 0)),
        )

    result = {
        "params": {
            "conf_min": float(chosen_params["conf_min"]),
            "debounce_min": int(chosen_params["debounce_min"]),
            "horizon_h": int(chosen_params["horizon_h"]),
        },
        "agg": _metrics_to_dict(agg),
        "per_symbol": per_symbol,
        "grid_evaluations": [
            {
                "params": e["params"],
                "agg": e["agg"],
                "meets_constraints": bool(e["meets_constraints"]),
            }
            for e in evaluations
        ],
    }

    # Persist artifacts
    _ensure_dir(ART_MODELS_THRESH)
    _ensure_dir(ART_MODELS_BTSUM)

    try:
        with open(ART_MODELS_THRESH, "w") as f:
            json.dump(result["params"], f, indent=2)
    except Exception:
        # Keep CI resilient
        pass

    # Summarize backtest for CI (aggregate + per_symbol)
    try:
        bt_summary = {
            "aggregate": {
                "win_rate": result["agg"]["win_rate"],
                "profit_factor": result["agg"]["profit_factor"],
                "max_drawdown": result["agg"]["max_drawdown"],
                "signals_per_day": result["agg"]["signals_per_day"],
            },
            "per_symbol": result["per_symbol"],
        }
        with open(ART_MODELS_BTSUM, "w") as f:
            json.dump(bt_summary, f, indent=2)
    except Exception:
        pass

    return result
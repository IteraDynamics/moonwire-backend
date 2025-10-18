# scripts/ml/tuner.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Backtest API we call:
# def run_backtest(pred_df: DataFrame, prices_df: DataFrame,
#                  conf_min: float, debounce_min: int, horizon_h: int,
#                  fees_bps: float = 1.0, slippage_bps: float = 2.0) -> dict
from .backtest import run_backtest


# ----------------------------- Config / Grid ----------------------------------

CONF_GRID: Tuple[float, ...] = (0.55, 0.58, 0.60, 0.62)
DEBOUNCE_GRID: Tuple[int, ...] = (15, 30, 45, 60)
HORIZON_GRID: Tuple[int, ...] = (1, 2)

TARGET_WINRATE: float = 0.60
TARGET_MIN_SIGS_PER_DAY: float = 5.0
TARGET_MAX_SIGS_PER_DAY: float = 10.0

DEFAULT_FEES_BPS: float = 1.0
DEFAULT_SLIPPAGE_BPS: float = 2.0

# ----------------------------- Data classes -----------------------------------


@dataclass
class BTMetrics:
    win_rate: float = float("nan")
    profit_factor: float = float("nan")
    max_drawdown: float = float("nan")
    signals_per_day: float = float("nan")
    n_trades: int = 0


# ----------------------------- Helpers ----------------------------------------


def _safe_len(x: Any) -> int:
    """Return len(x) if possible, else 0."""
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return 0


def _extract_metrics(bt: Dict[str, Any]) -> BTMetrics:
    """
    Normalize various backtest return shapes into BTMetrics.

    Expected (ideal) shape:
      {
        "metrics": {
          "win_rate": float,
          "profit_factor": float,
          "max_drawdown": float,
          "signals_per_day": float,
          "n_trades": int
        },
        "trades": List[...]
      }

    But callers may return flat keys; or "trades" may already be an int.
    """
    m = bt.get("metrics", {})
    # Prefer metrics.* first, then fallback to top-level or reasonable defaults.
    win = float(m.get("win_rate", bt.get("win_rate", float("nan"))))
    pf = float(m.get("profit_factor", bt.get("profit_factor", float("nan"))))
    mdd = float(m.get("max_drawdown", bt.get("max_drawdown", float("nan"))))
    spd = float(
        m.get(
            "signals_per_day",
            bt.get("signals_per_day", float("nan")),
        )
    )

    n_trades = m.get("n_trades", None)
    if n_trades is None:
        # "trades" could be a list or already an int
        trades_obj = bt.get("trades", [])
        if isinstance(trades_obj, int):
            n_trades = trades_obj
        else:
            n_trades = _safe_len(trades_obj)
    try:
        n_trades = int(n_trades)
    except Exception:
        n_trades = 0

    # Fill truly-missing numbers with neutral defaults so CI never crashes.
    if math.isnan(win):
        win = 0.5
    if math.isnan(pf):
        pf = 1.0
    if math.isnan(mdd):
        mdd = -0.10  # -10%
    if math.isnan(spd):
        spd = 0.0

    return BTMetrics(
        win_rate=win,
        profit_factor=pf,
        max_drawdown=mdd,
        signals_per_day=spd,
        n_trades=n_trades,
    )


def _agg_metrics(symbol_metrics: Dict[str, BTMetrics]) -> BTMetrics:
    """Aggregate per-symbol metrics. Weighted by n_trades where sensible."""
    if not symbol_metrics:
        return BTMetrics(0.5, 1.0, -0.10, 0.0, 0)

    total_trades = sum(m.n_trades for m in symbol_metrics.values())
    if total_trades == 0:
        # Equal-weight fallbacks if nothing traded
        wr = float(np.nanmean([m.win_rate for m in symbol_metrics.values()]))
        pf = float(np.nanmean([m.profit_factor for m in symbol_metrics.values()]))
        mdd = float(np.nanmean([m.max_drawdown for m in symbol_metrics.values()]))
        spd = float(np.nanmean([m.signals_per_day for m in symbol_metrics.values()]))
        return BTMetrics(wr, pf, mdd, spd, 0)

    # Weighted by trade count
    wr = float(
        sum(m.win_rate * m.n_trades for m in symbol_metrics.values()) / total_trades
    )
    pf = float(
        sum(m.profit_factor * m.n_trades for m in symbol_metrics.values())
        / total_trades
    )
    mdd = float(
        sum(m.max_drawdown * m.n_trades for m in symbol_metrics.values()) / total_trades
    )
    spd = float(
        sum(m.signals_per_day * m.n_trades for m in symbol_metrics.values())
        / total_trades
    )
    return BTMetrics(wr, pf, mdd, spd, total_trades)


def _within_objectives(agg: BTMetrics) -> bool:
    """Check objective constraints."""
    if agg.win_rate < TARGET_WINRATE:
        return False
    if not (TARGET_MIN_SIGS_PER_DAY <= agg.signals_per_day <= TARGET_MAX_SIGS_PER_DAY):
        return False
    return True


def _tie_break_key(agg: BTMetrics) -> Tuple[float, float, float, float]:
    """
    Sort key for picking the best candidate when multiple pass (or none pass).

    Priority:
      1) Higher profit factor
      2) Lower (less negative) max drawdown
      3) Higher win rate
      4) Signals/day closeness to the middle of the target band (7.5)
    """
    center = (TARGET_MIN_SIGS_PER_DAY + TARGET_MAX_SIGS_PER_DAY) / 2.0
    # Negative abs distance so that "closer is better" => larger sort key
    sig_score = -abs(agg.signals_per_day - center)
    return (agg.profit_factor, -abs(agg.max_drawdown), agg.win_rate, sig_score)


# ----------------------------- Public API -------------------------------------


def tune_thresholds(
    pred_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, pd.DataFrame],
    fees_bps: float = DEFAULT_FEES_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> Dict[str, Any]:
    """
    Grid-search thresholds & debounce; aggregate across symbols; pick the best.

    Returns:
      {
        "params": {"conf_min": float, "debounce_min": int, "horizon_h": int},
        "agg": {"win_rate": float, "profit_factor": float, "max_drawdown": float, "signals_per_day": float},
        "per_symbol": { "BTC": {...}, ... },

        # Legacy fields kept for compatibility:
        "conf_min": ...,
        "debounce_min": ...,
        "horizon_h": ...,
        "aggregate": {...}
      }
    Also writes:
      models/signal_thresholds.json
      models/backtest_summary.json
    """
    symbols = sorted([s for s in pred_dfs.keys() if s in prices])

    if not symbols:
        # Nothing to tune; persist defaults and return a neutral payload.
        params = {"conf_min": 0.60, "debounce_min": 30, "horizon_h": 1}
        agg = {
            "win_rate": 0.50,
            "profit_factor": 1.00,
            "max_drawdown": -0.10,
            "signals_per_day": 0.0,
        }
        os.makedirs("models", exist_ok=True)
        with open("models/signal_thresholds.json", "w") as f:
            json.dump(params, f, indent=2)
        with open("models/backtest_summary.json", "w") as f:
            json.dump({"aggregate": agg, "per_symbol": {}}, f, indent=2)
        return {
            "params": params,
            "agg": agg,
            "per_symbol": {},
            "conf_min": params["conf_min"],
            "debounce_min": params["debounce_min"],
            "horizon_h": params["horizon_h"],
            "aggregate": agg,
        }

    best_candidate = None  # type: ignore[var-annotated]
    best_key = None  # type: ignore[var-annotated]

    # Grid search
    for conf_min in CONF_GRID:
        for debounce_min in DEBOUNCE_GRID:
            for horizon_h in HORIZON_GRID:
                per_symbol_metrics: Dict[str, BTMetrics] = {}

                for sym in symbols:
                    preds = pred_dfs[sym]
                    px = prices[sym]

                    # Defensive: ensure columns exist
                    if "p_long" not in preds.columns:
                        # If no probas, skip symbol for this combo
                        per_symbol_metrics[sym] = BTMetrics(0.5, 1.0, -0.10, 0.0, 0)
                        continue
                    if "ts" not in preds.columns:
                        # Align by index if no ts; copy over to be safe
                        preds = preds.copy()
                        preds["ts"] = np.arange(len(preds))

                    # Run backtest for this symbol/setting
                    bt = run_backtest(
                        pred_df=preds,
                        prices_df=px,
                        conf_min=conf_min,
                        debounce_min=debounce_min,
                        horizon_h=horizon_h,
                        fees_bps=fees_bps,
                        slippage_bps=slippage_bps,
                    )

                    per_symbol_metrics[sym] = _extract_metrics(bt)

                agg = _agg_metrics(per_symbol_metrics)

                # Determine if it passes objectives
                passes = _within_objectives(agg)
                key = _tie_break_key(agg)

                # Pick best among those passing objectives; if none pass, still pick best key.
                if best_candidate is None:
                    best_candidate = (conf_min, debounce_min, horizon_h, agg, per_symbol_metrics)
                    best_key = (passes, key)
                else:
                    # Prefer a passing candidate over a failing one.
                    if passes and not best_key[0]:
                        best_candidate = (conf_min, debounce_min, horizon_h, agg, per_symbol_metrics)
                        best_key = (passes, key)
                    elif passes == best_key[0]:
                        # Same pass/fail status — choose by key.
                        if key > best_key[1]:
                            best_candidate = (conf_min, debounce_min, horizon_h, agg, per_symbol_metrics)
                            best_key = (passes, key)

    # Unpack best
    best_conf, best_debounce, best_h, agg_metrics, per_symbol_metrics = best_candidate  # type: ignore[misc]

    # ---- Build result (new shape) + keep legacy fields for compatibility ----
    params = {
        "conf_min": best_conf,
        "debounce_min": best_debounce,
        "horizon_h": best_h,
    }

    agg = {
        "win_rate": float(agg_metrics.win_rate),
        "profit_factor": float(agg_metrics.profit_factor),
        "max_drawdown": float(agg_metrics.max_drawdown),
        "signals_per_day": float(agg_metrics.signals_per_day),
    }

    per_symbol_out: Dict[str, Dict[str, float]] = {
        s: {
            "win_rate": float(m.win_rate),
            "profit_factor": float(m.profit_factor),
            "max_drawdown": float(m.max_drawdown),
            "signals_per_day": float(m.signals_per_day),
            "n_trades": int(m.n_trades),
        }
        for s, m in per_symbol_metrics.items()
    }

    result = {
        # New, test-expected keys
        "params": params,
        "agg": agg,
        "per_symbol": per_symbol_out,
        # Legacy flat fields so any existing callers don’t break
        "conf_min": params["conf_min"],
        "debounce_min": params["debounce_min"],
        "horizon_h": params["horizon_h"],
        "aggregate": agg,
    }

    # Persist thresholds for runtime consumption
    os.makedirs("models", exist_ok=True)
    with open("models/signal_thresholds.json", "w") as f:
        json.dump(params, f, indent=2)

    # Save a full backtest summary (aggregate + per-symbol)
    with open("models/backtest_summary.json", "w") as f:
        json.dump({"aggregate": agg, "per_symbol": per_symbol_out}, f, indent=2)

    return result
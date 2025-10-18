# scripts/ml/tuner.py
# -*- coding: utf-8 -*-

# --- path shim: allow running/importing from arbitrary CWDs ---
from pathlib import Path
import sys as _sys
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # repo root
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ----------------------------------------------------------------

from typing import Dict, Any, Iterable, Tuple, Optional
import os
import json
import math
import numpy as np
import pandas as pd

# Primary import (works once the path shim is in place)
from scripts.ml.backtest import run_backtest  # type: ignore


# ---------- constants / paths ----------
MODELS_DIR = (_ROOT / "models")
ARTIFACTS_DIR = (_ROOT / "artifacts")
LOGS_DIR = (_ROOT / "logs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- small utils ----------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, (list, tuple)):
            return len(x)
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Safely get obj.name or obj[name] or default."""
    if obj is None:
        return default
    if hasattr(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _safe_trades_list(bt: Dict[str, Any]) -> Iterable:
    t = bt.get("trades", [])
    if t is None:
        return []
    if isinstance(t, (list, tuple)):
        return t
    # occasionally some backtests return a count — normalize to empty list
    return []


def _extract_metrics(bt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts flexible shapes from run_backtest and extracts consistent metrics.
    Supports metrics at root or under 'metrics', and trades as list or count.
    Computes wins/losses if only trades are available.
    """
    m = bt.get("metrics", {})

    # base pulls
    n_trades = _as_int(_get_attr_or_key(m, "n_trades", bt.get("trades", 0)), 0)
    win_rate = _as_float(bt.get("win_rate", m.get("win_rate", 0.0)), 0.0)
    profit_factor = _as_float(bt.get("profit_factor", m.get("profit_factor", 0.0)), 0.0)
    max_drawdown = _as_float(bt.get("max_drawdown", m.get("max_drawdown", 0.0)), 0.0)
    signals_per_day = _as_float(bt.get("signals_per_day", m.get("signals_per_day", 0.0)), 0.0)

    wins = _as_int(m.get("wins", bt.get("wins", 0)), 0)
    losses = _as_int(m.get("losses", bt.get("losses", 0)), 0)

    # Trades present? compute wins/losses if missing
    trades = _safe_trades_list(bt)
    if trades:
        if wins == 0 and losses == 0:
            w = 0
            l = 0
            for t in trades:
                pnl = _get_attr_or_key(t, "pnl", None)
                if pnl is None:
                    pnl = _get_attr_or_key(t, "pnl_pct", 0.0)
                try:
                    pnl = float(pnl)
                except Exception:
                    pnl = 0.0
                if pnl > 0:
                    w += 1
                elif pnl < 0:
                    l += 1
            wins, losses = w, l
        n_trades = max(n_trades, len(trades))

        # profit factor fallback if missing
        if profit_factor == 0.0:
            gross_profit = 0.0
            gross_loss = 0.0
            for t in trades:
                pnl = _get_attr_or_key(t, "pnl", None)
                if pnl is None:
                    pnl = _get_attr_or_key(t, "pnl_pct", 0.0)
                try:
                    pnl = float(pnl)
                except Exception:
                    pnl = 0.0
                if pnl >= 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            elif gross_profit > 0:
                profit_factor = float("inf")
            else:
                profit_factor = 0.0

    # win rate fallback
    if win_rate == 0.0 and (wins + losses) > 0:
        win_rate = wins / float(wins + losses)

    return {
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "signals_per_day": float(signals_per_day),
        "wins": int(wins),
        "losses": int(losses),
    }


def _aggregate_metrics(per_symbol: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Simple aggregate: mean of rates/pf/mdd/spd; total n_trades."""
    if not per_symbol:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "signals_per_day": 0.0,
        }
    df = pd.DataFrame(per_symbol).T.fillna(0.0)
    agg = {
        "n_trades": int(df["n_trades"].sum()),
        "win_rate": float(df["win_rate"].mean()),
        "profit_factor": float(df["profit_factor"].replace(np.inf, np.nan).fillna(0.0).mean()),
        "max_drawdown": float(df["max_drawdown"].mean()),
        "signals_per_day": float(df["signals_per_day"].mean()),
    }
    return agg


def _objective_ok(agg: Dict[str, Any]) -> bool:
    wr = _as_float(agg.get("win_rate", 0.0))
    spd = _as_float(agg.get("signals_per_day", 0.0))
    return (wr >= 0.60) and (spd >= 5.0) and (spd <= 10.0)


def _grid() -> Tuple[Iterable[float], Iterable[int], Iterable[int]]:
    confs = [0.55, 0.58, 0.60, 0.62]
    debounces = [15, 30, 45, 60]
    horizons = [1, 2]
    return confs, debounces, horizons


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, default))
        return v
    except Exception:
        return default


def tune_thresholds(
    pred_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Grid-search thresholds using backtest outputs.
    Always writes:
      - models/signal_thresholds.json
      - models/backtest_summary.json
    Returns a dict containing:
      { "params": {...}, "agg": {...}, "per_symbol": {...} }
    """
    fees_bps = _env_float("MW_FEES_BPS", 1.0)
    slippage_bps = _env_float("MW_SLIPPAGE_BPS", 2.0)

    best: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_per_symbol: Optional[Dict[str, Any]] = None

    confs, debounces, horizons = _grid()

    # Try full objective first, then relax if needed
    def score(agg: Dict[str, Any]) -> Tuple[float, float]:
        # higher PF, lower MDD
        pf = _as_float(agg.get("profit_factor", 0.0))
        mdd = _as_float(agg.get("max_drawdown", 0.0))
        return (pf, -mdd)

    # Iterate grid
    for conf in confs:
        for db in debounces:
            for hz in horizons:
                per_sym_metrics: Dict[str, Any] = {}
                for sym, pdf in pred_dfs.items():
                    p = prices.get(sym)
                    if p is None or pdf is None or pdf.empty:
                        per_sym_metrics[sym] = {
                            "n_trades": 0,
                            "win_rate": 0.0,
                            "profit_factor": 0.0,
                            "max_drawdown": 0.0,
                            "signals_per_day": 0.0,
                        }
                        continue

                    # Backtest: expects aligned inputs
                    bt = run_backtest(
                        pred_df=pdf,
                        prices_df=p,
                        conf_min=float(conf),
                        debounce_min=int(db),
                        horizon_h=int(hz),
                        fees_bps=float(fees_bps),
                        slippage_bps=float(slippage_bps),
                    )
                    per_sym_metrics[sym] = _extract_metrics(bt)

                agg = _aggregate_metrics(per_sym_metrics)
                params = {"conf_min": conf, "debounce_min": db, "horizon_h": hz}

                # choose if meets hard objective and better than current best
                if _objective_ok(agg):
                    if best is None or score(agg) > score(best):  # type: ignore
                        best = agg
                        best_params = params
                        best_per_symbol = per_sym_metrics

    # If nothing met objective, choose the best by PF then MDD that has *any* trades
    if best is None:
        for conf in confs:
            for db in debounces:
                for hz in horizons:
                    per_sym_metrics: Dict[str, Any] = {}
                    total_trades = 0
                    for sym, pdf in pred_dfs.items():
                        p = prices.get(sym)
                        if p is None or pdf is None or pdf.empty:
                            per_sym_metrics[sym] = {
                                "n_trades": 0,
                                "win_rate": 0.0,
                                "profit_factor": 0.0,
                                "max_drawdown": 0.0,
                                "signals_per_day": 0.0,
                            }
                            continue
                        bt = run_backtest(
                            pred_df=pdf,
                            prices_df=p,
                            conf_min=float(conf),
                            debounce_min=int(db),
                            horizon_h=int(hz),
                            fees_bps=float(fees_bps),
                            slippage_bps=float(slippage_bps),
                        )
                        m = _extract_metrics(bt)
                        total_trades += m.get("n_trades", 0)
                        per_sym_metrics[sym] = m

                    if total_trades == 0:
                        continue  # don't pick a combo with zero trades

                    agg = _aggregate_metrics(per_sym_metrics)
                    params = {"conf_min": conf, "debounce_min": db, "horizon_h": hz}

                    if best is None or score(agg) > score(best):  # type: ignore
                        best = agg
                        best_params = params
                        best_per_symbol = per_sym_metrics

    # Fallback: still nothing? write a conservative default so CI always has an artifact
    if best is None or best_params is None or best_per_symbol is None:
        best_params = {"conf_min": 0.55, "debounce_min": 15, "horizon_h": 1}
        best = {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "signals_per_day": 0.0,
        }
        best_per_symbol = {}

    # Persist artifacts
    thresholds_path = MODELS_DIR / "signal_thresholds.json"
    summary_path = MODELS_DIR / "backtest_summary.json"

    _write_json(thresholds_path, best_params)
    _write_json(summary_path, {
        "aggregate": {
            "n_trades": int(best.get("n_trades", 0)),
            "win_rate": float(best.get("win_rate", 0.0)),
            "profit_factor": float(best.get("profit_factor", 0.0)),
            "max_drawdown": float(best.get("max_drawdown", 0.0)),
            "signals_per_day": float(best.get("signals_per_day", 0.0)),
        },
        "per_symbol": best_per_symbol,
    })

    return {
        "params": best_params,
        "agg": best,
        "per_symbol": best_per_symbol,
        "paths": {
            "signal_thresholds": str(thresholds_path),
            "backtest_summary": str(summary_path),
        }
    }


if __name__ == "__main__":
    # Optional: tiny smoke run if someone executes this file directly.
    # We won't run anything heavy here—just print a friendly note.
    print("[tuner] Ready. Import tune_thresholds(pred_dfs, prices) to run the grid search.")
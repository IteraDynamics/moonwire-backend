# scripts/ml/tuner.py
# ---------------------------------------------------------------------
# Threshold grid search over model probabilities -> trading signals.
# Writes:
#   - models/signal_thresholds.json
#   - models/backtest_summary.json
# and returns a dict containing {'params', 'agg', 'per_symbol'}.
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# --- universal import shim + paths (works in pytest, CI, and python __main__) ---
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# -------------------------------------------------------------------------------

# Local import after path shim
from scripts.ml.backtest import run_backtest  # type: ignore

# IO locations
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
ART_DIR = ROOT / "artifacts"
for d in (MODELS_DIR, LOGS_DIR, ART_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------- helpers & typing ---------------------------------


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.floating, np.integer)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
        return default
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            return int(round(x))
        if isinstance(x, str):
            return int(x.strip())
        if isinstance(x, (list, tuple)):
            return len(x)  # interpret as count
        return default
    except Exception:
        return default


def _trade_to_dict(t: Any) -> Dict[str, Any]:
    """Accept dataclass / object / dict and return a plain dict with at least pnl."""
    if t is None:
        return {"pnl": 0.0}
    if isinstance(t, dict):
        return t
    if is_dataclass(t):
        try:
            return asdict(t)  # type: ignore
        except Exception:
            pass
    # generic object: try attrs
    out = {}
    for key in ("pnl", "pnl_pct", "entry_ts", "exit_ts", "symbol", "side"):
        if hasattr(t, key):
            out[key] = getattr(t, key)
    if "pnl" not in out:
        out["pnl"] = 0.0
    return out


def _extract_metrics(bt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize varying shapes from run_backtest into a single metrics dict.
    Supports metrics at root or under 'metrics', and trades as list or count.
    """
    m = bt.get("metrics", {})
    trades_obj = bt.get("trades", m.get("trades", []))

    # n_trades
    n_trades = _as_int(m.get("n_trades", trades_obj), 0)

    # wins / losses
    wins = _as_int(m.get("wins", bt.get("wins", 0)), 0)
    losses = _as_int(m.get("losses", bt.get("losses", 0)), 0)

    # If wins/losses not provided but we have trades, derive by pnl
    if (wins + losses == 0) and isinstance(trades_obj, (list, tuple)) and len(trades_obj) > 0:
        w = 0
        l = 0
        for t in trades_obj:
            td = _trade_to_dict(t)
            pnl = _as_float(td.get("pnl", td.get("pnl_pct", 0.0)), 0.0)
            if pnl > 0:
                w += 1
            elif pnl < 0:
                l += 1
        wins, losses = w, l
        n_trades = max(n_trades, w + l)

    # win_rate
    win_rate = _as_float(m.get("win_rate", bt.get("win_rate", 0.0)), 0.0)
    if win_rate == 0.0 and n_trades > 0 and (wins + losses) > 0:
        win_rate = wins / max(1, wins + losses)

    # profit_factor / max_drawdown / signals_per_day
    profit_factor = _as_float(m.get("profit_factor", bt.get("profit_factor", 0.0)), 0.0)
    max_drawdown = _as_float(m.get("max_drawdown", bt.get("max_drawdown", 0.0)), 0.0)
    spd = _as_float(m.get("signals_per_day", bt.get("signals_per_day", 0.0)), 0.0)

    return {
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "signals_per_day": float(spd),
    }


def _agg_metrics(metrics_per_symbol: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate over symbols with sensible weighting:
      - win_rate weighted by n_trades
      - signals_per_day is sum across symbols
      - profit_factor: weighted mean by n_trades
      - max_drawdown: worst (min)
    """
    if not metrics_per_symbol:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "signals_per_day": 0.0,
            "n_trades": 0,
        }

    total_trades = sum(_as_int(v.get("n_trades", 0), 0) for v in metrics_per_symbol.values())
    win_rate = 0.0
    if total_trades > 0:
        win_rate = sum(
            _as_float(v.get("win_rate", 0.0), 0.0) * _as_int(v.get("n_trades", 0), 0)
            for v in metrics_per_symbol.values()
        ) / max(1, total_trades)

    signals_per_day = sum(_as_float(v.get("signals_per_day", 0.0), 0.0) for v in metrics_per_symbol.values())

    if total_trades > 0:
        profit_factor = sum(
            _as_float(v.get("profit_factor", 0.0), 0.0) * _as_int(v.get("n_trades", 0), 0)
            for v in metrics_per_symbol.values()
        ) / max(1, total_trades)
    else:
        profit_factor = 0.0

    max_drawdown = min(_as_float(v.get("max_drawdown", 0.0), 0.0) for v in metrics_per_symbol.values())

    return {
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "signals_per_day": float(signals_per_day),
        "n_trades": int(total_trades),
    }


def _objective_rank(agg: Dict[str, Any]) -> Tuple[int, float, float]:
    """
    Primary feasibility flags then tie-breakers:
      1) feasibility flag (0=good meets constraints, 1=fails)
      2) -profit_factor (higher better)
      3) max_drawdown (less negative is better -> higher)
    Sort ascending on this tuple.
    """
    wr = _as_float(agg.get("win_rate", 0.0))
    spd = _as_float(agg.get("signals_per_day", 0.0))
    pf = _as_float(agg.get("profit_factor", 0.0))
    mdd = _as_float(agg.get("max_drawdown", 0.0))

    feasible = 0 if (wr >= 0.60 and 5.0 <= spd <= 10.0) else 1
    return (feasible, -pf, mdd)


def _grid_from_env(name: str, default: Iterable, cast=float) -> List:
    """
    Parse CSV grid from env, e.g. "0.55,0.58,0.60" or "15,30,60".
    """
    v = os.getenv(name)
    if not v:
        return list(default)
    out: List = []
    for tok in v.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(cast(t))
        except Exception:
            continue
    return out or list(default)


# ----------------------------- main API ---------------------------------------


def tune_thresholds(
    pred_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, pd.DataFrame],
    conf_grid: Iterable[float] | None = None,
    debounce_grid: Iterable[int] | None = None,
    horizon_grid: Iterable[int] | None = None,
    fees_bps: float | None = None,
    slippage_bps: float | None = None,
) -> Dict[str, Any]:
    """
    Grid-search thresholds over per-symbol predictions.
    Returns dict with:
      {
        "params": {"conf_min": ..., "debounce_min": ..., "horizon_h": ...},
        "agg": {...},
        "per_symbol": {SYM: {...}, ...}
      }
    and writes models/signal_thresholds.json and models/backtest_summary.json
    """
    # defaults for grid (used when env not set)
    # --- fallback grids used when env vars are not set ---
    default_conf = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    default_db   = [10, 15, 20, 30, 45, 60]
    default_hz   = [1, 2, 3]
    
    conf_grid     = list(conf_grid) if conf_grid is not None else _grid_from_env("MW_CONF_GRID", default_conf, float)
    debounce_grid = list(debounce_grid) if debounce_grid is not None else _grid_from_env("MW_DEBOUNCE_GRID_MIN", default_db, int)
    horizon_grid  = list(horizon_grid) if horizon_grid is not None else _grid_from_env("MW_HORIZON_GRID_H", default_hz, int)

    strict = (os.getenv("TUNER_STRICT", "0").strip() in ("1","true","yes","on"))

    # grids & params (env overrides allowed)
    conf_grid = list(conf_grid) if conf_grid is not None else _grid_from_env("MW_TUNE_CONF", default_conf, float)
    debounce_grid = list(debounce_grid) if debounce_grid is not None else _grid_from_env("MW_TUNE_DEBOUNCE", default_db, int)
    horizon_grid = list(horizon_grid) if horizon_grid is not None else _grid_from_env("MW_TUNE_HORIZON", default_hz, int)
    fees_bps = 1.0 if fees_bps is None else fees_bps
    slippage_bps = 2.0 if slippage_bps is None else slippage_bps

    # Normalize/clean prediction frames
    symbols = sorted(pred_dfs.keys())
    pred_norm: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = pred_dfs[sym].copy()
        # expected columns: ts, p_long
        if "ts" not in df.columns:
            # try index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "ts"})
            else:
                df["ts"] = np.arange(len(df))
        if "p_long" not in df.columns:
            # best-effort: any prob-like column?
            for c in df.columns:
                if c.lower() in ("prob", "proba", "p", "plong", "p_long"):
                    df = df.rename(columns={c: "p_long"})
                    break
        # enforce ordering
        df = df.sort_values("ts").reset_index(drop=True)
        pred_norm[sym] = df[["ts", "p_long"]].copy()

    # Run grid search
    candidates: List[Tuple[Tuple[float, int, int], Dict[str, Any], Dict[str, Any]]] = []
    for conf_min in conf_grid:
        for debounce_min in debounce_grid:
            for horizon_h in horizon_grid:
                per_symbol_metrics: Dict[str, Dict[str, Any]] = {}

                for sym in symbols:
                    p_df = pred_norm.get(sym)
                    px_df = prices.get(sym)
                    if p_df is None or px_df is None or p_df.empty or px_df.empty:
                        per_symbol_metrics[sym] = {
                            "n_trades": 0,
                            "win_rate": 0.0,
                            "profit_factor": 0.0,
                            "max_drawdown": 0.0,
                            "signals_per_day": 0.0,
                        }
                        continue

                    bt = run_backtest(
                        pred_df=p_df,
                        prices_df=px_df,
                        conf_min=float(conf_min),
                        debounce_min=int(debounce_min),
                        horizon_h=int(horizon_h),
                        fees_bps=float(fees_bps),
                        slippage_bps=float(slippage_bps),
                    )
                    per_symbol_metrics[sym] = _extract_metrics(bt)

                agg = _agg_metrics(per_symbol_metrics)
                key = (float(conf_min), int(debounce_min), int(horizon_h))
                candidates.append((key, agg, per_symbol_metrics))

    # Choose best by objective
    if len(candidates) == 0:
        chosen = ((0.55, 15, 1),
                  {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0, "n_trades": 0},
                  {})
    else:
        candidates.sort(key=lambda x: _objective_rank(x[1]))
        chosen = candidates[0]

    (c_conf, c_db, c_h), agg, per_symbol = chosen

    # Write artifacts
    thresholds_payload = {
        "conf_min": c_conf,
        "debounce_min": c_db,
        "horizon_h": c_h,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELS_DIR / "signal_thresholds.json", "w") as f:
        json.dump(thresholds_payload, f, indent=2)

    backtest_summary_payload = {
        "aggregate": {
            "win_rate": round(_as_float(agg.get("win_rate", 0.0)), 4),
            "profit_factor": round(_as_float(agg.get("profit_factor", 0.0)), 4),
            "max_drawdown": round(_as_float(agg.get("max_drawdown", 0.0)), 4),
            "signals_per_day": round(_as_float(agg.get("signals_per_day", 0.0)), 4),
            "n_trades": _as_int(agg.get("n_trades", 0), 0),
        },
        "per_symbol": {
            s: {
                "win_rate": round(_as_float(m.get("win_rate", 0.0)), 4),
                "profit_factor": round(_as_float(m.get("profit_factor", 0.0)), 4),
                "max_drawdown": round(_as_float(m.get("max_drawdown", 0.0)), 4),
                "signals_per_day": round(_as_float(m.get("signals_per_day", 0.0)), 4),
                "n_trades": _as_int(m.get("n_trades", 0), 0),
            }
            for s, m in per_symbol.items()
        },
    }
    with open(MODELS_DIR / "backtest_summary.json", "w") as f:
        json.dump(backtest_summary_payload, f, indent=2)

    # Return structure that tests and summary blocks expect
    return {
        "params": {"conf_min": c_conf, "debounce_min": c_db, "horizon_h": c_h},
        "agg": agg,
        "per_symbol": per_symbol,
    }


# Allow running by itself for quick smoke test
if __name__ == "__main__":
    # Minimal synthetic input to prove it runs without the rest of the pipeline.
    rng = pd.date_range("2024-01-01", periods=200, freq="H", tz="UTC")
    demo_px = pd.DataFrame({"ts": rng, "close": np.cumsum(np.random.randn(len(rng))) + 100})
    demo_pred = pd.DataFrame({"ts": rng, "p_long": np.clip(np.random.rand(len(rng)) * 0.2 + 0.4, 0, 1)})

    preds = {"BTC": demo_pred.copy(), "ETH": demo_pred.copy(), "SOL": demo_pred.copy()}
    prices = {"BTC": demo_px.copy(), "ETH": demo_px.copy(), "SOL": demo_px.copy()}

    res = tune_thresholds(preds, prices)
    print(json.dumps(res["params"], indent=2))
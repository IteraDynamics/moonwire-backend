# scripts/perf/paper_trader.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import math

# Matplotlib (headless)
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None

from scripts.summary_sections.common import ensure_dir, _iso
from .performance_metrics import compute_metrics


# ------------------------
# Helpers
# ------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _synthesize_prices(symbols: List[str], hours: int = 72) -> Dict[str, List[Tuple[str, float]]]:
    """Deterministic synthetic price series (hourly) for demo."""
    now = _now_utc().replace(minute=0, second=0)
    ts = [now - timedelta(hours=h) for h in reversed(range(hours + 1))]
    base = {"BTC": 60000.0, "ETH": 3000.0, "SOL": 150.0}
    out: Dict[str, List[Tuple[str, float]]] = {}
    for s in symbols:
        b = base.get(s.upper(), 100.0)
        series = []
        for i, t in enumerate(ts):
            # gentle wave + slight drift
            px = b * (1.0 + 0.02 * math.sin(i / 6.28)) * (1.0 + 0.0005 * i)
            series.append((_iso(t), round(px, 2)))
        out[s.upper()] = series
    return out


def _extract_prices_from_market_context(mc: Dict, symbols: List[str]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Expected shape (tolerant):
    {
      "window_hours": 72,
      "coins": [{"symbol":"BTC","prices":[["2025-10-12T00:00Z", 60000.0], ...]}, ...]
    }
    We accept a few variants and try to coerce.
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    syms_upper = [s.upper() for s in symbols]
    coins = []

    if isinstance(mc, dict):
        if "coins" in mc and isinstance(mc["coins"], list):
            coins = mc["coins"]
        elif "data" in mc and isinstance(mc["data"], list):
            coins = mc["data"]

    for c in coins:
        sym = (c.get("symbol") or c.get("id") or "").upper()
        if sym and sym in syms_upper:
            series: List[Tuple[str, float]] = []
            prices = c.get("prices") or c.get("series") or []
            for p in prices:
                try:
                    ts, px = p[0], float(p[1])
                    # normalize ISO
                    ts_iso = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=timezone.utc).isoformat()
                    series.append((ts_iso, px))
                except Exception:
                    continue
            if series:
                out[sym] = series
    return out


def _align_price_at_or_after(series: List[Tuple[str, float]], ts_iso: str) -> Optional[Tuple[str, float]]:
    """Return first (ts, px) at or after ts_iso; if none, return last known."""
    if not series:
        return None
    t0 = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    for ts, px in series:
        if datetime.fromisoformat(ts.replace("Z", "+00:00")) >= t0:
            return ts, px
    return series[-1]


def _time_add_iso(ts_iso: str, minutes: int) -> str:
    dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    return _iso(dt + timedelta(minutes=minutes))


def _side_to_sign(side: str) -> int:
    side = (side or "").lower()
    if side in ("long", "buy", "bull", "up"):
        return 1
    if side in ("short", "sell", "bear", "down"):
        return -1
    return 1  # default long


# ------------------------
# Core
# ------------------------

@dataclass
class _Cfg:
    mode: str
    symbols: List[str]
    horizon_min: int
    slippage_bps: float
    fees_bps: float
    capital: float
    risk_free: float
    lookback_h: int


def _load_cfg() -> _Cfg:
    return _Cfg(
        mode=os.getenv("MW_PERF_MODE", "backtest").lower(),
        symbols=[s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()],
        horizon_min=int(os.getenv("MW_PERF_HORIZON_MIN", "60")),
        slippage_bps=float(os.getenv("MW_PERF_SLIPPAGE_BPS", "2")),
        fees_bps=float(os.getenv("MW_PERF_FEES_BPS", "1")),
        capital=float(os.getenv("MW_PERF_CAPITAL", "100000")),
        risk_free=float(os.getenv("MW_PERF_RISK_FREE", "0.0")),
        lookback_h=int(os.getenv("MW_PERF_LOOKBACK_H", "72")),
    )


def _load_signals(logs_dir: Path, symbols: List[str], lookback_h: int, is_demo: bool) -> List[Dict]:
    path = logs_dir / "signals.jsonl"
    rows = _read_jsonl(path)
    if rows:
        # filter by symbols & lookback
        cutoff = _now_utc() - timedelta(hours=lookback_h)
        out = []
        for r in rows:
            try:
                sym = (r.get("symbol") or "").upper()
                ts = datetime.fromisoformat(str(r["ts"]).replace("Z", "+00:00"))
                if symbols and sym not in symbols:
                    continue
                if ts < cutoff:
                    continue
                out.append({
                    "id": r.get("id"),
                    "ts": _iso(ts),
                    "symbol": sym,
                    "direction": r.get("direction") or ("long" if float(r.get("confidence", 0.0)) >= 0 else "short"),
                    "confidence": float(r.get("confidence", 0.0)),
                    "price": float(r.get("price", 0.0)) if r.get("price") is not None else None,
                })
            except Exception:
                continue
        if out:
            return sorted(out, key=lambda x: (x["symbol"], x["ts"]))
    # demo synth if missing/empty
    if is_demo:
        now = _now_utc().replace(minute=0, second=0)
        out = []
        i = 0
        for sym in symbols:
            for h in range(0, min(lookback_h, 12), 2):
                ts = now - timedelta(hours=12 - h)
                out.append({
                    "id": f"demo_{sym}_{i}",
                    "ts": _iso(ts),
                    "symbol": sym,
                    "direction": "long" if (i % 2 == 0) else "short",
                    "confidence": 0.6 if (i % 2 == 0) else -0.6,
                    "price": None,
                })
                i += 1
        return sorted(out, key=lambda x: (x["symbol"], x["ts"]))
    return []


def _load_prices(models_dir: Path, symbols: List[str], lookback_h: int, is_demo: bool) -> Dict[str, List[Tuple[str, float]]]:
    mc = _read_json(models_dir / "market_context.json")
    prices = _extract_prices_from_market_context(mc, symbols) if mc else {}
    if not prices:
        prices = _synthesize_prices(symbols, hours=lookback_h if lookback_h > 0 else 72)
    # ensure sorted
    for s in list(prices.keys()):
        prices[s] = sorted(prices[s], key=lambda x: x[0])
    return prices


def _simulate_for_symbol(
    sym: str,
    signals: List[Dict],
    prices: List[Tuple[str, float]],
    cfg: _Cfg,
) -> Tuple[List[Dict], List[Tuple[str, float]]]:
    """
    Returns (trades, equity_points)
    - equity_points is [(ts_iso, equity_value)] with updates at each trade close
    """
    capital = cfg.capital
    equity = capital
    equity_points: List[Tuple[str, float]] = []
    trades: List[Dict] = []
    position: Optional[Dict] = None  # {side: 1|-1, entry_ts, entry_px}

    if not prices:
        return trades, equity_points

    # Index for quick "price at or after" lookup
    prices_sorted = prices

    # walk signals in time
    for sig in [s for s in signals if s["symbol"] == sym]:
        # ENTRY handling
        entry_ts_iso = sig["ts"]
        # using next available price at/after ts
        entry_hit = _align_price_at_or_after(prices_sorted, entry_ts_iso) or prices_sorted[-1]
        entry_ts, entry_px = entry_hit

        # If already in a position:
        if position:
            # reverse-signal => close then open reversed
            if position["side"] != _side_to_sign(sig["direction"]):
                # close existing
                exit_ts_iso = _time_add_iso(entry_ts, 0)  # close at the same bar
                exit_ts, exit_px = _align_price_at_or_after(prices_sorted, exit_ts_iso) or prices_sorted[-1]
                trades.append(_close_trade(sym, position, exit_ts, exit_px, cfg))
                equity += trades[-1]["pnl"]
                equity_points.append((exit_ts, equity))
                position = None

        # open if flat
        if not position:
            position = {"side": _side_to_sign(sig["direction"]), "entry_ts": entry_ts, "entry_px": float(entry_px)}

        # TIME-BASED EXIT for this signal (independent horizon)
        horizon_exit_ts = _time_add_iso(entry_ts, cfg.horizon_min)
        exit_hit = _align_price_at_or_after(prices_sorted, horizon_exit_ts) or prices_sorted[-1]
        # If still the same position, close at horizon (unless a later reverse closes earlier; here we keep it simple)
        if position:
            trades.append(_close_trade(sym, position, exit_hit[0], float(exit_hit[1]), cfg))
            equity += trades[-1]["pnl"]
            equity_points.append((exit_hit[0], equity))
            position = None

    # No open position carried
    return trades, equity_points


def _close_trade(sym: str, pos: Dict, exit_ts: str, exit_px: float, cfg: _Cfg) -> Dict:
    side = int(pos["side"])
    entry_px = float(pos["entry_px"])
    # slippage + fees (both entry & exit)
    fr = (cfg.slippage_bps + cfg.fees_bps) / 10000.0
    eff_entry = entry_px * (1 + fr if side > 0 else 1 - fr)
    eff_exit = exit_px * (1 - fr if side > 0 else 1 + fr)

    raw_ret = (eff_exit - eff_entry) / eff_entry
    pnl_pct = raw_ret * side
    pnl = cfg.capital * pnl_pct * 0.10  # use 10% notional per trade for realism without leverage

    return {
        "entry_ts": pos["entry_ts"],
        "exit_ts": exit_ts,
        "symbol": sym,
        "side": "long" if side > 0 else "short",
        "entry": round(entry_px, 6),
        "exit": round(exit_px, 6),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 6),
    }


def _plots(arts: Path, equity: List[Tuple[str, float]], rets: List[float]], per_sym: Dict[str, Dict]) -> None:
    if plt is None:
        return
    ensure_dir(arts)

    # Equity curve
    try:
        ts = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t, _ in equity]
        vals = [v for _, v in equity]
        fig = plt.figure(figsize=(7, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(ts, vals, lw=1.5)
        ax.set_title("Equity Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity")
        fig.tight_layout()
        fig.savefig(str(arts / "perf_equity_curve.png"))
        plt.close(fig)
    except Exception:
        pass

    # Drawdown
    try:
        dd_vals = []
        peak = None
        for _, v in equity:
            if peak is None or v > peak:
                peak = v
            dd_vals.append((v - peak) / peak if peak else 0.0)
        fig = plt.figure(figsize=(7, 2.5), dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(dd_vals)
        ax.set_title("Drawdown (ratio)")
        ax.set_xlabel("Trade closures")
        ax.set_ylabel("Drawdown")
        fig.tight_layout()
        fig.savefig(str(arts / "perf_drawdown.png"))
        plt.close(fig)
    except Exception:
        pass

    # Returns histogram
    try:
        fig = plt.figure(figsize=(6, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.hist(rets, bins=20)
        ax.set_title("Per-trade Returns (fraction)")
        fig.tight_layout()
        fig.savefig(str(arts / "perf_returns_hist.png"))
        plt.close(fig)
    except Exception:
        pass

    # By-symbol bar (Sharpe)
    try:
        labels = list(per_sym.keys())
        vals = [per_sym[s].get("sharpe") or 0.0 for s in labels]
        fig = plt.figure(figsize=(6, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.bar(labels, vals)
        ax.set_title("Sharpe by Symbol (per-trade unit)")
        fig.tight_layout()
        fig.savefig(str(arts / "perf_by_symbol_bar.png"))
        plt.close(fig)
    except Exception:
        pass


def run_paper_trader(ctx, mode: str = None) -> Dict[str, Any]:
    """
    Entry point used by CI orchestrator.
    Returns summary dict and writes artifacts:

      - logs/trades.jsonl
      - logs/equity_curve.jsonl
      - models/performance_metrics.json
      - artifacts/perf_*.png
      - (optional) artifacts/performance_report.html
    """
    # Resolve paths from ctx
    models_dir: Path = getattr(ctx, "models_dir", Path("models"))
    logs_dir: Path = getattr(ctx, "logs_dir", Path("logs"))
    arts_dir: Path = getattr(ctx, "artifacts_dir", Path("artifacts"))
    ensure_dir(models_dir); ensure_dir(logs_dir); ensure_dir(arts_dir)

    cfg = _load_cfg()
    if mode:
        cfg.mode = mode
    is_demo = bool(getattr(ctx, "is_demo", False))

    # Load
    signals = _load_signals(logs_dir, cfg.symbols, cfg.lookback_h, is_demo)
    prices_by_sym = _load_prices(models_dir, cfg.symbols, cfg.lookback_h, is_demo)

    # Simulate (per symbol)
    all_trades: List[Dict] = []
    equity_points: List[Tuple[str, float]] = []
    for sym in cfg.symbols:
        sym_sigs = [s for s in signals if s["symbol"] == sym]
        sym_prices = prices_by_sym.get(sym, [])
        t, e = _simulate_for_symbol(sym, sym_sigs, sym_prices, cfg)
        all_trades.extend(t)
        equity_points.extend(e)

    # Sort equity by ts
    equity_points = sorted(equity_points, key=lambda x: x[0])

    # Build returns (per trade close)
    returns_series = [t["pnl_pct"] for t in all_trades]

    # Persist logs
    _write_jsonl(logs_dir / "trades.jsonl", all_trades)
    _write_jsonl(logs_dir / "equity_curve.jsonl", [{"ts": ts, "equity": v} for ts, v in equity_points])

    # Metrics
    met = compute_metrics(equity_points, returns_series, all_trades, risk_free=cfg.risk_free, mode=cfg.mode)
    out = {
        "generated_at": _iso(_now_utc()),
        "mode": cfg.mode,
        "window_hours": cfg.lookback_h,
        "capital": cfg.capital,
        "by_symbol": met["by_symbol"],
        "aggregate": met["aggregate"],
        "demo": is_demo and not signals,  # demo if we had to synthesize signals
    }

    # Save metrics JSON
    (models_dir / "performance_metrics.json").write_text(json.dumps(out, indent=2))

    # Charts
    _plots(arts_dir, equity_points, returns_series, out["by_symbol"])

    # Lightweight HTML (optional)
    try:
        html = _render_html_report(out)
        (arts_dir / "performance_report.html").write_text(html)
    except Exception:
        pass

    return out


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x*100:.1f}%"


def _render_html_report(m: Dict[str, Any]) -> str:
    agg = m.get("aggregate", {})
    sym = m.get("by_symbol", {})
    def nv(k): 
        v = agg.get(k)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "n/a"
        if k in ("max_drawdown", "win_rate", "profit_factor", "exposure_pct", "avg_trade", "cagr"):
            if k in ("profit_factor",):
                return f"{v:.2f}"
            if k in ("avg_trade", "max_drawdown", "win_rate", "exposure_pct", "cagr"):
                return _fmt_pct(v)
        return f"{v:.2f}"

    rows = []
    for s, d in sym.items():
        rows.append(
            f"<tr><td>{s}</td>"
            f"<td>{d.get('trades', 0)}</td>"
            f"<td>{'n/a' if d.get('sharpe') is None else f'{d.get('sharpe'):.2f}'}</td>"
            f"<td>{'n/a' if d.get('sortino') is None else f'{d.get('sortino'):.2f}'}</td>"
            f"<td>{_fmt_pct(d.get('win_rate'))}</td>"
            f"<td>{'n/a' if d.get('profit_factor') in (None,0) else f'{d.get('profit_factor'):.2f}'}</td></tr>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>MoonWire Performance Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; padding:16px; }}
h1 {{ margin:0 0 8px; }}
.grid {{ display:grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); grid-gap: 12px; }}
.card {{ border:1px solid #ddd; border-radius:8px; padding:12px; }}
table {{ border-collapse: collapse; width:100%; }}
th, td {{ border-bottom:1px solid #eee; padding:6px 8px; text-align:left; font-size:14px; }}
small {{ color:#666; }}
img {{ width:100%; height:auto; border:1px solid #eee; border-radius:4px; }}
</style></head>
<body>
<h1>Signal Performance Report</h1>
<small>Generated: {m.get('generated_at')} • Mode: {m.get('mode')} • Window: {m.get('window_hours')}h • Capital: {m.get('capital')}</small>
<div class="grid" style="margin-top:12px">
  <div class="card">
    <h3>Aggregate</h3>
    <table>
      <tr><th>Trades</th><td>{agg.get('trades',0)}</td></tr>
      <tr><th>Sharpe</th><td>{nv('sharpe')}</td></tr>
      <tr><th>Sortino</th><td>{nv('sortino')}</td></tr>
      <tr><th>MaxDD</th><td>{nv('max_drawdown')}</td></tr>
      <tr><th>Win rate</th><td>{nv('win_rate')}</td></tr>
      <tr><th>Profit factor</th><td>{nv('profit_factor')}</td></tr>
      <tr><th>Avg trade</th><td>{nv('avg_trade')}</td></tr>
      <tr><th>Exposure</th><td>{nv('exposure_pct')}</td></tr>
      <tr><th>CAGR</th><td>{nv('cagr')}</td></tr>
    </table>
  </div>
  <div class="card">
    <h3>By Symbol</h3>
    <table>
      <tr><th>Symbol</th><th>Trades</th><th>Sharpe</th><th>Sortino</th><th>Win%</th><th>PF</th></tr>
      {''.join(rows)}
    </table>
  </div>
  <div class="card"><h3>Equity Curve</h3><img src="perf_equity_curve.png" alt="equity"></div>
  <div class="card"><h3>Drawdown</h3><img src="perf_drawdown.png" alt="drawdown"></div>
  <div class="card"><h3>Returns Histogram</h3><img src="perf_returns_hist.png" alt="returns"></div>
  <div class="card"><h3>Sharpe by Symbol</h3><img src="perf_by_symbol_bar.png" alt="by_symbol"></div>
</div>
</body></html>
"""
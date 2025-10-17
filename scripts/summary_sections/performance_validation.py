# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json, math, os, statistics, time
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple

# Minimal 1x1 PNG (fallback if matplotlib not available)
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

SECTION_TITLE = "Signal Performance Validation (v0.9.0)"

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _ensure_seed_trades(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    trades = logs_dir / "trades.jsonl"
    if (not trades.exists()) or trades.stat().st_size == 0:
        now = int(time.time())
        rows = [
            {"ts": now-5400, "symbol": "BTC", "side": "long",  "price": 60000, "qty": 0.01, "pnl": 12.0},
            {"ts": now-4200, "symbol": "ETH", "side": "long",  "price": 3000,  "qty": 0.10, "pnl": -3.0},
            {"ts": now-3000, "symbol": "SOL", "side": "short", "price": 156,   "qty": 1.00, "pnl":  1.5},
            {"ts": now-1800, "symbol": "BTC", "side": "short", "price": 60120, "qty": 0.01, "pnl": -4.0},
            {"ts": now- 600, "symbol": "ETH", "side": "long",  "price": 3010,  "qty": 0.10, "pnl":  2.0},
        ]
        with trades.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return trades

def _sharpe(returns: List[float]) -> float | None:
    if len(returns) < 2:
        return None
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns) or 0.0
    if sd == 0.0:
        return None
    # daily-ish scale factor doesn't matter in demo; keep 1.0
    return mu / sd

def _sortino(returns: List[float]) -> float | None:
    if len(returns) < 2:
        return None
    mu = statistics.mean(returns)
    downs = [min(r, 0.0) for r in returns]
    dd = statistics.pstdev(downs) or 0.0
    if dd == 0.0:
        return None
    return mu / abs(dd)

def _perf_from_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0}

    # Use PnL as “return” proxy (demo)
    rets = [float(t.get("pnl", 0.0)) for t in trades]
    cum = sum(rets)
    wins = [r for r in rets if r > 0]
    losses = [abs(r) for r in rets if r < 0]
    win_rate = (len(wins) / len(rets)) * 100.0
    pf = (sum(wins) / sum(losses)) if losses else (sum(wins) and math.inf or 0.0)

    eq = []
    cur = 0.0
    mdd = 0.0
    peak = 0.0
    for r in rets:
        cur += r
        peak = max(peak, cur)
        dd = cur - peak
        mdd = min(mdd, dd)
        eq.append(cur)

    return {
        "trades": len(rets),
        "sharpe": _sharpe(rets),
        "sortino": _sortino(rets),
        "max_drawdown": mdd,  # absolute demo units
        "win_rate": win_rate,
        "profit_factor": pf,
        "by_symbol": _by_symbol(trades),
        "equity_curve": eq[-100:],  # cap for plot
    }

def _by_symbol(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        sym = t.get("symbol", "UNK")
        out.setdefault(sym, {"rets": []})["rets"].append(float(t.get("pnl", 0.0)))
    for s, d in out.items():
        rets = d["rets"]
        d["sharpe"] = _sharpe(rets)
        wins = [r for r in rets if r > 0]
        d["win_rate"] = (len(wins) / len(rets)) * 100.0 if rets else None
        del d["rets"]
    return out

def _write_plot(equity: List[float], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(6, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(list(range(len(equity))), equity)
        ax.set_title("Demo Equity Curve")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL")
        fig.tight_layout()
        fig.savefig(str(out_png))
        plt.close(fig)
    except Exception:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(_PNG_1x1)

def append(md: List[str], ctx) -> None:
    """
    Render the compact demo performance section.
    - Reads one of: logs/trades.jsonl, logs/fills.jsonl, logs/executions.jsonl
    - If empty AND ctx.is_demo, seeds a tiny trades file and proceeds.
    - Writes:
        artifacts/performance_metrics.json
        artifacts/performance_curve.png
    """
    logs = Path("logs")
    candidates = [logs/"trades.jsonl", logs/"fills.jsonl", logs/"executions.jsonl"]
    rows: List[Dict[str, Any]] = []
    src_used = None
    for p in candidates:
        rows = _read_jsonl(p)
        if rows:
            src_used = p
            break

    # Auto-seed in DEMO mode if nothing found
    if not rows and getattr(ctx, "is_demo", False):
        src_used = _ensure_seed_trades(logs)
        rows = _read_jsonl(src_used)

    md.append(f"\n### 🚀 {SECTION_TITLE}")

    if not rows:
        md.append("⚠️ No backtestable trades found in the lookback window.")
        return

    perf = _perf_from_trades(rows)

    # Write artifacts
    arts = getattr(ctx, "artifacts_dir", Path("artifacts"))
    arts = Path(arts); arts.mkdir(parents=True, exist_ok=True)
    (arts / "performance_metrics.json").write_text(json.dumps(perf, indent=2))
    _write_plot(perf.get("equity_curve", []), arts / "performance_curve.png")

    # Build compact summary line
    sharpe = perf.get("sharpe")
    sortino = perf.get("sortino")
    wr = perf.get("win_rate")
    pf = perf.get("profit_factor")
    mdd = perf.get("max_drawdown")

    def _fmt(x, fmt="{:.2f}", na="n/a"):
        try:
            return fmt.format(x) if (x is not None and x == x) else na
        except Exception:
            return na

    bysym = perf.get("by_symbol", {})
    bysym_str = []
    for sym, d in bysym.items():
        bysym_str.append(f"{sym}(S={_fmt(d.get('sharpe'))}, WR={_fmt(d.get('win_rate'), '{:.1f}%')})")
    bysym_line = ", ".join(bysym_str)

    md.append(
        f"trades={perf['trades']} │ "
        f"Sharpe={_fmt(sharpe)} │ Sortino={_fmt(sortino)} │ "
        f"MaxDD={_fmt(mdd, '{:.1f}')} │ Win={_fmt(wr, '{:.1f}%')} │ PF={_fmt(pf)}\n"
        f"by symbol: {bysym_line}"
    )
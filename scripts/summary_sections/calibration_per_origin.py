# scripts/summary_sections/calibration_per_origin.py
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

###############################################################################
# Local fallbacks: this module will prefer ctx.common.* if available.
###############################################################################

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # tolerate malformed lines in CI demo mode
                continue
    return rows

def _coerce_ts_utc(x: Any) -> Optional[datetime]:
    """Accepts ISO8601 strings or epoch (seconds/ms) and returns aware UTC dt."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # heuristic: treat > 10^12 as ms
        if x > 1_000_000_000_000:
            x = x / 1000.0
        return datetime.fromtimestamp(float(x), tz=timezone.utc)
    if isinstance(x, str):
        s = x.strip()
        # Try int epoch
        if re.fullmatch(r"\d{10,13}", s):
            val = int(s)
            if val > 1_000_000_000_000:
                val = val / 1000.0
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        # Try ISO
        try:
            # tolerate Z or offset
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    try:
        if den == 0:
            return default
        return num / den
    except Exception:
        return default

def _linspace_0_1(n_bins: int) -> List[Tuple[float, float]]:
    step = 1.0 / n_bins
    return [(i * step, (i + 1) * step) for i in range(n_bins)]

def _compute_calibration_local(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Pure-Python ECE + Brier + fixed-edge reliability bins (length == n_bins).
    Bins cover [0,1), last bin includes 1.0.
    """
    assert len(y_true) == len(y_prob)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, [{"low": a, "high": b, "avg_conf": None, "emp_rate": None, "count": 0}
                          for a, b in _linspace_0_1(n_bins)]

    # Brier
    brier = sum((float(p) - float(y)) ** 2 for p, y in zip(y_prob, y_true)) / n

    # Bins
    edges = _linspace_0_1(n_bins)
    acc = [{"sum_p": 0.0, "sum_y": 0.0, "count": 0} for _ in range(n_bins)]
    for y, p in zip(y_true, y_prob):
        # clamp p into [0,1]
        p = max(0.0, min(1.0, float(p)))
        # map to bin
        idx = min(int(p * n_bins), n_bins - 1)  # [0..n_bins-1]
        acc[idx]["sum_p"] += p
        acc[idx]["sum_y"] += float(y)
        acc[idx]["count"] += 1

    bins = []
    ece = 0.0
    for (low, high), a in zip(edges, acc):
        c = a["count"]
        avg_conf = None
        emp_rate = None
        if c > 0:
            avg_conf = a["sum_p"] / c
            emp_rate = a["sum_y"] / c
            ece += (c / n) * abs(emp_rate - avg_conf)
        bins.append({
            "low": low, "high": high,
            "avg_conf": avg_conf, "emp_rate": emp_rate, "count": c
        })

    return float(ece), float(brier), bins

def _plot_reliability_local(bins: List[Dict[str, Any]], title: str, out_path: Path):
    # Lazy import to keep test env lean.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Prepare points, skipping empty bins to avoid NaNs.
    xs = []
    ys = []
    for b in bins:
        if b.get("count", 0) > 0 and b.get("avg_conf") is not None and b.get("emp_rate") is not None:
            xs.append(float(b["avg_conf"]))
            ys.append(float(b["emp_rate"]))

    _ensure_dir(out_path.parent)
    plt.figure()
    # Identity line
    plt.plot([0, 1], [0, 1])
    # Points
    if xs and ys:
        plt.scatter(xs, ys)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical outcome rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

###############################################################################
# Core helpers that prefer ctx.common.* if present, else use local fallback.
###############################################################################

def _get(ctx, name: str, default: Any):
    try:
        val = getattr(ctx.common, name)
        if callable(val) or not isinstance(val, property):
            return val
        return default
    except Exception:
        return default

def _read_jsonl_ctx(ctx, path: Path) -> List[dict]:
    fn = _get(ctx, "read_jsonl", _read_jsonl)
    return fn(path)

def _coerce_ts_utc_ctx(ctx, x: Any) -> Optional[datetime]:
    fn = _get(ctx, "coerce_ts_utc", _coerce_ts_utc)
    return fn(x)

def _ensure_dir_ctx(ctx, path: Path):
    fn = _get(ctx, "ensure_dir", _ensure_dir)
    return fn(path)

def _compute_calibration_ctx(ctx, y_true: List[int], y_prob: List[float], n_bins: int = 10):
    fn = _get(ctx, "compute_calibration", None)
    if fn is not None:
        try:
            ece, brier, bins = fn(y_true, y_prob, n_bins=n_bins)
            # Normalize expected keys to our names if needed
            norm_bins = []
            for b in bins:
                # Accept either {low, high, avg_conf, emp_rate, count} or
                # {bin_low, bin_high, ...}
                low = b.get("low", b.get("bin_low"))
                high = b.get("high", b.get("bin_high"))
                avg_conf = b.get("avg_conf")
                emp = b.get("emp_rate", b.get("empirical"))
                cnt = b.get("count")
                norm_bins.append({
                    "low": low, "high": high,
                    "avg_conf": avg_conf, "emp_rate": emp, "count": cnt
                })
            # If common returns only non-empty bins, we pad to n_bins (schema requires fixed length)
            if len(norm_bins) != n_bins:
                edges = _linspace_0_1(n_bins)
                by_edge = {(round(b["low"], 6), round(b["high"], 6)): b for b in norm_bins}
                padded = []
                for (lo, hi) in edges:
                    k = (round(lo, 6), round(hi, 6))
                    b = by_edge.get(k, {"low": lo, "high": hi, "avg_conf": None, "emp_rate": None, "count": 0})
                    padded.append(b)
                norm_bins = padded
            return float(ece), float(brier), norm_bins
        except Exception:
            pass
    return _compute_calibration_local(y_true, y_prob, n_bins=n_bins)

def _plot_reliability_ctx(ctx, bins: List[Dict[str, Any]], title: str, out_path: Path):
    fn = _get(ctx, "plot_reliability", None)
    if fn is not None:
        try:
            return fn(bins, title=title, out_path=out_path)
        except Exception:
            pass
    return _plot_reliability_local(bins, title, out_path)

###############################################################################
# Module logic
###############################################################################

def _norm_origin(name: str) -> str:
    s = (name or "unknown").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s or "unknown"

def _load_windowed_rows(ctx, hours: int) -> List[Dict[str, Any]]:
    logs_dir: Path = ctx.paths.LOGS_DIR
    hist = _read_jsonl_ctx(ctx, logs_dir / "trigger_history.jsonl")
    labs = _read_jsonl_ctx(ctx, logs_dir / "label_feedback.jsonl")

    # Latest label per id wins.
    label_by_id: Dict[str, int] = {}
    for r in labs:
        rid = r.get("id")
        if not rid:
            continue
        lab = r.get("label")
        # normalize to int 0/1
        lab_i = 1 if str(lab).lower() in ("1", "true", "t", "yes") or lab is True else 0
        label_by_id[rid] = lab_i

    now_utc: datetime = ctx.now_utc
    start = now_utc - timedelta(hours=hours)
    rows: List[Dict[str, Any]] = []
    for r in hist:
        rid = r.get("id")
        if not rid:
            continue
        ts = _coerce_ts_utc_ctx(ctx, r.get("ts"))
        if not ts or ts < start or ts > now_utc:
            continue
        if rid not in label_by_id:
            continue  # keep only labeled examples
        origin = r.get("origin") or "unknown"
        p = r.get("score")
        if p is None:
            continue
        try:
            p = float(p)
        except Exception:
            continue
        rows.append({"origin": origin, "y": label_by_id[rid], "p": p})
    return rows

def _seed_demo_rows() -> List[Dict[str, Any]]:
    import random
    random.seed(1337)
    rows: List[Dict[str, Any]] = []
    # 3 origins with varied behaviors
    cfg = [
        ("reddit", 100, 0.95, 0.02),   # well-calibrated-ish
        ("twitter", 60, 0.70, 0.00),   # overconfident
        ("news", 24, 0.90, 0.01)       # smaller n, decent
    ]
    for origin, n, slope, intercept in cfg:
        for _ in range(n):
            p = random.random() * 0.75 + 0.15  # 0.15..0.9
            # Empirical rate ~= slope * p + intercept, clipped
            pr = max(0.0, min(1.0, slope * p + intercept))
            y = 1 if random.random() < pr else 0
            rows.append({"origin": origin, "y": y, "p": p})
    return rows

def append(md, ctx):
    """
    Public API: appends a per-origin calibration report to the markdown list,
    and writes JSON + PNG artifacts.

    Side effects:
      - models/calibration_per_origin.json
      - artifacts/cal_reliability_<origin>.png
    """
    hours = int(os.getenv("MW_CAL_WINDOW_H", "72"))
    # 1) Gather rows for window or seed in demo
    rows = _load_windowed_rows(ctx, hours)
    demo_used = False
    if len(rows) < 30 and getattr(ctx, "demo", False):
        rows = _seed_demo_rows()
        demo_used = True

    # 2) Group by origin
    by_origin: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        o = r["origin"]
        d = by_origin.setdefault(o, {"y": [], "p": []})
        d["y"].append(int(r["y"]))
        d["p"].append(float(r["p"]))

    # 3) Compute per-origin metrics + plots
    results: List[Dict[str, Any]] = []
    n_bins = 10  # matches v0.6.3 default
    for origin, d in by_origin.items():
        n = len(d["y"])
        if n == 0:
            continue
        ece, brier, bins = _compute_calibration_ctx(ctx, d["y"], d["p"], n_bins=n_bins)
        low_n = n < 30
        high_ece = (ece is not None) and (ece > 0.06)
        safe_name = _norm_origin(origin)
        out_png = Path(ctx.paths.ARTIFACTS_DIR) / f"cal_reliability_{safe_name}.png"
        _plot_reliability_ctx(ctx, bins, title=f"Reliability: {origin}", out_path=out_png)

        results.append({
            "origin": origin,
            "n": n,
            "ece": round(float(ece), 3),
            "brier": round(float(brier), 3),
            "low_n": low_n,
            "high_ece": high_ece,
            "bins": [
                {
                    "bin_low": float(b["low"]),
                    "bin_high": float(b["high"]),
                    "avg_conf": (None if b["avg_conf"] is None else float(b["avg_conf"])),
                    "empirical": (None if b["emp_rate"] is None else float(b["emp_rate"])),
                    "count": int(b["count"]),
                }
                for b in bins
            ],
            "artifact_png": str(out_png).replace("\\", "/"),
        })

    # Sort: n desc, then origin asc
    results.sort(key=lambda r: (-int(r["n"]), str(r["origin"]).lower()))

    # 4) Write JSON artifact
    json_path = Path(ctx.paths.MODELS_DIR) / "calibration_per_origin.json"
    _ensure_dir_ctx(ctx, json_path.parent)
    payload = {
        "window_hours": hours,
        "generated_at": ctx.now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "demo": bool(demo_used),
        "origins": results,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 5) Append Markdown
    md.append(f"🧮 Per-Origin Calibration ({hours}h)")
    if not results:
        md.append(" • (no labeled samples in window)")
        return

    for r in results:
        flags = []
        if r["low_n"]:
            flags.append("low_n")
        if r["high_ece"]:
            flags.append("high_ece")
        suffix = f" [{ ' | '.join(flags) }]" if flags else ""
        md.append(
            f" • {r['origin']:<7} → ECE={r['ece']:.2f} | Brier={r['brier']:.2f} | n={r['n']}{suffix}"
        )

# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, json, math
from typing import Dict, List, Any, Tuple, DefaultDict
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt

from scripts.summary_sections.common import parse_ts, is_demo_mode, SummaryContext

# ------------------------------- helpers ------------------------------------

def _hr_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _safe_div(a: float, b: float) -> float | None:
    try:
        if b == 0:
            return None
        return a / b
    except Exception:
        return None

def _round(x: float | None, k: int = 2) -> float | None:
    if x is None:
        return None
    try:
        return round(float(x), k)
    except Exception:
        return None

def _class_from_f1(f1: float | None, n: int) -> Tuple[str, str]:
    if n < 2:
        return "Insufficient", "ℹ️"
    if f1 is None:
        return "Insufficient", "ℹ️"
    if f1 >= 0.75:
        return "Strong", "✅"
    if f1 >= 0.40:
        return "Mixed", "⚠️"
    return "Weak", "❌"

def _get_models_dir(ctx: SummaryContext) -> Path:
    return ctx.models_dir

def _get_artifacts_dir(ctx: SummaryContext) -> Path:
    """
    Tests create tmp_path/artifacts and expect us to write there.
    Use ctx.logs_dir.parent/artifacts when available; fall back to ./artifacts.
    """
    if ctx and ctx.logs_dir:
        d = ctx.logs_dir.parent / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = Path("artifacts")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _persist_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# --------------------------- chart rendering ---------------------------------

def _render_version_trend_chart(series: List[Dict[str, Any]],
                                hours: int,
                                out_path: Path,
                                demo: bool = False) -> bool:
    """
    series: list of {version, t(ISO), precision(float)}
    """
    if not series:
        return False

    # Group by version
    by_ver: DefaultDict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    for row in series:
        v = row.get("version") or "unknown"
        t = parse_ts(row.get("t"))
        p = row.get("precision")
        if t is None or p is None:
            continue
        by_ver[v].append((t, float(p)))

    if not by_ver:
        return False

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)

    # Shaded quality bands
    ax.axhspan(0.0, 0.40, alpha=0.10, color="red")
    ax.axhspan(0.40, 0.75, alpha=0.10, color="gold")
    ax.axhspan(0.75, 1.00, alpha=0.10, color="green")

    # Plot one line per version
    for ver, pts in sorted(by_ver.items()):
        pts = sorted(pts, key=lambda x: x[0])
        xs = [t for (t, _) in pts]
        ys = [p for (_, p) in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=ver)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Time")
    ttl = f"Per-Version Signal Quality ({hours}h)"
    if demo:
        ttl += " (demo)"
    ax.set_title(ttl)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True

# ------------------------------- main API ------------------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Renders per-version snapshot lines (already in this section previously),
    and now appends a precision trend chart across runs from
    models/signal_quality_per_version.json -> artifacts/signal_quality_by_version_{H}h.png
    """
    models_dir = _get_models_dir(ctx)
    art_dir = _get_artifacts_dir(ctx)
    hours = _hr_env("MW_SIGNAL_WINDOW_H", 72)
    join_min = _hr_env("MW_SIGNAL_JOIN_MIN", 5)
    chart_on = os.getenv("MW_SIGNAL_VERSION_CHART", "true").lower() in ("1", "true", "yes")

    path = models_dir / "signal_quality_per_version.json"
    data = _load_json(path) or {}

    per_version: List[Dict[str, Any]] = data.get("per_version") or []
    series: List[Dict[str, Any]] = data.get("series") or []
    is_demo = bool(data.get("demo"))

    # --- Header & snapshot block ---
    header = f"### 🧪 Per-Version Signal Quality ({hours}h)"
    if is_demo:
        header += " (demo)"
    md.append(header)

    if not per_version and not series and ctx.is_demo:
        # Seed a tiny demo summary if absolutely empty
        per_version = [
            {"version": "v0.5.9", "precision": 0.70, "recall": 0.64, "f1": 0.67, "labels": 10, "class": "Mixed", "emoji": "⚠️"},
            {"version": "v0.5.8", "precision": 0.80, "recall": 0.75, "f1": 0.77, "labels": 12, "class": "Strong", "emoji": "✅"},
        ]
        series = [
            {"version": "v0.5.9", "t": datetime.now(timezone.utc).isoformat(), "precision": 0.70},
            {"version": "v0.5.8", "t": datetime.now(timezone.utc).isoformat(), "precision": 0.80},
        ]
        is_demo = True
        data = {
            "window_hours": hours,
            "join_minutes": join_min,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "per_version": per_version,
            "series": series,
            "demo": True,
        }
        _persist_json(path, data)

    # Snapshot lines
    if per_version:
        # Sort: Strong → Mixed → Weak → Insufficient, then by version
        order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
        per_version_sorted = sorted(per_version, key=lambda r: (order.get(r.get("class", "Mixed"), 1), r.get("version", "")))
        for row in per_version_sorted:
            ver = row.get("version", "unknown")
            f1 = _round(row.get("f1"), 2)
            p  = _round(row.get("precision"), 2)
            r  = _round(row.get("recall"), 2)
            n  = row.get("labels") or row.get("triggers") or 0
            cls = row.get("class", "Mixed")
            emoji = row.get("emoji", "⚠️")
            md.append(f"- `{ver}` → {emoji} {cls} (F1={f1}, P={p}, R={r}, n={n})")
    else:
        md.append("_no per-version summary available_")

    # --- Trend chart ---
    if chart_on:
        img_name = f"signal_quality_by_version_{hours}h.png"
        img_path = art_dir / img_name
        made = _render_version_trend_chart(series, hours, img_path, demo=is_demo)
        if made:
            # Use relative path in markdown so it shows in CI summary
            md.append(f"\n📈 Per-Version Precision Trend: ![](artifacts/{img_name})")
        else:
            md.append("\n📈 Per-Version Precision Trend: _no time series available_ (enable DEMO_MODE or accumulate runs)")
    else:
        md.append("\n📈 Per-Version Precision Trend: _disabled via MW_SIGNAL_VERSION_CHART=false_")
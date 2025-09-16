# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json
from typing import Dict, List, Any, Tuple, DefaultDict
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt

from scripts.summary_sections.common import parse_ts, is_demo_mode, SummaryContext

# ------------------------------- helpers ------------------------------------

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _bool_env(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "on")

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

def _class_from_f1(f1: float | None, n_labels: int) -> Tuple[str, str]:
    if n_labels < 2 or f1 is None:
        return "Insufficient", "ℹ️"
    if f1 >= 0.75:
        return "Strong", "✅"
    if f1 >= 0.40:
        return "Mixed", "⚠️"
    return "Weak", "❌"

def _get_models_dir(ctx: SummaryContext) -> Path:
    return ctx.models_dir

def _get_artifacts_dir(ctx: SummaryContext) -> Path:
    # Tests create tmp_path/artifacts and expect us to write there.
    if ctx and ctx.logs_dir:
        d = ctx.logs_dir.parent / "artifacts"
    else:
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

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

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

# ------------------------- compute from raw logs ------------------------------

def _within_window(ts: datetime | None, start: datetime) -> bool:
    return ts is not None and ts >= start

def _nearest_trigger(label_ts: datetime,
                     trig_rows: List[Dict[str, Any]],
                     join_delta: timedelta,
                     origin: str) -> Dict[str, Any] | None:
    """
    Brute-force nearest-in-time trigger on same origin within join_delta.
    Good enough for small CI/test logs.
    """
    best = None
    best_dt = None
    for tr in trig_rows:
        if tr.get("origin") != origin:
            continue
        tts = parse_ts(tr.get("timestamp"))
        if tts is None:
            continue
        dt = abs((tts - label_ts).total_seconds())
        if dt <= join_delta.total_seconds():
            if best is None or dt < best_dt:
                best = tr
                best_dt = dt
    return best

def _compute_per_version_from_logs(models_dir: Path,
                                   hours: int,
                                   join_min: int) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Build per_version snapshot from trigger_history.jsonl + label_feedback.jsonl.
    Returns (per_version_list, demo_flag).
    """
    trig = _load_jsonl(models_dir / "trigger_history.jsonl")
    labs = _load_jsonl(models_dir / "label_feedback.jsonl")

    if not trig and not labs:
        # nothing to compute; caller may seed demo
        return [], False

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    trig_w = [r for r in trig if _within_window(parse_ts(r.get("timestamp")), start)]
    labs_w = [r for r in labs if _within_window(parse_ts(r.get("timestamp")), start)]

    if not trig_w and not labs_w:
        return [], False

    join_delta = timedelta(minutes=join_min)

    # Aggregate per version
    agg: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: {"true": 0, "false": 0, "fn": 0})

    for lab in labs_w:
        lts = parse_ts(lab.get("timestamp"))
        if lts is None:
            continue
        origin = lab.get("origin") or "unknown"
        matched_tr = _nearest_trigger(lts, trig_w, join_delta, origin)

        # Determine version attribution: prefer label.model_version, else trigger.model_version, else unknown
        ver = lab.get("model_version") or (matched_tr or {}).get("model_version") or "unknown"

        label_val = lab.get("label")
        if matched_tr is None:
            # unmatched true → count as FN for that version; ignore unmatched false
            if label_val is True:
                agg[ver]["fn"] += 1
            continue

        # joined event → count true/false
        if label_val is True:
            agg[ver]["true"] += 1
        else:
            agg[ver]["false"] += 1

    # Convert to per_version list with metrics
    out: List[Dict[str, Any]] = []
    for ver, c in agg.items():
        T = int(c["true"])
        F = int(c["false"])
        FN = int(c["fn"])
        n_labels = T + F  # joined labels only
        prec = _safe_div(T, T + F) or 0.0 if (T + F) > 0 else None
        rec  = _safe_div(T, T + FN) if (T + FN) > 0 else None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = None
        cls, emoji = _class_from_f1(f1, n_labels)
        out.append({
            "version": ver,
            "triggers": n_labels,   # joined count; we don't track TN here
            "true": T,
            "false": F,
            "labels": n_labels,
            "precision": _round(prec, 2) if prec is not None else None,
            "recall": _round(rec, 2) if rec is not None else None,
            "f1": _round(f1, 2) if f1 is not None else None,
            "class": cls,
            "emoji": emoji,
            "demo": False,
        })

    # Sort for stable output
    order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
    out.sort(key=lambda r: (order.get(r.get("class", "Mixed"), 1), r.get("version", "")))

    return out, False

# ------------------------------- main API ------------------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Renders per-version snapshot lines and a precision trend chart.
    - Prefers existing models/signal_quality_per_version.json
    - If missing/empty, computes snapshot from raw logs and persists JSON (series empty)
    """
    models_dir = _get_models_dir(ctx)
    art_dir = _get_artifacts_dir(ctx)
    hours = _int_env("MW_SIGNAL_WINDOW_H", 72)
    join_min = _int_env("MW_SIGNAL_JOIN_MIN", 5)
    chart_on = _bool_env("MW_SIGNAL_VERSION_CHART", True)

    json_path = models_dir / "signal_quality_per_version.json"
    data = _load_json(json_path) or {}

    per_version: List[Dict[str, Any]] = data.get("per_version") or []
    series: List[Dict[str, Any]] = data.get("series") or []
    is_demo = bool(data.get("demo"))

    # Fallback: compute snapshot from raw logs if nothing present
    if not per_version:
        computed, demo_flag = _compute_per_version_from_logs(models_dir, hours, join_min)
        if computed:
            per_version = computed
            is_demo = False
            # Persist a minimal artifact so future runs (or chart tests) can find it
            data = {
                "window_hours": hours,
                "join_minutes": join_min,
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "per_version": per_version,
                "series": series,  # none yet
                "demo": False,
            }
            _persist_json(json_path, data)
        elif ctx.is_demo:
            # Seed minimal demo so CI summary shows something
            per_version = [
                {"version": "v0.5.9", "precision": 0.70, "recall": 0.64, "f1": 0.67, "labels": 10, "class": "Mixed", "emoji": "⚠️", "demo": True},
                {"version": "v0.5.8", "precision": 0.80, "recall": 0.75, "f1": 0.77, "labels": 12, "class": "Strong", "emoji": "✅", "demo": True},
            ]
            is_demo = True
            data = {
                "window_hours": hours,
                "join_minutes": join_min,
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "per_version": per_version,
                "series": [],
                "demo": True,
            }
            _persist_json(json_path, data)

    # --- Header & snapshot block ---
    header = f"### 🧪 Per-Version Signal Quality ({hours}h)"
    if is_demo:
        header += " (demo)"
    md.append(header)

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

    # --- Trend chart (if we have a time series in the JSON or demo-seeded) ---
    img_name = f"signal_quality_by_version_{hours}h.png"
    img_path = art_dir / img_name
    if chart_on:
        made = _render_version_trend_chart(series, hours, img_path, demo=is_demo)
        if made:
            md.append(f"\n📈 Per-Version Precision Trend: ![](artifacts/{img_name})")
        else:
            md.append("\n📈 Per-Version Precision Trend: _no time series available_ (enable DEMO_MODE or accumulate runs)")
    else:
        md.append("\n📈 Per-Version Precision Trend: _disabled via MW_SIGNAL_VERSION_CHART=false_")
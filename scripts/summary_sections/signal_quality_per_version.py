# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .common import SummaryContext, parse_ts, is_demo_mode


# ---------------------------
# Small helpers
# ---------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)

def _f1(p: float, r: float) -> float:
    if p <= 0.0 or r <= 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)

def _class_from_f1(f1: float, n_labels: int) -> Tuple[str, str]:  # (label, emoji)
    if n_labels < 2:
        return ("Insufficient", "ℹ️")
    if f1 >= 0.75:
        return ("Strong", "✅")
    if f1 >= 0.40:
        return ("Mixed", "⚠️")
    return ("Weak", "❌")

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    return rows

def _nearest_trigger_index(trig_times: List[datetime], t: datetime, max_delta: timedelta) -> int | None:
    """
    Given sorted trig_times, find index of the nearest time to t within max_delta.
    Binary search would be nicer; linear is fine for our sizes.
    """
    best_i, best_dt = None, None
    for i, tt in enumerate(trig_times):
        dt = abs((tt - t).total_seconds())
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_i = i
    if best_i is None:
        return None
    if abs((trig_times[best_i] - t)) <= max_delta:
        return best_i
    return None


# ---------------------------
# Chart rendering
# ---------------------------

def _render_version_precision_trend(md: List[str], ctx: SummaryContext, artifact_path: Path):
    """
    Reads models/signal_quality_per_version.json and renders a per-version precision trend
    to artifacts/signal_quality_by_version_<window>h.png. If no series is present, falls
    back to a single point per version using the current snapshot.
    """
    if os.getenv("MW_SIGNAL_VERSION_CHART", "true").lower() in ("0", "false", "no"):
        md.append("_per-version trend chart disabled via MW_SIGNAL_VERSION_CHART_")
        return

    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        md.append("_per-version trend chart skipped (artifact missing)_")
        return

    window = int(data.get("window_hours") or os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    is_demo = bool(data.get("demo"))

    series = data.get("series") or []
    # Fallback: synthesize a single point per version from this snapshot
    if not series:
        gen_ts = parse_ts(data.get("generated_at")) or datetime.now(timezone.utc)
        for row in data.get("per_version", []):
            series.append({
                "version": row.get("version", "unknown"),
                "t": gen_ts.isoformat(),
                "precision": row.get("precision", 0.0),
                "n": row.get("labels", row.get("triggers", 0)),
                "class": row.get("class", None),
            })

    # Group by version
    by_ver: Dict[str, List[Tuple[datetime, float]]] = {}
    for s in series:
        ver = s.get("version") or "unknown"
        tkey = s.get("t") or s.get("time") or s.get("timestamp") or s.get("end") or s.get("start")
        t = parse_ts(tkey)
        p = s.get("precision", None)
        if t is None or p is None:
            continue
        by_ver.setdefault(ver, []).append((t, float(p)))

    if all(len(v) == 0 for v in by_ver.values()):
        md.append("_per-version trend chart: no points to plot_")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Shaded quality bands (Weak/Mixed/Strong)
    ax.axhspan(0.00, 0.40, alpha=0.08, color="red")
    ax.axhspan(0.40, 0.75, alpha=0.08, color="yellow")
    ax.axhspan(0.75, 1.00, alpha=0.08, color="green")

    for ver, pts in sorted(by_ver.items()):
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        xs = [t for (t, _) in pts]
        ys = [p for (_, p) in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=ver)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Precision")

    title = f"Per-Version Signal Quality ({window}h)"
    if is_demo:
        title += " (demo)"
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out = Path("artifacts") / f"signal_quality_by_version_{window}h.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    md.append(f"📈 Per-Version Precision Trend: ![]({out.as_posix()})")


# ---------------------------
# Main section
# ---------------------------

def append(md: List[str], ctx: SummaryContext):
    """
    Build per-version signal quality over the last N hours, persist snapshot to
    models/signal_quality_per_version.json (while merging an accumulating `series`
    time history), and render a trend chart into artifacts/.
    """
    # Config knobs
    window_h = _env_int("MW_SIGNAL_WINDOW_H", 72)
    join_min = _env_int("MW_SIGNAL_JOIN_MIN", 5)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_h)
    join_delta = timedelta(minutes=join_min)

    # Load logs (with cross-run cache to avoid re-reads)
    trig_rows = ctx.caches.get("trigger_rows")
    if trig_rows is None:
        trig_rows = _load_jsonl(ctx.models_dir / "trigger_history.jsonl")
        ctx.caches["trigger_rows"] = trig_rows

    lab_rows = ctx.caches.get("label_rows")
    if lab_rows is None:
        lab_rows = _load_jsonl(ctx.models_dir / "label_feedback.jsonl")
        ctx.caches["label_rows"] = lab_rows

    # Filter window
    f_trig: List[Dict[str, Any]] = []
    for r in trig_rows:
        t = parse_ts(r.get("timestamp"))
        if t and t >= cutoff:
            f_trig.append(r)

    f_lab: List[Dict[str, Any]] = []
    for r in lab_rows:
        t = parse_ts(r.get("timestamp"))
        if t and t >= cutoff:
            f_lab.append(r)

    # Count triggers per version for recall denominator
    # "Decision" filter: prefer decision=='triggered' if present; otherwise count all.
    trig_by_origin: Dict[str, List[Dict[str, Any]]] = {}
    trig_times_by_origin: Dict[str, List[datetime]] = {}
    trig_ver_counts: Dict[str, int] = {}
    for r in f_trig:
        if "decision" in r and r.get("decision") != "triggered":
            continue
        origin = r.get("origin", "unknown")
        t = parse_ts(r.get("timestamp"))
        if not t:
            continue
        ver = (r.get("model_version") or "unknown")
        trig_ver_counts[ver] = trig_ver_counts.get(ver, 0) + 1

        trig_by_origin.setdefault(origin, []).append(r)

    # Sort and cache times per origin
    for origin, rows in trig_by_origin.items():
        rows.sort(key=lambda x: parse_ts(x.get("timestamp")) or datetime.fromtimestamp(0, tz=timezone.utc))
        trig_times_by_origin[origin] = [parse_ts(x.get("timestamp")) for x in rows]  # type: ignore

    # Join labels → nearest trigger within ±join_min on SAME origin.
    # Version attribution: prefer label.model_version; else matched trigger.model_version; else "unknown".
    agg: Dict[str, Dict[str, Any]] = {}  # version → counters
    for lab in f_lab:
        origin = lab.get("origin", "unknown")
        lt = parse_ts(lab.get("timestamp"))
        if lt is None:
            continue
        label_val = bool(lab.get("label"))

        rows = trig_by_origin.get(origin) or []
        times = trig_times_by_origin.get(origin) or []
        if not rows or not times:
            # No trigger to join; skip this label for counting
            continue

        idx = _nearest_trigger_index(times, lt, join_delta)
        if idx is None:
            continue

        trow = rows[idx]
        ver = (lab.get("model_version")
               or trow.get("model_version")
               or "unknown")

        st = agg.setdefault(ver, {"true": 0, "false": 0})
        if label_val:
            st["true"] += 1
        else:
            st["false"] += 1

    # Build per-version metrics
    per_version: List[Dict[str, Any]] = []
    for ver in sorted(set(list(agg.keys()) + list(trig_ver_counts.keys()))):
        counts = agg.get(ver, {"true": 0, "false": 0})
        tp = int(counts["true"])
        fp = int(counts["false"])
        labels_n = tp + fp
        triggers_n = int(trig_ver_counts.get(ver, 0))

        # Precision = TP / (TP + FP) over JOINED labels
        precision = _safe_div(tp, labels_n)
        # Recall approx = TP / (#triggered) over window (proxy)
        recall = _safe_div(tp, triggers_n)
        f1 = _f1(precision, recall)
        klass, emoji = _class_from_f1(f1, labels_n)

        per_version.append({
            "version": ver,
            "triggers": triggers_n,
            "true": tp,
            "false": fp,
            "labels": labels_n,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "class": klass,
            "emoji": emoji,
        })

    # DEMO fallback if empty
    demo_flag = False
    if not per_version and is_demo_mode():
        demo_flag = True
        per_version = [
            {"version": "v0.5.9", "triggers": 10, "true": 7, "false": 3, "labels": 10,
             "precision": 0.70, "recall": 0.64, "f1": 0.67, "class": "Mixed", "emoji": "⚠️"},
            {"version": "v0.5.8", "triggers": 8,  "true": 6, "false": 2, "labels": 8,
             "precision": 0.75, "recall": 0.75, "f1": 0.75, "class": "Strong", "emoji": "✅"},
            {"version": "v0.5.7", "triggers": 6,  "true": 2, "false": 4, "labels": 6,
             "precision": 0.33, "recall": 0.50, "f1": 0.40, "class": "Mixed", "emoji": "⚠️"},
        ]

    # Persist artifact (merge/append time series)
    art_path = ctx.models_dir / "signal_quality_per_version.json"

    # Try to merge existing "series" history
    existing_series: List[Dict[str, Any]] = []
    existing_meta: Dict[str, Any] = {}
    if art_path.exists():
        try:
            prev = json.loads(art_path.read_text(encoding="utf-8"))
            existing_series = list(prev.get("series") or [])
            # carry forward anything useful if desired
            existing_meta = {k: v for k, v in prev.items() if k not in ("per_version", "series")}
        except Exception:
            existing_series = []

    # Append this run's points into series with a timestamp
    gen_at = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    for row in per_version:
        existing_series.append({
            "version": row.get("version", "unknown"),
            "t": gen_at,
            "precision": row.get("precision", 0.0),
            "n": row.get("labels", 0),
            "class": row.get("class", None),
        })

    out_payload = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "generated_at": gen_at,
        "per_version": per_version,
        "series": existing_series,
        "demo": demo_flag,
    }
    # retain any prior meta keys (non-conflicting)
    out_payload.update({k: v for k, v in existing_meta.items() if k not in out_payload})

    art_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    # Render markdown
    title = f"### 🧪 Per-Version Signal Quality ({window_h}h)"
    if demo_flag:
        title += " (demo)"
    md.append(title)

    # Sort display: Strong → Mixed → Weak → Insufficient
    order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
    def _key(row):
        return order.get(row["class"], 9), row["version"]

    for row in sorted(per_version, key=_key):
        ver = row["version"]
        emoji = row["emoji"]
        md.append(f"- `{ver}` → {emoji} {row['class']} "
                  f"(F1={row['f1']:.2f}, P={row['precision']:.2f}, R={row['recall']:.2f}, "
                  f"n={row['labels']}, thresholdless, triggers={row['triggers']})")

    # Render & link the per-version precision trend chart
    try:
        _render_version_precision_trend(md, ctx, art_path)
    except Exception as _e:
        md.append(f"_per-version trend chart skipped: {type(_e).__name__}_")
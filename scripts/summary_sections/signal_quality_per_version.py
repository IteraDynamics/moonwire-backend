# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .common import SummaryContext, parse_ts, _iso, _load_jsonl, ensure_dir, _safe_div


# --------------------------------------------------------------------------
# Classification helper
# --------------------------------------------------------------------------
def _class_from_f1(f1: float, n_labels: int) -> Tuple[str, str]:
    if n_labels < 2:
        return ("Insufficient", "ℹ️")
    if f1 >= 0.75:
        return ("Strong", "✅")
    if f1 >= 0.40:
        return ("Mixed", "⚠️")
    return ("Weak", "❌")


# --------------------------------------------------------------------------
# Nearest join (label ↔ trigger) within ±join_min minutes, same origin
# --------------------------------------------------------------------------
def _nearest_join(
    labels: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
    join_min: int,
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    by_origin_trig: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in triggers:
        o = (t.get("origin") or "").strip().lower()
        if not o:
            continue
        ts = parse_ts(t.get("timestamp"))
        if ts:
            t["_ts"] = ts
            by_origin_trig[o].append(t)
    for lst in by_origin_trig.values():
        lst.sort(key=lambda r: r["_ts"])

    out: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    max_delta = timedelta(minutes=join_min)

    for lab in labels:
        o = (lab.get("origin") or "").strip().lower()
        if not o:
            continue
        ts = parse_ts(lab.get("timestamp"))
        if not ts:
            continue
        lab["_ts"] = ts

        candidates = by_origin_trig.get(o, [])
        if not candidates:
            out.append((lab, None))
            continue

        # binary search-ish nearest
        lo, hi = 0, len(candidates) - 1
        best_i, best_dt = None, timedelta.max
        while lo <= hi:
            mid = (lo + hi) // 2
            dt = abs(candidates[mid]["_ts"] - ts)
            if dt < best_dt:
                best_dt, best_i = dt, mid
            if candidates[mid]["_ts"] < ts:
                lo = mid + 1
            else:
                hi = mid - 1

        trig_match = candidates[best_i] if (best_i is not None and best_dt <= max_delta) else None
        out.append((lab, trig_match))

    return out


# --------------------------------------------------------------------------
# Demo series seeding (for chart) when data is sparse
# --------------------------------------------------------------------------
def _maybe_seed_series_if_demo(
    series: List[Dict[str, Any]] | None,
    per_version: List[Dict[str, Any]],
    now: datetime,
    is_demo: bool,
) -> List[Dict[str, Any]]:
    series = series or []
    if series or not is_demo or not per_version:
        return series

    # Build a tiny 3-point trend for up to 2 versions so CI has something to draw.
    kept = per_version[:2]
    seeded: List[Dict[str, Any]] = []
    for i, pv in enumerate(kept):
        v = pv.get("version", f"v{i+1}")
        base = float(pv.get("precision", 0.6))
        pts = [
            {"version": v, "t": _iso(now - timedelta(hours=6)), "precision": max(0.0, min(1.0, base - 0.05))},
            {"version": v, "t": _iso(now - timedelta(hours=3)), "precision": max(0.0, min(1.0, base - 0.01))},
            {"version": v, "t": _iso(now),                     "precision": max(0.0, min(1.0, base))},
        ]
        seeded.extend(pts)
    return seeded


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def _plot_version_trend(series: List[Dict[str, Any]], window_h: int, demo: bool) -> Path | None:
    if not series:
        return None

    # Group by version
    by_v: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    for row in series:
        v = str(row.get("version", "unknown"))
        ts = parse_ts(row.get("t"))
        pr = row.get("precision")
        if ts is None or pr is None:
            continue
        by_v[v].append((ts, float(pr)))

    for lst in by_v.values():
        lst.sort(key=lambda x: x[0])

    if not by_v:
        return None

    ensure_dir(Path("artifacts"))
    out_path = Path("artifacts") / f"signal_quality_by_version_{window_h}h.png"

    # Draw
    plt.figure(figsize=(8, 4.5))
    # Shaded zones for class bands
    plt.axhspan(0.75, 1.0, alpha=0.08)  # strong
    plt.axhspan(0.40, 0.75, alpha=0.06)  # mixed
    plt.axhspan(0.00, 0.40, alpha=0.05)  # weak

    for v, pts in by_v.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=v)

    title = f"Per-Version Precision Trend ({window_h}h)" + (" — demo" if demo else "")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("precision")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

    return out_path


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
def append(md: List[str], ctx: SummaryContext) -> None:
    models_dir = ctx.models_dir
    window_h = int(os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_SIGNAL_JOIN_MIN", "5"))
    want_chart = os.getenv("MW_SIGNAL_VERSION_CHART", "true").lower() in ("1", "true", "yes")

    out_json = models_dir / "signal_quality_per_version.json"
    now = datetime.now(timezone.utc)

    # Try to load an existing snapshot; if not present, compute from raw logs
    data: Dict[str, Any] = {}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    per_version = data.get("per_version")
    series = data.get("series")

    if not isinstance(per_version, list):
        # Compute a snapshot from raw logs
        trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
        lab_rows  = _load_jsonl(models_dir / "label_feedback.jsonl")

        cutoff = now - timedelta(hours=window_h)
        trig_rows = [r for r in trig_rows if (parse_ts(r.get("timestamp")) or now) >= cutoff]
        lab_rows  = [r for r in lab_rows  if (parse_ts(r.get("timestamp")) or now) >= cutoff]

        joined = _nearest_join(lab_rows, trig_rows, join_min)

        # per-version counting
        counts = defaultdict(lambda: {"true": 0, "false": 0, "labels": 0, "triggers": 0})
        for lab, trig in joined:
            # prefer label.model_version > trigger.model_version > "unknown"
            version = lab.get("model_version") or (trig or {}).get("model_version") or "unknown"
            if trig is not None:
                counts[version]["triggers"] += 1
            if lab.get("label") is True:
                counts[version]["true"] += 1
                counts[version]["labels"] += 1
            elif lab.get("label") is False:
                counts[version]["false"] += 1
                counts[version]["labels"] += 1

        per_version = []
        for v, c in counts.items():
            tp = int(c["true"])
            fp = int(c["false"])
            fn = 0  # without unmatched positives we keep fn=0 for consistency here
            P = _safe_div(tp, tp + fp)
            R = _safe_div(tp, tp + fn)
            F1 = _safe_div(2 * P * R, (P + R)) if (P + R) else 0.0
            klass, emoji = _class_from_f1(F1, int(c["labels"]))
            per_version.append({
                "version": v,
                "triggers": int(c["triggers"]),
                "true": tp,
                "false": fp,
                "labels": int(c["labels"]),
                "precision": round(P, 2),
                "recall": round(R, 2),
                "f1": round(F1, 2),
                "class": klass,
                "emoji": emoji,
                "demo": False,
            })

        # Order: Strong → Mixed → Weak → Insufficient
        order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
        per_version.sort(key=lambda r: (order.get(r["class"], 9), -r["f1"], r["version"]))

        # Demo fallback if totally empty
        if not per_version and ctx.is_demo:
            per_version = [
                {"version": "v0.5.9", "triggers": 8,  "true": 6, "false": 2, "labels": 8,
                 "precision": 0.75, "recall": 0.75, "f1": 0.75, "class": "Strong", "emoji": "✅", "demo": True},
                {"version": "v0.5.8", "triggers": 10, "true": 7, "false": 3, "labels": 10,
                 "precision": 0.70, "recall": 0.64, "f1": 0.67, "class": "Mixed",  "emoji": "⚠️", "demo": True},
            ]

        data = {
            "window_hours": window_h,
            "join_minutes": join_min,
            "generated_at": _iso(now),
            "per_version": per_version,
            "series": [],
            "demo": ctx.is_demo,
        }

    # Ensure each per_version row has emoji/class (in case a prior artifact was minimal)
    for r in data.get("per_version", []):
        if "emoji" not in r or "class" not in r:
            f1 = float(r.get("f1", 0.0))
            labels = int(r.get("labels", 0))
            klass, emoji = _class_from_f1(f1, labels)
            r.setdefault("class", klass)
            r.setdefault("emoji", emoji)

    # Seed series in demo if empty so the chart shows
    if not isinstance(data.get("series"), list):
        data["series"] = []
    data["series"] = _maybe_seed_series_if_demo(data["series"], data.get("per_version") or [], now, data.get("demo", False))

    # Persist snapshot (so the chart test can read the same)
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # -------- markdown --------
    md.append(f"### 🧪 Per-Version Signal Quality ({window_h}h){' (demo)' if data.get('demo') else ''}")
    pv = data.get("per_version") or []
    if pv:
        for r in pv:
            md.append(
                f"- `{r['version']}` → {r['emoji']} {r['class']} "
                f"(F1={float(r['f1']):.2f}, P={float(r['precision']):.2f}, R={float(r['recall']):.2f}, n={int(r['labels'])})"
            )
    else:
        md.append("_no per-version summary available_")

    # -------- chart --------
    img_written = False
    rel = f"artifacts/signal_quality_by_version_{window_h}h.png"
    if want_chart:
        img_path = _plot_version_trend(data.get("series", []), window_h, data.get("demo", False))
        img_written = bool(img_path and img_path.exists())

    if img_written:
        md.append(f"\n📈 Per-Version Precision Trend: {rel}")
    else:
        md.append("\n📈 Per-Version Precision Trend: _no time series available_ (enable DEMO_MODE or accumulate runs)")
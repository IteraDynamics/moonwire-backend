# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import os, json

# Shared helpers
from .common import parse_ts, SummaryContext

# ---------- tiny utils ----------
def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

def _safe_div(a: float, b: float) -> Optional[float]:
    try:
        if b == 0:
            return None
        return a / b
    except Exception:
        return None

def _class_from_f1(f1: float, n_labels: int) -> Tuple[str, str]:
    # Insufficient labels always wins
    if n_labels < int(os.getenv("MW_THRESHOLD_MIN_LABELS", "2")):
        return "Insufficient", "ℹ️"
    if f1 >= 0.75:
        return "Strong", "✅"
    if f1 >= 0.40:
        return "Mixed", "⚠️"
    return "Weak", "❌"

def _nearest_join(
    labels: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
    join_min: int,
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """
    Join each label to the closest trigger within ±join_min minutes on the same origin.
    Returns list of (label_row, matched_trigger_or_None). A trigger may match multiple labels.
    """
    by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in triggers:
        by_origin[str(t.get("origin") or "unknown")].append(t)
    for lst in by_origin.values():
        lst.sort(key=lambda r: (parse_ts(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc)))

    out: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    window = timedelta(minutes=join_min)
    for lab in labels:
        o = str(lab.get("origin") or "unknown")
        lt = parse_ts(lab.get("timestamp"))
        if lt is None:
            out.append((lab, None))
            continue
        best: Optional[Dict[str, Any]] = None
        best_dt: Optional[timedelta] = None
        for trig in by_origin.get(o, []):
            tt = parse_ts(trig.get("timestamp"))
            if tt is None:
                continue
            dt = abs(tt - lt)
            if dt <= window and (best_dt is None or dt < best_dt):
                best, best_dt = trig, dt
        out.append((lab, best))
    return out

def _ensure_emoji(per_version: List[Dict[str, Any]]) -> None:
    """Guarantee each row has an emoji based on class."""
    emap = {"Strong": "✅", "Mixed": "⚠️", "Weak": "❌", "Insufficient": "ℹ️"}
    for r in per_version or []:
        if "emoji" not in r or not r.get("emoji"):
            r["emoji"] = emap.get(r.get("class") or "", "ℹ️")

def _maybe_seed_series_if_demo(
    series: List[Dict[str, Any]],
    per_version: List[Dict[str, Any]],
    now: datetime,
    is_demo: bool,
) -> List[Dict[str, Any]]:
    if series and len(series) >= 2:
        return series
    if not is_demo:
        return series or []
    # Seed 2–3 versions using their snapshot precision as endpoint, with slight slope
    out: List[Dict[str, Any]] = []
    base_times = [now - timedelta(hours=6), now - timedelta(hours=3), now]
    use = per_version[:3] if per_version else [
        {"version": "v0.5.9", "precision": 0.70, "class": "Mixed"},
        {"version": "v0.5.8", "precision": 0.80, "class": "Strong"},
    ]
    for row in use:
        v = row.get("version") or "unknown"
        p_end = float(row.get("precision") or 0.6)
        # simple drift up over time
        vals = [max(0.0, min(1.0, p_end - 0.05)), max(0.0, min(1.0, p_end - 0.02)), p_end]
        for t, p in zip(base_times, vals):
            out.append({"version": v, "t": _iso(t), "precision": p})
    return out

# ---------- main section ----------
def append(md: List[str], ctx: SummaryContext) -> None:
    models_dir = ctx.models_dir
    window_h = int(os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_SIGNAL_JOIN_MIN", "5"))
    min_labels = int(os.getenv("MW_THRESHOLD_MIN_LABELS", "2"))
    want_chart = os.getenv("MW_SIGNAL_VERSION_CHART", "true").lower() in ("1", "true", "yes")

    out_json = models_dir / "signal_quality_per_version.json"
    now = datetime.now(timezone.utc)

    # If the file already exists, load it; otherwise compute from logs.
    data: Dict[str, Any] = {}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    per_version = data.get("per_version")
    series = data.get("series")

    if not isinstance(per_version, list):
        # compute snapshot from raw logs (join by origin ±join_min)
        trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
        lab_rows = _load_jsonl(models_dir / "label_feedback.jsonl")

        t_cut = now - timedelta(hours=window_h)
        trig_rows = [r for r in trig_rows if (parse_ts(r.get("timestamp")) or now) >= t_cut]
        lab_rows = [r for r in lab_rows if (parse_ts(r.get("timestamp")) or now) >= t_cut]

        joined = _nearest_join(lab_rows, trig_rows, join_min)

        # per-version counting (version preference: label.model_version > trigger.model_version > "unknown")
        by_v: Dict[str, Dict[str, int]] = defaultdict(lambda: {"true": 0, "false": 0, "labels": 0, "triggers": 0})
        for lab, trig in joined:
            v = lab.get("model_version") or (trig or {}).get("model_version") or "unknown"
            if trig is not None:
                by_v[v]["triggers"] += 1
            if lab.get("label") is True:
                by_v[v]["true"] += 1
                by_v[v]["labels"] += 1
            elif lab.get("label") is False:
                by_v[v]["false"] += 1
                by_v[v]["labels"] += 1
            # None labels ignored

        per_version = []
        for v, c in by_v.items():
            tp = int(c["true"])
            fp = int(c["false"])
            # We cannot safely infer FN without unmatched positives context; keep recall=TP/(TP+FN) with FN=0 for snapshot parity
            fn = 0
            P = _safe_div(tp, tp + fp) or 0.0
            R = _safe_div(tp, tp + fn) or 0.0
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

        # Sort Strong → Mixed → Weak → Insufficient, tie-break by f1 desc then version
        order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
        per_version.sort(key=lambda r: (order.get(r["class"], 9), -r["f1"], r["version"]))

        # If empty and demo mode, seed plausible snapshot
        if not per_version and ctx.is_demo:
            per_version = [
                {"version": "v0.5.8", "triggers": 8, "true": 6, "false": 2, "labels": 8,
                 "precision": 0.75, "recall": 0.75, "f1": 0.75, "class": "Strong", "emoji": "✅", "demo": True},
                {"version": "v0.5.9", "triggers": 10, "true": 7, "false": 3, "labels": 10,
                 "precision": 0.70, "recall": 0.64, "f1": 0.67, "class": "Mixed", "emoji": "⚠️", "demo": True},
            ]

        data = {
            "window_hours": window_h,
            "join_minutes": join_min,
            "generated_at": _iso(now),
            "per_version": per_version,
            "series": [],
            "demo": ctx.is_demo,
        }

    # Ensure emoji keys are present even if loaded from file
    _ensure_emoji(per_version or [])

    # Ensure series exists; seed if demo
    if not isinstance(data.get("series"), list):
        data["series"] = []
    data["series"] = _maybe_seed_series_if_demo(data["series"], data.get("per_version") or [], now, ctx.is_demo)

    # Persist JSON (with possibly seeded series)
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # -------- markdown (snapshot) --------
    md.append(f"### 🧪 Per-Version Signal Quality ({window_h}h){' (demo)' if data.get('demo') else ''}")
    if per_version:
        for r in per_version:
            v = r.get("version", "unknown")
            emoji = r.get("emoji", "ℹ️")
            klass = r.get("class", "Insufficient")
            f1 = float(r.get("f1", 0.0))
            p = float(r.get("precision", 0.0))
            rcl = float(r.get("recall", 0.0))
            n = int(r.get("labels", 0))
            md.append(f"- `{v}` → {emoji} {klass} (F1={f1:.2f}, P={p:.2f}, R={rcl:.2f}, n={n})")
    else:
        md.append("_no per-version summary available_")

    # -------- chart (trend) --------
    series_data = data.get("series") or []
    if want_chart and series_data:
        # Resolve artifacts dir beside models/, or via env override
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(ctx.models_dir.parent / "artifacts")))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_png = artifacts_dir / f"signal_quality_by_version_{window_h}h.png"

        # Headless plotting
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        # Prepare series by version
        by_v: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        for pt in series_data:
            t = parse_ts(pt.get("t"))
            p = pt.get("precision")
            v = pt.get("version") or "unknown"
            if t is None or p is None:
                continue
            by_v[v].append((t, float(p)))

        if by_v:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            # Shaded classification zones
            ax.axhspan(0.75, 1.00, alpha=0.08)  # Strong
            ax.axhspan(0.40, 0.75, alpha=0.05)  # Mixed
            ax.axhspan(0.00, 0.40, alpha=0.08)  # Weak

            for v, pts in sorted(by_v.items()):
                pts.sort(key=lambda x: x[0])
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker="o", linewidth=1.25, label=v)

            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Precision")
            ax.set_title(f"Per-Version Precision ({window_h}h)")
            ax.legend(loc="best", fontsize=8, ncol=2)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)

            md.append(f"\n📈 Per-Version Precision Trend: {out_png}")
        else:
            md.append("\n📈 Per-Version Precision Trend: _no time series available_ (enable DEMO_MODE or accumulate runs)")
    else:
        md.append("\n📈 Per-Version Precision Trend: _no time series available_ (enable DEMO_MODE or accumulate runs)")
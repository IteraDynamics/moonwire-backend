# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json, math
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import SummaryContext, parse_ts

# ----------------- helpers -----------------

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def _safe_div(a: float, b: float) -> float | None:
    return (a / b) if b else None

def _class_from_f1(f1: float | None, n: int) -> Tuple[str, str]:
    if n < int(os.getenv("MW_THRESHOLD_MIN_LABELS", "2")):
        return ("Insufficient", "ℹ️")
    if f1 is None:
        return ("Insufficient", "ℹ️")
    if f1 >= 0.75: return ("Strong", "✅")
    if f1 >= 0.40: return ("Mixed", "⚠️")
    return ("Weak", "❌")

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln: continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

def _nearest_join(
    labels: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
    join_min: int,
) -> List[Tuple[Dict[str,Any], Dict[str,Any] | None]]:
    """Join each label to the nearest trigger on same origin within ±join_min."""
    by_origin: DefaultDict[str, List[Dict[str,Any]]] = defaultdict(list)
    for t in triggers:
        if t.get("origin"):
            by_origin[str(t.get("origin"))].append(t)
    for o in by_origin:
        by_origin[o].sort(key=lambda r: parse_ts(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))

    joined: List[Tuple[Dict[str,Any], Dict[str,Any] | None]] = []
    tol = timedelta(minutes=join_min)
    for lab in labels:
        o = lab.get("origin")
        if not o:
            joined.append((lab, None)); continue
        t_lab = parse_ts(lab.get("timestamp"))
        if not t_lab:
            joined.append((lab, None)); continue
        cand = None
        best_dt = None
        for t in by_origin.get(o, []):
            ts = parse_ts(t.get("timestamp"))
            if not ts: continue
            d = abs(ts - t_lab)
            if best_dt is None or d < best_dt:
                best_dt = d
                cand = t
        if best_dt is not None and best_dt <= tol:
            joined.append((lab, cand))
        else:
            joined.append((lab, None))
    return joined

def _read_per_origin_thresholds(models_dir: Path) -> Dict[str, float]:
    p = models_dir / "per_origin_thresholds.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _maybe_seed_series_if_demo(series: List[Dict[str,Any]], per_version: List[Dict[str,Any]], now: datetime, is_demo: bool) -> List[Dict[str,Any]]:
    """If DEMO_MODE and no series yet, synthesize a tiny time series so CI shows the chart."""
    if series or not is_demo:
        return series
    out = []
    # three points per version: now-6h, now-3h, now; small jitter trend towards current precision
    for pv in per_version:
        v = pv.get("version") or "unknown"
        p_now = pv.get("precision") or 0.6
        pts = [-6, -3, 0]
        base = max(0.05, min(0.95, p_now))
        for h in pts:
            # simple linear ramp to current precision + slight noise
            frac = (h + 6) / 6.0  # -6 => 0, 0 => 1
            val = max(0.01, min(0.99, 0.6 * (1-frac) + base * frac + (0.02 if h == -3 else 0.0)))
            out.append({"version": v, "t": _iso(now + timedelta(hours=h)), "precision": round(val, 2)})
    return out

def _plot_series(series: List[Dict[str,Any]], window_h: int, out_path: Path) -> None:
    if not series:
        return
    # group by version
    by_v: DefaultDict[str, List[Tuple[datetime,float]]] = defaultdict(list)
    for r in series:
        t = parse_ts(r.get("t"))
        p = r.get("precision")
        if t and isinstance(p, (int,float)):
            by_v[str(r.get("version","unknown"))].append((t, float(p)))
    if not by_v:
        return
    for v in by_v:
        by_v[v].sort(key=lambda x: x[0])

    plt.figure(figsize=(8, 3.5))
    # Shaded bands: red <0.4, yellow 0.4–0.75, green >=0.75
    ax = plt.gca()
    ax.axhspan(0.0, 0.40, alpha=0.08)
    ax.axhspan(0.40, 0.75, alpha=0.08)
    ax.axhspan(0.75, 1.00, alpha=0.08)

    for v, pts in by_v.items():
        xs = [t for (t, _) in pts]
        ys = [p for (_, p) in pts]
        ax.plot(xs, ys, marker="o", label=v)  # (No explicit colors/styles per project rules)

    ax.set_title(f"Per-Version Signal Quality ({window_h}h)")
    ax.set_xlabel("time")
    ax.set_ylabel("precision")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()

# ----------------- main section -----------------

def append(md: List[str], ctx: SummaryContext) -> None:
    models_dir = ctx.models_dir
    window_h = int(os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_SIGNAL_JOIN_MIN", "5"))
    min_labels = int(os.getenv("MW_THRESHOLD_MIN_LABELS", "2"))
    want_chart = os.getenv("MW_SIGNAL_VERSION_CHART", "true").lower() in ("1","true","yes")

    out_json = models_dir / "signal_quality_per_version.json"
    now = datetime.now(timezone.utc)

    # If the file already exists, load it; otherwise compute snapshot from logs (labels+triggers).
    data: Dict[str, Any] = {}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    per_version = data.get("per_version")
    series = data.get("series")

    if not isinstance(per_version, list):
        # compute snapshot from raw logs (same join as earlier)
        trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
        lab_rows  = _load_jsonl(models_dir / "label_feedback.jsonl")

        t_cut = now - timedelta(hours=window_h)
        trig_rows = [r for r in trig_rows if (parse_ts(r.get("timestamp")) or now) >= t_cut]
        lab_rows  = [r for r in lab_rows  if (parse_ts(r.get("timestamp")) or now) >= t_cut]

        joined = _nearest_join(lab_rows, trig_rows, join_min)

        # per-version counting
        by_v = defaultdict(lambda: {"true":0,"false":0,"labels":0,"triggers":0})
        # We count "triggers" as joined rows that have a matched trigger
        for lab, trig in joined:
            # resolve version preference: label.model_version > trigger.model_version > "unknown"
            v = lab.get("model_version") or (trig or {}).get("model_version") or "unknown"
            if trig is not None:
                by_v[v]["triggers"] += 1
            if lab.get("label") is True:
                by_v[v]["true"] += 1
                by_v[v]["labels"] += 1
            elif lab.get("label") is False:
                by_v[v]["false"] += 1
                by_v[v]["labels"] += 1
            # labels that are None/absent are ignored

        per_version = []
        for v, c in by_v.items():
            tp = int(c["true"])
            fp = int(c["false"])
            fn = 0  # We can’t infer FN reliably without unmatched positives; keep consistent with earlier spec
            P = _safe_div(tp, tp+fp) or 0.0
            R = _safe_div(tp, tp+fn) or 0.0
            F1 = _safe_div(2*P*R, (P+R)) if (P+R) else 0.0
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

        # Sort Strong → Mixed → Weak → Insufficient
        order = {"Strong":0, "Mixed":1, "Weak":2, "Insufficient":3}
        per_version.sort(key=lambda r: (order.get(r["class"], 9), -r["f1"], r["version"]))

        # If empty and demo mode, seed a plausible snapshot (kept from prior behavior)
        if not per_version and ctx.is_demo:
            per_version = [
                {"version":"v0.5.9","triggers":8,"true":6,"false":2,"labels":8,"precision":0.75,"recall":0.75,"f1":0.75,"class":"Strong","emoji":"✅","demo":True},
                {"version":"v0.5.8","triggers":10,"true":7,"false":3,"labels":10,"precision":0.7,"recall":0.64,"f1":0.67,"class":"Mixed","emoji":"⚠️","demo":True},
            ]

        data = {
            "window_hours": window_h,
            "join_minutes": join_min,
            "generated_at": _iso(now),
            "per_version": per_version,
            "series": [],
            "demo": ctx.is_demo,
        }

    # Ensure series exists; if demo and empty, synthesize so the chart appears in CI.
    if not isinstance(data.get("series"), list):
        data["series"] = []
    data["series"] = _maybe_seed_series_if_demo(data["series"], data.get("per_version") or [], now, ctx.is_demo)

    # Persist JSON (with possibly seeded series)
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # -------- markdown --------
    md.append(f"### 🧪 Per-Version Signal Quality ({window_h}h){' (demo)' if data.get('demo') else ''}")
    if per_version:
        for r in per_version:
            v = r["version"]
            md.append(f"- `{v}` → {r['emoji']} {r['class']} (F1={r['f1']:.2f}, P={r['precision']:.2f}, R={r['recall']:.2f}, n={int(r['labels'])})")
    else:
        md.append("_no per-version summary available_")

    # -------- chart --------
    chart_path = Path("artifacts") / f"signal_quality_by_version_{window_h}h.png"
    if want_chart and data.get("series"):
        _plot_series(data["series"], window_h, chart_path)
        md.append(f"\n📈 Per-Version Precision Trend: ![]({chart_path.as_posix()})")
    else:
        md.append("\n📈 Per-Version Precision Trend: _no time series available_ (enable DEMO_MODE or accumulate runs)")
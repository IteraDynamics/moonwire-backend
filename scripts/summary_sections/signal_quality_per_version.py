# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import SummaryContext, parse_ts

# ----------------- helpers -----------------

CLASS_EMOJI = {
    "Strong": "✅",
    "Mixed": "⚠️",
    "Weak": "❌",
    "Insufficient": "ℹ️",
}

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def _safe_div(a: float, b: float) -> float | None:
    return (a / b) if b else None

def _class_from_f1(f1: float | None, n: int) -> Tuple[str, str]:
    min_labels = int(os.getenv("MW_THRESHOLD_MIN_LABELS", "2"))
    if n < min_labels: 
        return ("Insufficient", CLASS_EMOJI["Insufficient"])
    if f1 is None:
        return ("Insufficient", CLASS_EMOJI["Insufficient"])
    if f1 >= 0.75: return ("Strong", CLASS_EMOJI["Strong"])
    if f1 >= 0.40: return ("Mixed", CLASS_EMOJI["Mixed"])
    return ("Weak", CLASS_EMOJI["Weak"])

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
        o = t.get("origin")
        if o:
            by_origin[str(o)].append(t)
    for o in by_origin:
        by_origin[o].sort(key=lambda r: parse_ts(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))

    joined: List[Tuple[Dict[str,Any], Dict[str,Any] | None]] = []
    tol = timedelta(minutes=join_min)
    for lab in labels:
        o = lab.get("origin")
        t_lab = parse_ts(lab.get("timestamp"))
        if not o or not t_lab:
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

def _maybe_seed_series_if_demo(series: List[Dict[str,Any]], per_version: List[Dict[str,Any]], now: datetime, is_demo: bool) -> List[Dict[str,Any]]:
    """If DEMO_MODE and no series yet, synthesize a tiny time series so CI shows the chart."""
    if series or not is_demo:
        return series
    out = []
    for pv in per_version:
        v = pv.get("version") or "unknown"
        p_now = pv.get("precision") or 0.6
        pts = [-6, -3, 0]
        base = max(0.05, min(0.95, float(p_now)))
        for h in pts:
            frac = (h + 6) / 6.0  # -6 => 0, 0 => 1
            # simple ramp toward current with a tiny mid bump
            val = max(0.01, min(0.99, 0.6 * (1-frac) + base * frac + (0.02 if h == -3 else 0.0)))
            out.append({"version": v, "t": _iso(now + timedelta(hours=h)), "precision": round(val, 2)})
    return out

def _plot_series(series: List[Dict[str,Any]], window_h: int, out_path: Path) -> None:
    if not series:
        return
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
    ax = plt.gca()
    # Background quality bands (no explicit colors; rely on default facecolor + alpha)
    ax.axhspan(0.0, 0.40, alpha=0.08)
    ax.axhspan(0.40, 0.75, alpha=0.08)
    ax.axhspan(0.75, 1.00, alpha=0.08)

    for v, pts in by_v.items():
        xs = [t for (t, _) in pts]
        ys = [p for (_, p) in pts]
        ax.plot(xs, ys, marker="o", label=v)

    ax.set_title(f"Per-Version Signal Quality ({window_h}h)")
    ax.set_xlabel("time")
    ax.set_ylabel("precision")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()

def _normalize_per_version(per_version: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Make loaded entries robust: fill emoji, compute F1 if missing, round numbers, backfill counts."""
    out = []
    min_labels = int(os.getenv("MW_THRESHOLD_MIN_LABELS", "2"))
    for r in per_version:
        v = str(r.get("version", "unknown"))
        tp = int(r.get("true", 0) or 0)
        fp = int(r.get("false", 0) or 0)
        labels = r.get("labels")
        if labels is None:
            labels = tp + fp
        labels = int(labels)
        # precision/recall/f1
        P = r.get("precision")
        R = r.get("recall")
        F1 = r.get("f1")
        # If P/R missing, try to infer from counts (FN unknown → recall becomes 1.0 if labels==tp)
        if not isinstance(P, (int, float)):
            P = _safe_div(tp, (tp + fp)) or 0.0
        if not isinstance(R, (int, float)):
            # Without FN we can't infer properly; treat observed positives as recall proxy
            R = 1.0 if labels == tp and labels > 0 else (tp / labels if labels else 0.0)
        if not isinstance(F1, (int, float)):
            F1 = (2*P*R / (P+R)) if (P+R) else 0.0
        # class / emoji
        klass = r.get("class")
        if not klass:
            klass, emoji = _class_from_f1(F1, labels)
        else:
            emoji = r.get("emoji") or CLASS_EMOJI.get(klass, CLASS_EMOJI["Insufficient"])
        # finalize
        out.append({
            "version": v,
            "triggers": int(r.get("triggers", labels) or 0),
            "true": tp,
            "false": fp,
            "labels": labels,
            "precision": round(float(P), 2),
            "recall": round(float(R), 2),
            "f1": round(float(F1), 2),
            "class": klass if klass else _class_from_f1(F1, labels)[0],
            "emoji": emoji,
            "demo": bool(r.get("demo", False)),
        })
    # sort Strong → Mixed → Weak → Insufficient
    order = {"Strong":0, "Mixed":1, "Weak":2, "Insufficient":3}
    out.sort(key=lambda r: (order.get(r["class"], 9), -r["f1"], r["version"]))
    return out

# ----------------- main section -----------------

def append(md: List[str], ctx: SummaryContext) -> None:
    models_dir = ctx.models_dir
    window_h = int(os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_SIGNAL_JOIN_MIN", "5"))
    want_chart = os.getenv("MW_SIGNAL_VERSION_CHART", "true").lower() in ("1","true","yes")

    out_json = models_dir / "signal_quality_per_version.json"
    now = datetime.now(timezone.utc)

    # If the file already exists, load it; otherwise compute snapshot from logs.
    data: Dict[str, Any] = {}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    per_version = data.get("per_version")
    series = data.get("series")

    if not isinstance(per_version, list):
        # compute snapshot from raw logs
        trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
        lab_rows  = _load_jsonl(models_dir / "label_feedback.jsonl")

        t_cut = now - timedelta(hours=window_h)
        trig_rows = [r for r in trig_rows if (parse_ts(r.get("timestamp")) or now) >= t_cut]
        lab_rows  = [r for r in lab_rows  if (parse_ts(r.get("timestamp")) or now) >= t_cut]

        joined = _nearest_join(lab_rows, trig_rows, join_min)

        by_v = defaultdict(lambda: {"true":0,"false":0,"labels":0,"triggers":0})
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

        computed = []
        for v, c in by_v.items():
            tp = int(c["true"])
            fp = int(c["false"])
            fn = 0  # unknown from these logs
            P = _safe_div(tp, tp+fp) or 0.0
            R = _safe_div(tp, tp+fn) or 0.0
            F1 = _safe_div(2*P*R, (P+R)) if (P+R) else 0.0
            klass, emoji = _class_from_f1(F1, int(c["labels"]))
            computed.append({
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
        per_version = _normalize_per_version(computed)

        data = {
            "window_hours": window_h,
            "join_minutes": join_min,
            "generated_at": _iso(now),
            "per_version": per_version,
            "series": [],
            "demo": ctx.is_demo,
        }
    else:
        # Normalize existing loaded snapshot so tests that omit fields won't break
        per_version = _normalize_per_version(per_version)
        data["per_version"] = per_version
        if not isinstance(series, list):
            data["series"] = []

    # Seed series in DEMO if empty so CI shows a chart
    data["series"] = _maybe_seed_series_if_demo(data.get("series") or [], per_version, now, bool(data.get("demo")) or ctx.is_demo)

    # Persist JSON (normalized + possibly seeded series)
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
from __future__ import annotations

import os, json, statistics
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .common import SummaryContext, parse_ts, _iso, ensure_dir, is_demo_mode

# ---- Config defaults ----
_LOW_N = 30
_HIGH_ECE = 0.06
_DEFAULT_BINS = int(os.getenv("MW_CAL_BINS", "10"))


def _floor_bucket(ts: datetime, bucket_min: int) -> datetime:
    base = ts.replace(second=0, microsecond=0)
    delta = (base.minute + base.hour * 60) % bucket_min
    return base - timedelta(minutes=delta)


def _compute_metrics(y_true: List[int], y_prob: List[float], bins: int) -> tuple[float, float]:
    if not y_true:
        return 0.0, 0.0
    probs = [min(1.0, max(0.0, float(p))) for p in y_prob]
    trues = [1 if y else 0 for y in y_true]

    # Brier
    brier = statistics.fmean((p - t) ** 2 for p, t in zip(probs, trues))

    # ECE
    edges = [i / bins for i in range(bins + 1)]
    per_bin = [[] for _ in range(bins)]
    for i, p in enumerate(probs):
        b = min(int(p * bins), bins - 1)
        per_bin[b].append(i)
    ece = 0.0
    N = len(probs)
    for b, idxs in enumerate(per_bin):
        if not idxs:
            continue
        p_mean = statistics.fmean(probs[i] for i in idxs)
        t_mean = statistics.fmean(trues[i] for i in idxs)
        ece += (len(idxs) / N) * abs(t_mean - p_mean)
    return float(ece), float(brier)


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def _bucket_join(triggers: List[dict], labels: List[dict],
                 window_h: int, bucket_min: int,
                 dim: str) -> Dict[str, Dict[datetime, Dict[str, Any]]]:
    now = datetime.now(timezone.utc)
    t_start = now - timedelta(hours=window_h)

    # Map labels by (id)
    label_by_id: Dict[str, int] = {}
    for r in labels:
        rid = r.get("id")
        if not rid:
            continue
        lab = r.get("label")
        lab_i = 1 if lab is True or str(lab).lower() in ("1","true","yes") else 0
        label_by_id[rid] = lab_i

    series: Dict[str, Dict[datetime, Dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {"y": [], "p": []}))
    for tr in triggers:
        rid = tr.get("id")
        if not rid or rid not in label_by_id:
            continue
        ts = parse_ts(tr.get("ts") or tr.get("timestamp"))
        if not ts or ts < t_start or ts > now:
            continue
        key = tr.get("origin") if dim == "origin" else tr.get("model_version") or "unknown"
        p = tr.get("score") or tr.get("adjusted_score")
        try:
            p = float(p)
        except Exception:
            continue
        b = _floor_bucket(ts, bucket_min)
        series[key][b]["y"].append(label_by_id[rid])
        series[key][b]["p"].append(p)
    return series


def _synthesize_demo(dim: str, buckets: List[datetime]) -> Dict[str, Dict[datetime, Dict[str, Any]]]:
    out: Dict[str, Dict[datetime, Dict[str, Any]]] = defaultdict(dict)
    keys = ["reddit","twitter","rss_news"] if dim=="origin" else ["v_good","v_bad"]
    import random
    for k in keys:
        for i,b in enumerate(buckets[-3:]):
            out[k][b] = {
                "y": [1 if random.random() < 0.5 else 0 for _ in range(40)],
                "p": [random.random() for _ in range(40)],
            }
    return out


def append(md: List[str], ctx: SummaryContext) -> None:
    window_h = int(os.getenv("MW_CAL_TREND_WINDOW_H", "72"))
    bucket_min = int(os.getenv("MW_CAL_TREND_BUCKET_MIN", "180"))
    dim = os.getenv("MW_CAL_TREND_DIM", "origin").lower()
    bins = _DEFAULT_BINS

    trig_rows = _load_jsonl(ctx.logs_dir / "trigger_history.jsonl")
    lab_rows = _load_jsonl(ctx.logs_dir / "label_feedback.jsonl")

    series = _bucket_join(trig_rows, lab_rows, window_h, bucket_min, dim)

    # build buckets timeline
    now = datetime.now(timezone.utc)
    t_start = now - timedelta(hours=window_h)
    buckets: List[datetime] = []
    t = _floor_bucket(t_start, bucket_min)
    while t <= _floor_bucket(now, bucket_min):
        buckets.append(t)
        t += timedelta(minutes=bucket_min)

    demo = False
    if (not series or all(len(v)==0 for v in series.values())) and (ctx.is_demo or is_demo_mode()):
        demo = True
        series = _synthesize_demo(dim, buckets)

    results = []
    for key, per_b in series.items():
        pts = []
        for b in buckets:
            y = per_b.get(b, {}).get("y", [])
            p = per_b.get(b, {}).get("p", [])
            if not y:
                continue
            ece, brier = _compute_metrics(y,p,bins)
            n = len(y)
            pts.append({
                "bucket_start": _iso(b),
                "bucket_mid": _iso(b + timedelta(minutes=bucket_min//2)),
                "n": n,
                "ece": round(ece,4),
                "brier": round(brier,4),
                "low_n": n < _LOW_N,
                "high_ece": ece > _HIGH_ECE,
            })
        results.append({"key": key, "points": pts})

    # Write JSON
    out = {
        "window_hours": window_h,
        "bucket_minutes": bucket_min,
        "dimension": dim,
        "generated_at": _iso(now),
        "demo": demo,
        "series": results,
    }
    (ctx.models_dir / "calibration_reliability_trend.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8"
    )

    # Plots
    art_dirs = [
        Path("artifacts"),
        ctx.logs_dir.parent / "artifacts",
        ctx.models_dir.parent / "artifacts",
    ]
    for metric in ("ece","brier"):
        fig, ax = plt.subplots(figsize=(9,3))
        for r in results:
            xs = [parse_ts(p["bucket_mid"]) for p in r["points"]]
            ys = [p[metric] for p in r["points"]]
            if xs and ys:
                ax.plot(xs, ys, marker="o", label=r["key"])
        ax.set_title(f"Calibration Trend — {metric.upper()} ({window_h}h)" + (" (demo)" if demo else ""))
        ax.set_ylabel(metric)
        ax.legend(loc="upper left", fontsize=8)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        last_path = None
        for d in art_dirs:
            try:
                ensure_dir(d)
                p = d / f"calibration_trend_{metric}.png"
                fig.savefig(p, dpi=150)
                last_path = p
            except Exception:
                continue
        plt.close(fig)
    # Markdown
    md.append(f"🧮 Calibration & Reliability Trend ({window_h}h){' (demo)' if demo else ''}")
    for r in results:
        pts = r["points"]
        if len(pts) >= 2:
            e0, e1 = pts[0]["ece"], pts[-1]["ece"]
            b0, b1 = pts[0]["brier"], pts[-1]["brier"]
            trend_e = "upward" if e1>e0 else "downward" if e1<e0 else "steady"
            trend_b = "upward" if b1>b0 else "downward" if b1<b0 else "steady"
            md.append(f"{r['key']:<8} → ECE {trend_e} ({e0:.2f}→{e1:.2f}), Brier {trend_b} ({b0:.2f}→{b1:.2f})")
    md.append(f"- saved: artifacts/calibration_trend_ece.png")
    md.append(f"- saved: artifacts/calibration_trend_brier.png")
# scripts/summary_sections/rolling_accuracy_snapshot.py
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os, json
from scripts.summary_sections.common import SummaryContext

def _parse_ts_any(v):
    if v is None: return None
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        pass
    try:
        s = str(v); s = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None

def _bucket_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)

def _load_jsonl_safe(p: Path) -> list:
    try:
        if not p.exists(): return []
        out = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: continue
        return out
    except Exception:
        return []

def _norm_origin(o): return str(o or "unknown").strip().lower()

def _prf(counts):
    tp = int(counts.get("tp", 0)); fp = int(counts.get("fp", 0)); fn = int(counts.get("fn", 0))
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / float(prec + rec)) if (prec + rec) > 0 else 0.0
    n = tp + fp + fn
    return prec, rec, f1, n

def append(md: list[str], ctx: SummaryContext):
    try:
        min_labels_required = int(os.getenv("METRICS_MIN_LABELS", "10"))
    except Exception:
        min_labels_required = 10
    try:
        lookback_hours = int(os.getenv("METRICS_LOOKBACK_HOURS", "72"))
    except Exception:
        lookback_hours = 72

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)

    hpath = ctx.models_dir / "trigger_history.jsonl"
    fpath = ctx.models_dir / "label_feedback.jsonl"

    H = _load_jsonl_safe(hpath)
    F = _load_jsonl_safe(fpath)

    H = [r for r in H if (_parse_ts_any(r.get("timestamp")) or now) >= cutoff]
    F = [r for r in F if (_parse_ts_any(r.get("timestamp")) or now) >= cutoff]

    hist = {}
    for r in H:
        ts = _parse_ts_any(r.get("timestamp"))
        if not ts: continue
        key = (_norm_origin(r.get("origin")), _bucket_hour(ts))
        trig = bool(r.get("decision") == "triggered" or r.get("triggered") is True)
        hist[key] = {"triggered": trig}

    feed = {}
    for r in F:
        ts = _parse_ts_any(r.get("timestamp"))
        if not ts: continue
        key = (_norm_origin(r.get("origin")), _bucket_hour(ts))
        feed[key] = {"label": bool(r.get("label"))}

    matches = []
    for k in (set(hist.keys()) & set(feed.keys())):
        o = k[0]
        matches.append((o, hist[k]["triggered"], feed[k]["label"]))

    by_origin = {}
    for o, trig, lab in matches:
        c = by_origin.setdefault(o, {"tp": 0, "fp": 0, "fn": 0})
        if trig and lab: c["tp"] += 1
        elif trig and not lab: c["fp"] += 1
        elif (not trig) and lab: c["fn"] += 1

    total_counts = {"tp": 0, "fp": 0, "fn": 0}
    for d in by_origin.values():
        total_counts["tp"] += d["tp"]; total_counts["fp"] += d["fp"]; total_counts["fn"] += d["fn"]
    total_labels = sum(d["tp"] + d["fp"] + d["fn"] for d in by_origin.values())

    seeded = False
    if total_labels < min_labels_required and os.getenv("DEMO_MODE", "false").lower() in ("1","true","yes"):
        seeded = True
        by_origin = {
            "reddit":   {"tp": 2, "fp": 1, "fn": 1},
            "twitter":  {"tp": 1, "fp": 0, "fn": 2},
            "rss_news": {"tp": 0, "fp": 1, "fn": 1},
        }
        total_counts = {
            "tp": sum(d["tp"] for d in by_origin.values()),
            "fp": sum(d["fp"] for d in by_origin.values()),
            "fn": sum(d["fn"] for d in by_origin.values()),
        }

    if seeded:
        md.append(f"\n📈 Rolling Accuracy Snapshot (seeded demo; window={lookback_hours}h)")
    else:
        md.append(f"\n📈 Rolling Accuracy Snapshot (N={total_labels} labels, window={lookback_hours}h)")

    try:
        if not by_origin:
            md.append("waiting for more labels…")
        else:
            for origin in sorted(by_origin.keys()):
                p, r, f1, n = _prf(by_origin[origin])
                tp = by_origin[origin]["tp"]; fp = by_origin[origin]["fp"]; fn = by_origin[origin]["fn"]
                md.append(f"{origin} → precision={p:.2f}, recall={r:.2f}, F1={f1:.2f} (tp={tp}, fp={fp}, fn={fn}, n={n})")
            p, r, f1, n = _prf(total_counts)
            md.append(f"overall → precision={p:.2f}, recall={r:.2f}, F1={f1:.2f} (tp={total_counts['tp']}, fp={total_counts['fp']}, fn={total_counts['fn']}, n={n})")
    except Exception as e:
        md.append(f"⚠️ Rolling accuracy section failed: {e}")
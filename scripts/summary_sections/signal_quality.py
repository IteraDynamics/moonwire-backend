# scripts/summary_sections/signal_quality.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, os, math, statistics

# Matplotlib headless for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .common import parse_ts, is_demo_mode

# ------------------------------------------------------------
# Env helpers
# ------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# ------------------------------------------------------------
# Data containers
# ------------------------------------------------------------
@dataclass
class Batch:
    start: datetime
    end: datetime
    triggers: int
    true: int
    false: int
    precision: float | None
    klass: str
    emoji: str
    demo: bool = False

# ------------------------------------------------------------
# Core compute (3h buckets over window; join within ±join_min)
# ------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
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

def _classify(precision: float | None) -> tuple[str, str]:
    if precision is None:
        return ("Insufficient", "ℹ️")
    if precision >= 0.75:
        return ("Strong", "✅")
    if precision >= 0.40:
        return ("Mixed", "⚠️")
    return ("Weak", "❌")

def _compute_batches(models_dir: Path) -> tuple[list[Batch], dict]:
    """
    Compute batches over window by joining triggers and labels
    on origin + nearest timestamp within ±join_min.
    NOTE: We do NOT require scores here; join is timestamp-based.
    """
    window_h = _env_int("MW_SIGNAL_WINDOW_H", 72)
    batch_h  = _env_int("MW_SIGNAL_BATCH_H", 3)
    join_min = _env_int("MW_SIGNAL_JOIN_MIN", 5)

    now = datetime.now(timezone.utc)
    start_window = now - timedelta(hours=window_h)

    # Keep only events inside the window
    triggers_raw = [r for r in _load_jsonl(models_dir / "trigger_history.jsonl")
                    if (ts := parse_ts(r.get("timestamp"))) and ts >= start_window]
    labels_raw   = [r for r in _load_jsonl(models_dir / "label_feedback.jsonl")
                    if (ts := parse_ts(r.get("timestamp"))) and ts >= start_window]

    # Demo seeding if logs are sparse and DEMO_MODE is enabled
    demo_seeded = False
    if is_demo_mode() and len(triggers_raw) < 6 and len(labels_raw) < 6:
        demo_seeded = True
        base = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=12)
        seeds = [
            (0.88, 8),  # strong
            (0.50, 6),  # mixed
            (0.20, 5),  # weak
            (0.57, 7),  # mixed
        ]
        batches = []
        for i, (p, n) in enumerate(seeds):
            bs = base + timedelta(hours=3*i)
            be = bs + timedelta(hours=3)
            tp = max(0, round(p * n))
            fp = max(0, n - tp)
            klass, emoji = _classify(p)
            batches.append(Batch(bs, be, n, tp, fp, p, klass, emoji, demo=True))
        meta = {
            "window_hours": window_h,
            "batch_hours": batch_h,
            "generated_at": now.isoformat().replace("+00:00","Z"),
            "batches": [
                {
                    "start": b.start.isoformat().replace("+00:00","Z"),
                    "end": b.end.isoformat().replace("+00:00","Z"),
                    "triggers": b.triggers, "true": b.true, "false": b.false,
                    "precision": b.precision, "class": b.klass, "emoji": b.emoji, "demo": True,
                } for b in batches
            ],
            "demo": True,
        }
        return batches, meta

    # Build per-origin trigger timestamp index (no score required)
    trig_by_origin: dict[str, list[datetime]] = {}
    for r in triggers_raw:
        o = r.get("origin") or "unknown"
        ts = parse_ts(r.get("timestamp"))
        if not ts:
            continue
        trig_by_origin.setdefault(o, []).append(ts)
    for o in trig_by_origin:
        trig_by_origin[o].sort()

    # Helper to compute bucket bounds for a timestamp
    def bucket_bounds(ts: datetime) -> tuple[datetime, datetime]:
        delta = ts - start_window
        k = int(delta.total_seconds() // (batch_h * 3600))
        bs = start_window + timedelta(hours=k * batch_h)
        be = bs + timedelta(hours=batch_h)
        return bs, be

    # Join each label to the nearest trigger in the same origin within tolerance
    tol = timedelta(minutes=join_min)
    bucket_map: dict[tuple[datetime, datetime], dict[str, int]] = {}

    for lab in labels_raw:
        o = lab.get("origin") or "unknown"
        lt = parse_ts(lab.get("timestamp"))
        if not lt:
            continue
        label_bool = bool(lab.get("label"))
        times = trig_by_origin.get(o) or []
        # Nearest by absolute time difference
        best_t = None
        best_dt = None
        for tt in times:
            dt = abs(tt - lt)
            if dt <= tol and (best_dt is None or dt < best_dt):
                best_t = tt
                best_dt = dt
        if best_t is None:
            continue
        bs, be = bucket_bounds(best_t)
        d = bucket_map.setdefault((bs, be), {"triggers": 0, "true": 0, "false": 0})
        d["triggers"] += 1   # count of matched trigger/label pairs in this bucket
        if label_bool:
            d["true"] += 1
        else:
            d["false"] += 1

    # Materialize batches in time order
    batches: list[Batch] = []
    for (bs, be) in sorted(bucket_map.keys()):
        d = bucket_map[(bs, be)]
        n_pos = d["true"]
        n_neg = d["false"]
        denom = n_pos + n_neg
        precision = (n_pos / denom) if denom > 0 else None
        klass, emoji = _classify(precision)
        batches.append(Batch(bs, be, d["triggers"], n_pos, n_neg, precision, klass, emoji))

    meta = {
        "window_hours": window_h,
        "batch_hours": batch_h,
        "generated_at": now.isoformat().replace("+00:00","Z"),
        "batches": [
            {
                "start": b.start.isoformat().replace("+00:00","Z"),
                "end": b.end.isoformat().replace("+00:00","Z"),
                "triggers": b.triggers, "true": b.true, "false": b.false,
                "precision": b.precision, "class": b.klass, "emoji": b.emoji, "demo": b.demo,
            } for b in batches
        ],
        "demo": False,
    }
    return batches, meta

# ------------------------------------------------------------
# Trend chart from JSON artifact
# ------------------------------------------------------------
def _plot_trend_from_json(meta: dict, out_path: Path) -> bool:
    """
    Build a precision-over-time chart using meta['batches'].
    Returns True if an image was written.
    """
    batches = meta.get("batches") or []
    if not batches:
        return False

    xs, ys, classes, ns = [], [], [], []
    any_demo = False
    for b in batches:
        st = parse_ts(b.get("start"))
        en = parse_ts(b.get("end"))
        pr = b.get("precision", None)
        if not st or not en or pr is None:
            continue
        t = st + (en - st)/2  # midpoint
        xs.append(t)
        ys.append(float(pr))
        classes.append((b.get("class") or "").lower())
        ns.append(int(b.get("true", 0)) + int(b.get("false", 0)))
        if b.get("demo"):
            any_demo = True

    if not xs:
        return False

    fig, ax = plt.subplots(figsize=(8, 3))

    # Background class bands
    ax.axhspan(0.75, 1.00, alpha=0.10, color="green")
    ax.axhspan(0.40, 0.75, alpha=0.10, color="gold")
    ax.axhspan(0.00, 0.40, alpha=0.10, color="red")

    # Line (solid vs dashed if demo)
    ls = "--" if any_demo else "-"
    ax.plot(xs, ys, linewidth=1.2, linestyle=ls)

    # Scatter by class
    color_by_class = {"strong": "green", "mixed": "gold", "weak": "red"}
    for t, y, c, n in zip(xs, ys, classes, ns):
        ax.scatter([t], [y], s=28, zorder=3, color=color_by_class.get(c, "gray"))
        if n < 3:
            ax.annotate("n<3", (mdates.date2num(t), y),
                        xytext=(3, 6), textcoords="offset points", fontsize=7)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("precision")
    ax.set_xlabel("time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    title = "Signal Quality Trend (72h)"
    if any_demo:
        title += " [demo]"
    ax.set_title(title, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True

# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------
def append(md: list[str], ctx, *, write_json: bool = True) -> None:
    """
    Appends the textual Signal Quality Summary (as before) and
    now also renders a trend chart read from the JSON artifact.
    """
    models_dir: Path = ctx.models_dir
    window_h = _env_int("MW_SIGNAL_WINDOW_H", 72)
    batch_h  = _env_int("MW_SIGNAL_BATCH_H", 3)

    # If the artifact exists, reuse it; otherwise compute from logs (and write).
    meta_path = models_dir / "signal_quality_summary.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            window_h = int(meta.get("window_hours", window_h))
            batch_h  = int(meta.get("batch_hours", batch_h))
        except Exception:
            meta = None
    else:
        meta = None

    batches: list[Batch]
    if meta is None:
        batches, meta = _compute_batches(models_dir)
        if write_json:
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        batches = []
        for b in meta.get("batches", []):
            st = parse_ts(b.get("start"))
            en = parse_ts(b.get("end"))
            if not st or not en:
                continue
            pr = b.get("precision", None)
            klass = b.get("class") or "Insufficient"
            emoji = b.get("emoji") or "ℹ️"
            batches.append(Batch(
                start=st, end=en,
                triggers=int(b.get("triggers", 0)),
                true=int(b.get("true", 0)),
                false=int(b.get("false", 0)),
                precision=(float(pr) if pr is not None else None),
                klass=klass, emoji=emoji,
                demo=bool(b.get("demo", False)),
            ))

    # ---------- Markdown ----------
    md.append(f"### 🧪 Signal Quality Summary (last {window_h}h, {batch_h}h buckets)")
    if meta.get("demo"):
        md.append("(seeded demo data)")

    if not batches:
        md.append("_no recent batches_")
    else:
        for b in sorted(batches, key=lambda b: b.start):
            hhmm = lambda dt: dt.strftime("%H:%M")
            pr_str = "n/a" if b.precision is None else f"{b.precision:.2f}"
            md.append(f"[{hhmm(b.start)}–{hhmm(b.end)}] → {b.emoji} {b.klass} "
                      f"(precision={pr_str}, n={b.triggers})")

    # ---------- Trend chart ----------
    art_dir = Path("artifacts")
    img_path = art_dir / f"signal_quality_trend_{window_h}h.png"
    wrote = _plot_trend_from_json(meta, img_path)
    if wrote:
        md.append("")
        md.append(f"![Signal quality trend ({window_h}h)]({img_path.as_posix()})")
        # If very sparse, print inline values for convenience
        bs = meta.get("batches") or []
        vals = [(parse_ts(b.get("end")) or parse_ts(b.get("start")), b.get("precision"))
                for b in bs if b.get("precision") is not None]
        vals = [(t, float(p)) for (t, p) in vals if t]
        if 0 < len(vals) <= 3:
            vals_sorted = sorted(vals, key=lambda x: x[0])
            pretty = ", ".join(f"{t.strftime('%m-%d %H:%M')}={p:.2f}" for t, p in vals_sorted)
            md.append(f"_values_: {pretty}")

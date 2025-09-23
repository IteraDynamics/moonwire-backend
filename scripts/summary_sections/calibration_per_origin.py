# scripts/summary_sections/calibration_per_origin.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import common  # use your helpers & SummaryContext

# --- Config / thresholds ---
_LOW_N = 30
_HIGH_ECE = 0.06
_DEFAULT_BINS = 10


def _norm_origin(name: str | None) -> str:
    s = (name or "unknown").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s or "unknown"


def _linspace_0_1(n_bins: int) -> List[Tuple[float, float]]:
    step = 1.0 / n_bins
    # [0, step), [step, 2*step), ... last bin includes 1.0 when assigning
    return [(i * step, (i + 1) * step) for i in range(n_bins)]


def _compute_calibration(
    y_true: List[int], y_prob: List[float], n_bins: int = _DEFAULT_BINS
) -> tuple[float, float, list[dict]]:
    """
    Pure-Python ECE + Brier + fixed-edge reliability bins (length == n_bins).
    Bins cover [0,1), last bin includes 1.0 via index clamp.
    """
    assert len(y_true) == len(y_prob)
    n = len(y_true)
    edges = _linspace_0_1(n_bins)
    if n == 0:
        return 0.0, 0.0, [
            {"low": a, "high": b, "avg_conf": None, "emp_rate": None, "count": 0}
            for a, b in edges
        ]

    # Brier score
    brier = sum((float(p) - float(y)) ** 2 for p, y in zip(y_prob, y_true)) / n

    # Bin accumulation
    acc = [{"sum_p": 0.0, "sum_y": 0.0, "count": 0} for _ in range(n_bins)]
    for y, p in zip(y_true, y_prob):
        p = max(0.0, min(1.0, float(p)))
        idx = min(int(p * n_bins), n_bins - 1)  # [0..n_bins-1]
        a = acc[idx]
        a["sum_p"] += p
        a["sum_y"] += float(y)
        a["count"] += 1

    bins = []
    ece = 0.0
    for (low, high), a in zip(edges, acc):
        c = a["count"]
        if c > 0:
            avg_conf = a["sum_p"] / c
            emp_rate = a["sum_y"] / c
            ece += (c / n) * abs(emp_rate - avg_conf)
        else:
            avg_conf = None
            emp_rate = None
        bins.append(
            {"low": low, "high": high, "avg_conf": avg_conf, "emp_rate": emp_rate, "count": c}
        )

    return float(ece), float(brier), bins


def _plot_reliability(bins: list[dict], title: str, out_path: Path) -> None:
    # Matplotlib only when needed; CI-safe Agg backend.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = [], []
    for b in bins:
        if b.get("count", 0) and b.get("avg_conf") is not None and b.get("emp_rate") is not None:
            xs.append(float(b["avg_conf"]))
            ys.append(float(b["emp_rate"]))

    common.ensure_dir(out_path.parent)
    plt.figure()
    # identity line
    plt.plot([0, 1], [0, 1])
    # points
    if xs and ys:
        plt.scatter(xs, ys)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical outcome rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _read_trigger_and_labels(logs_dir: Path) -> tuple[list[dict], list[dict]]:
    th_path = logs_dir / "trigger_history.jsonl"
    lf_path = logs_dir / "label_feedback.jsonl"
    triggers = common._load_jsonl(th_path)
    labels = common._load_jsonl(lf_path)
    return triggers, labels


def _parse_ts_any(val):
    """
    Accept ISO8601, epoch seconds, or epoch milliseconds.
    Uses common.parse_ts for ISO and normal seconds; falls back to ms detection.
    """
    ts = common.parse_ts(val)
    if ts:
        return ts
    try:
        num = float(val)
        # If it's bigger than 1e12, treat it as milliseconds since epoch
        if num > 1_000_000_000_000:
            num = num / 1000.0
        return datetime.fromtimestamp(num, tz=timezone.utc)
    except Exception:
        return None


def _resolve_artifacts_dir(ctx) -> Path:
    """
    Pick artifacts dir the same way CI expects:
    1) ARTIFACTS_DIR env
    2) ctx.artifacts_dir (if orchestrator provides it)
    3) <repo_root>/artifacts where repo_root is models_dir.parent
    """
    env_dir = os.getenv("ARTIFACTS_DIR")
    if env_dir:
        return common.ensure_dir(Path(env_dir))
    # Some orchestrators stash this on the context
    if hasattr(ctx, "artifacts_dir") and getattr(ctx, "artifacts_dir"):
        return common.ensure_dir(Path(getattr(ctx, "artifacts_dir")))
    # Fallback: sibling to models/
    return common.ensure_dir(Path(ctx.models_dir).parent / "artifacts")


def _window_rows(
    triggers: list[dict],
    labels: list[dict],
    now_utc: datetime,
    hours: int,
) -> list[dict]:
    """Join triggers↔labels on id; keep labeled rows within [now-hours, now]."""
    label_by_id: dict[str, int] = {}
    for r in labels:
        rid = r.get("id")
        if not rid:
            continue
        lab = r.get("label")
        lab_i = 1 if str(lab).lower() in ("1", "true", "t", "yes") or lab is True else 0
        label_by_id[rid] = lab_i

    start = now_utc - timedelta(hours=hours)
    rows: list[dict] = []
    for r in triggers:
        rid = r.get("id")
        if not rid or rid not in label_by_id:
            continue
        ts = _parse_ts_any(r.get("ts") or r.get("timestamp"))
        if not ts or ts < start or ts > now_utc:
            continue
        p = r.get("score")
        try:
            p = float(p)
        except Exception:
            continue
        rows.append({"origin": r.get("origin") or "unknown", "y": label_by_id[rid], "p": p})
    return rows


def _seed_demo_rows() -> list[dict]:
    """Deterministic demo seeding for 2–3 origins."""
    import random
    random.seed(1337)
    out: list[dict] = []
    cfg = [
        ("reddit", 100, 0.95, 0.02),   # well-calibrated-ish
        ("twitter", 60, 0.70, 0.00),   # overconfident
        ("rss_news", 24, 0.90, 0.01),  # small-n decent
    ]
    for origin, n, slope, intercept in cfg:
        for _ in range(n):
            p = random.random() * 0.75 + 0.15  # 0.15..0.9
            pr = max(0.0, min(1.0, slope * p + intercept))
            y = 1 if random.random() < pr else 0
            out.append({"origin": origin, "y": y, "p": p})
    return out


def append(md: list[str], ctx: common.SummaryContext) -> None:
    """
    Appends a per-origin calibration block to `md` and writes artifacts:
      - models/calibration_per_origin.json
      - artifacts/cal_reliability_<origin>.png
    """
    hours = int(os.getenv("MW_CAL_WINDOW_H", "72"))

    artifacts_dir = _resolve_artifacts_dir(ctx)

    triggers, labels = _read_trigger_and_labels(ctx.logs_dir)
    rows = _window_rows(triggers, labels, now_utc=datetime.now(timezone.utc), hours=hours)

    demo_used = False
    if len(rows) < _LOW_N and ctx.is_demo:
        rows = _seed_demo_rows()
        demo_used = True

    # Group by origin
    grouped: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        g = grouped.setdefault(r["origin"], {"y": [], "p": []})
        g["y"].append(int(r["y"]))
        g["p"].append(float(r["p"]))

    # Compute metrics + plots
    results: list[dict] = []
    for origin, d in grouped.items():
        n = len(d["y"])
        if n == 0:
            continue
        ece, brier, bins = _compute_calibration(d["y"], d["p"], n_bins=_DEFAULT_BINS)
        low_n = n < _LOW_N
        high_ece = (ece is not None) and (ece > _HIGH_ECE)

        safe = _norm_origin(origin)
        png_path = artifacts_dir / f"cal_reliability_{safe}.png"
        _plot_reliability(bins, title=f"Reliability: {origin}", out_path=png_path)

        results.append(
            {
                "origin": origin,
                "n": n,
                "ece": round(float(ece), 3),
                "brier": round(float(brier), 3),
                "low_n": low_n,
                "high_ece": high_ece,
                "bins": [
                    {
                        "bin_low": float(b["low"]),
                        "bin_high": float(b["high"]),
                        "avg_conf": (None if b["avg_conf"] is None else float(b["avg_conf"])),
                        "empirical": (None if b["emp_rate"] is None else float(b["emp_rate"])),
                        "count": int(b["count"]),
                    }
                    for b in bins
                ],
                "artifact_png": str(png_path).replace("\\", "/"),
            }
        )

    # Sort and write JSON
    results.sort(key=lambda r: (-int(r["n"]), str(r["origin"]).lower()))
    payload = {
        "window_hours": hours,
        "generated_at": common._iso(datetime.now(timezone.utc)),
        "demo": bool(demo_used),
        "origins": results,
    }
    out_json = ctx.models_dir / "calibration_per_origin.json"
    common._write_json(out_json, payload)

    # Markdown
    md.append(f"🧮 Per-Origin Calibration ({hours}h)")
    if not results:
        md.append(" • (no labeled samples in window)")
        return
    for r in results:
        flags = []
        if r["low_n"]:
            flags.append("low_n")
        if r["high_ece"]:
            flags.append("high_ece")
        suffix = f" [{' | '.join(flags)}]" if flags else ""
        md.append(
            f" • {r['origin']:<7} → ECE={r['ece']:.2f} | Brier={r['brier']:.2f} | n={r['n']}{suffix}"
        )

    # Helpful: list artifact paths in the summary (eases CI upload debugging)
    if results:
        md.append("<details><summary>Per-origin calibration artifacts</summary>")
        for r in results:
            md.append(f" • {r['artifact_png']}")
        md.append("</details>")

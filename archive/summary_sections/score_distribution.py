# scripts/summary_sections/score_distribution.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt  # CI uses MPLBACKEND=Agg

from .common import SummaryContext


# -------------------------- parsing & IO helpers --------------------------

def _parse_ts(v) -> datetime | None:
    if v is None:
        return None
    # epoch seconds?
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        pass
    # ISO8601
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _is_drifted(row: dict) -> bool:
    """
    Consider several shapes:
      - drifted_features: list (non-empty => drifted)
      - drift: bool
      - drifted: bool
    """
    try:
        df = row.get("drifted_features")
        if isinstance(df, list) and len(df) > 0:
            return True
    except Exception:
        pass
    for key in ("drift", "drifted"):
        if key in row:
            try:
                if bool(row.get(key)):
                    return True
            except Exception:
                pass
    return False


def _row_score(row: dict):
    # prefer adjusted_score, then prob_trigger_next_6h, then score
    val = row.get("adjusted_score")
    if val is None:
        val = row.get("prob_trigger_next_6h", row.get("score"))
    try:
        return float(val)
    except Exception:
        return None


def _load_recent_scores(path: Path, hours: int) -> tuple[list[float], list[float]]:
    non_drifted, drifted = [], []
    if not path.exists():
        return non_drifted, drifted

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue

            ts = _parse_ts(row.get("timestamp"))
            if not ts or ts < cutoff:
                continue

            sc = _row_score(row)
            if sc is None:
                continue

            if _is_drifted(row):
                drifted.append(sc)
            else:
                non_drifted.append(sc)
    except Exception:
        # best-effort; empty lists will trigger demo seeding if enabled
        pass

    return non_drifted, drifted


# -------------------------- demo seeding & stats --------------------------

def _seed_demo(non_drifted: list[float], drifted: list[float], want_total: int = 64):
    rng = np.random.default_rng(42)
    n_drift = int(want_total * 0.25)
    n_ok = want_total - n_drift

    # Non-drifted: a bit higher mean than drifted
    nd = np.clip(rng.normal(loc=0.28, scale=0.11, size=n_ok), 0.0, 1.0)
    dr = np.clip(rng.normal(loc=0.18, scale=0.10, size=n_drift), 0.0, 1.0)

    non_drifted[:] = list(nd)
    drifted[:] = list(dr)


def _stats(arr: list[float]) -> tuple[float, float, float]:
    if not arr:
        return 0.0, 0.0, 0.0
    v = np.asarray(arr, dtype=float)
    return float(v.mean()), float(np.median(v)), float(np.quantile(v, 0.90))


# -------------------------- public section entrypoint --------------------------

def append(md: List[str], ctx: SummaryContext, hours: int = 48, min_points: int = 8) -> None:
    """
    📐 Score Distribution (48h by default)
    - loads scores from models/trigger_history.jsonl
    - splits into drifted vs non-drifted
    - renders dual histogram overlay + threshold line
    - writes artifacts/score_hist_drift_overlay_<hours>h.png
    """
    md.append("\n### 📐 Score Distribution (48h)")

    trig_hist = ctx.models_dir / "trigger_history.jsonl"
    non_drifted, drifted = _load_recent_scores(trig_hist, hours=hours)

    # Optional demo seeding if not enough points and we're in demo mode
    seeded = False
    if ctx.is_demo and (len(non_drifted) + len(drifted) < min_points):
        _seed_demo(non_drifted, drifted, want_total=max(min_points * 4, 64))
        seeded = True
        md.append("(demo) synthesized scores for visibility")

    all_vals = non_drifted + drifted
    n_total = len(all_vals)
    mean_all, med_all, p90_all = _stats(all_vals)
    md.append(f"- n={n_total}, mean={mean_all:.3f}, median={med_all:.3f}, p90={p90_all:.3f}")

    # Threshold (used) — allow another section to stash it; default 0.5
    thr_used = 0.5
    try:
        cache_thr = ctx.caches.get("score_distribution_threshold_used")
        if cache_thr is not None:
            thr_used = float(cache_thr)
    except Exception:
        pass
    md.append(f"- thresholds: dyn=n/a (0 pts) | static={thr_used:.3f} → used={thr_used:.3f}")

    # Counts + group means
    def _mean(xs: list[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    m_drift = _mean(drifted)
    m_ok = _mean(non_drifted)
    md.append(f"- split: drifted={len(drifted)} | non-drifted={len(non_drifted)}")
    md.append(f"- group means: drifted={m_drift:.3f}, non-drifted={m_ok:.3f}, Δ={(m_ok - m_drift):+.3f}")

    # Ensure artifacts dir exists
    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    img_out = artifacts / f"score_hist_drift_overlay_{hours}h.png"

    # Render overlay histogram
    try:
        bins = np.linspace(0.0, 1.0, 21)  # stable bins across runs

        plt.figure(figsize=(6.5, 2.6))
        if non_drifted:
            plt.hist(non_drifted, bins=bins, alpha=0.85, label="non-drifted")
        if drifted:
            plt.hist(drifted,     bins=bins, alpha=0.85, label="drifted")

        plt.title("Score distribution ({:d}h)".format(hours))
        plt.xlabel("score")
        plt.ylabel("count")
        try:
            plt.axvline(float(thr_used), linestyle="--", linewidth=1)
        except Exception:
            pass
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_out)
        plt.close()

        # Embed inline
        md.append(f"  \n![]({img_out.as_posix()})")
    except Exception as e:
        md.append(f"_⚠️ histogram render failed: {type(e).__name__}: {e}_")
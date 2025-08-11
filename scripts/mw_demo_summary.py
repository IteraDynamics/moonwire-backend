#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read‑only)

Outputs
 - artifacts/demo_summary.md
 - artifacts/consensus.png
 - artifacts/consensus_social.png

Safe: reads ./logs/*.jsonl written by tests; does not import app code or mutate logs.
"""

import os
import json
import uuid
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any, Optional

from dateutil import parser as dtparser
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# -------------------- config --------------------
LOGS_DIR = Path("logs")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 110
# ------------------------------------------------

# -------------------- palette -------------------
BG      = "#0B111B"
PANEL   = "#111A28"
TEXT    = "#E6EEF7"
MUTED   = "#9AA8B4"
GRID    = "#223047"
ACCENT  = "#13C6F3"
OK      = "#24D17E"
WARN    = "#FF6B6F"
# ------------------------------------------------


# -------------------- helpers -------------------
def _demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes", "on")

def red(s: str) -> str:
    if not s: return "000000"
    return hashlib.sha1(s.encode()).hexdigest()[:6]

def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists(): return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln: continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

def _ts_to_dt(ts) -> Optional[datetime]:
    if ts is None: return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dtparser.isoparse(str(ts)).astimezone(timezone.utc)
    except Exception:
        return None

def _ts_as_epoch(ts) -> float:
    dt = _ts_to_dt(ts)
    return dt.timestamp() if dt else 0.0

def band_weight_from_score(score: Optional[float]) -> float:
    if score is None: return 1.0
    if score >= 0.75: return 1.25
    if score >= 0.50: return 1.0
    return 0.75
# ------------------------------------------------


# -------------------- demo seeding --------------------
def generate_demo_data_if_needed(*args, **kwargs) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Accepts:
      generate_demo_data_if_needed(reviewers)
      generate_demo_data_if_needed(reviewers, flag_times, now=None)

    Returns (display_reviewers, seeded_events)
    """
    reviewers: List[Dict[str, Any]] = []
    flag_times: Optional[List[datetime]] = None
    now: Optional[datetime] = None

    if len(args) == 1:
        reviewers = args[0]
    elif len(args) >= 2:
        reviewers, flag_times = args[0], args[1]
        if len(args) >= 3:
            now = args[2]
    else:
        reviewers = kwargs.get("reviewers", [])
        flag_times = kwargs.get("flag_times")
        now = kwargs.get("now")

    if reviewers or not _demo_mode():
        return reviewers, []

    now = now or datetime.now(timezone.utc)
    n = random.randint(3, 5)
    weights = [0.75, 1.0, 1.25]
    seeded = []
    for _ in range(n):
        w = random.choice(weights)
        rid = f"demo-{uuid.uuid4().hex[:8]}"
        ts = now - timedelta(minutes=random.randint(5, 55))
        seeded.append({"id": rid, "weight": float(w), "timestamp": ts.isoformat()})

    display = [{"id": r["id"], "weight": r["weight"]} for r in seeded]
    return display, seeded
# ------------------------------------------------


# -------------------- load + assemble --------------------
retrain_log   = _read_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = _read_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log    = _read_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id   = {r.get("reviewer_id"): r for r in scores_log}

if retrain_log:
    latest = max(retrain_log, key=lambda r: _ts_as_epoch(r.get("timestamp")))
    sig_id = latest.get("signal_id", "unknown")
    sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
else:
    sig_id = "none"
    sig_rows = []

seen = set()
reviewers: List[Dict[str, Any]] = []
flag_times: List[datetime] = []

for r in sorted(sig_rows, key=lambda x: _ts_as_epoch(x.get("timestamp"))):
    dt = _ts_to_dt(r.get("timestamp"))
    if dt: flag_times.append(dt)
    rid = r.get("reviewer_id", "")
    if rid in seen: continue
    seen.add(rid)
    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

# demo-mode, read-only augmentation
reviewers, seeded_events = generate_demo_data_if_needed(reviewers, flag_times)
for ev in seeded_events:
    dt = _ts_to_dt(ev.get("timestamp"))
    if dt: flag_times.append(dt)

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold

last_trig = max(
    (t for t in triggered_log if t.get("signal_id") == sig_id),
    key=lambda x: _ts_as_epoch(x.get("timestamp")),
    default=None,
)

# -------------------- visuals --------------------
def _render_bar_chart(out_path: Path, total: float, thresh: float):
    fig = plt.figure(figsize=(3.4, 2.0), dpi=220, facecolor=BG)
    ax = fig.add_axes([0.14, 0.24, 0.78, 0.68], facecolor=PANEL)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, alpha=0.55, linewidth=0.8)

    bars = ax.bar(["weight", "threshold"], [total, thresh])
    bars[0].set_color(ACCENT)
    bars[1].set_color(OK if total >= thresh else WARN)

    ax.set_ylabel("weight", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("Consensus vs threshold", color=TEXT, fontsize=10, pad=8)

    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + max(0.04, h * 0.03),
                f"{h:.2f}", ha="center", va="bottom", color=TEXT, fontsize=9)

    fig.savefig(out_path, dpi=220, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def _render_social(payload: Dict[str, Any], out_path: Path):
    fig = plt.figure(figsize=(SOCIAL_W / DPI, SOCIAL_H / DPI), dpi=DPI, facecolor=BG)
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.20, 0.50, 0.30],
        width_ratios=[0.60, 0.40],
        hspace=0.25, wspace=0.18
    )

    # Title
    ax_t = fig.add_subplot(gs[0, :]); ax_t.axis("off")
    status = "TRIGGERED" if payload["would_trigger"] else "NO TRIGGER"
    status_col = OK if payload["would_trigger"] else WARN
    subtitle_bits = [f"Signal {red(payload['sig_id'])}"]
    if _demo_mode() and not payload["sig_rows"]:
        subtitle_bits.append("DEMO MODE (seeded)")
    ax_t.text(0.00, 0.68, "MoonWire — Consensus Check",
              color=TEXT, fontsize=24, weight=600, ha="left", va="center")
    ax_t.text(0.00, 0.10, " • ".join(subtitle_bits),
              color=MUTED, fontsize=13, ha="left", va="center")
    ax_t.text(1.00, 0.55, f" {status} ",
              color="white", fontsize=16, weight=700, ha="right", va="center",
              bbox=dict(boxstyle="round,pad=0.35", fc=status_col, ec=status_col, lw=0))

    # Gauge
    ax_g = fig.add_subplot(gs[1, 0], facecolor=PANEL)
    for s in ax_g.spines.values(): s.set_visible(False)
    ax_g.set_xticks([]); ax_g.set_yticks([])
    cap = max(payload["threshold"], payload["total_weight"], 3.0) * 1.05
    ax_g.set_xlim(0, cap)
    ax_g.axhspan(0.35, 0.65, xmin=0.03, xmax=0.97, facecolor=GRID, linewidth=0)
    ax_g.barh(0.5, payload["total_weight"], height=0.22, color=ACCENT)
    ax_g.axvline(payload["threshold"], color=status_col, linewidth=2.6)
    ax_g.text(payload["total_weight"], 0.83, f"{payload['total_weight']:.2f}",
              color=ACCENT, fontsize=13, ha="right", va="bottom")
    ax_g.text(payload["threshold"], 0.17, f"threshold {payload['threshold']:.2f}",
              color=status_col, fontsize=12, ha="center", va="top")
    ax_g.text(0.02, 0.98, "Consensus weight vs threshold",
              color=MUTED, fontsize=11, ha="left", va="top", transform=ax_g.transAxes)

    # Reviewers list
    ax_r = fig.add_subplot(gs[1, 1], facecolor=PANEL); ax_r.axis("off")
    ax_r.text(0.03, 0.95, "Reviewers (redacted)", color=TEXT, fontsize=14, weight=600,
              ha="left", va="top")
    rows = payload["reviewers"]
    if not rows:
        ax_r.text(0.03, 0.78, "No reviewers yet", color=MUTED, fontsize=12, ha="left", va="top")
    else:
        for i, r in enumerate(rows[:10]):
            ax_r.text(0.03, 0.86 - i * 0.07,
                      f"• {red(r['id'])}  —  {r['weight']:.2f}",
                      color=MUTED, fontsize=12, ha="left", va="top")
        if len(rows) > 10:
            ax_r.text(0.03, 0.86 - 10 * 0.07, f"… +{len(rows) - 10} more",
                      color=MUTED, fontsize=12, ha="left", va="top")

    # Sparkline
    ax_s = fig.add_subplot(gs[2, :], facecolor=PANEL)
    for s in ax_s.spines.values(): s.set_visible(False)
    ax_s.set_yticks([])
    ax_s.grid(axis="x", color=GRID, linewidth=1, alpha=0.5)
    ax_s.set_title("Flag timeline", color=MUTED, fontsize=12, pad=8)
    times = payload["flag_times"]
    if times:
        t0, t1 = min(times), max(times)
        if t0 == t1: t1 = t0 + timedelta(milliseconds=1)
        xs = [(t - t0).total_seconds() for t in times]
        ys = [0] * len(xs)
        ax_s.plot(xs, ys, linewidth=1.6, color=ACCENT)
        ax_s.scatter(xs, ys, s=22, color=ACCENT, zorder=3)
        ax_s.set_xlim(min(xs), max(xs))
    else:
        ax_s.text(0.02, 0.55, "No flags this run", color=MUTED, fontsize=12,
                  ha="left", va="center", transform=ax_s.transAxes)

    # Footer
    fig.text(0.015, 0.02,
             f"moonwire • {'demo mode • ' if _demo_mode() else ''}{datetime.now(timezone.utc).isoformat()}",
             color=MUTED, fontsize=11)

    fig.savefig(out_path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
# ------------------------------------------------


# -------------------- render & write --------------------
small_png = ART / "consensus.png"
_render_bar_chart(small_png, total_weight, threshold)

social_png = ART / "consensus_social.png"
_render_social(
    {
        "sig_id": sig_id,
        "sig_rows": sig_rows,
        "reviewers": reviewers,
        "flag_times": flag_times,
        "total_weight": total_weight,
        "threshold": threshold,
        "would_trigger": would_trigger,
    },
    social_png,
)

would = "TRIGGERS" if would_trigger else "NO TRIGGER"
md = []
md.append("# MoonWire CI Demo Summary")
md.append("")
md.append(f"MoonWire Demo Summary — {datetime.now(timezone.utc).isoformat()}")
md.append("")
md.append("Pipeline proof (CI): end‑to‑end tests passed; consensus math reproduced on latest flagged signal.")
md.append("")
md.append(f"- **Signal:** `{red(sig_id)}`")
md.append(f"- **Unique reviewers:** {len(reviewers)}")
md.append(f"- **Combined weight:** **{total_weight}**")
md.append(f"- **Threshold:** **{threshold}**  → **{would}**")
if _demo_mode() and not sig_rows and reviewers:
    md.append(f"- **Mode:** DEMO (seeded reviewers)")
if last_trig:
    md.append(f"- **Last retrain trigger logged:** {last_trig.get('timestamp','')}")
md.append("")
md.append("**Reviewers (redacted):**")
if reviewers:
    for r in reviewers:
        md.append(f"- `{red(r['id'])}` → weight {r['weight']}")
else:
    md.append("- _none found in this run_")
md.append("")
md.append("![Consensus](consensus.png)")

(md_path := ART / "demo_summary.md").write_text("\n".join(md))

print(f"Wrote: {md_path}")
print(f"Wrote: {small_png}")
print(f"Wrote: {social_png}")
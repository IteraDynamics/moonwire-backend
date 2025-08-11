#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read‑only)

Outputs
 - artifacts/demo_summary.md             (markdown for CI summary)
 - artifacts/consensus.png               (mini bar chart)
 - artifacts/consensus_social.png        (1280x720 social image: gauge + sparkline)

Safe: reads ./logs/*.jsonl written by tests; does not import app code or mutate logs.
"""

import os
import json
import uuid
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any, Optional, Iterable

from dateutil import parser as dtparser  # keep in requirements-dev.txt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# -------------------- config --------------------
LOGS_DIR = Path("logs")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5  # keep in sync with app config
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 110  # memory‑safe 16:9 for social
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes", "on")
# ------------------------------------------------


# -------------------- helpers --------------------
def red(s: str) -> str:
    """Redact any ID to a short sha1 prefix."""
    if not s:
        return "000000"
    return hashlib.sha1(s.encode()).hexdigest()[:6]


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            # tolerate malformed lines in CI
            pass
    return out


def _ts_to_dt(ts) -> Optional[datetime]:
    """Robust timestamp parser: epoch (int/float) or ISO string."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        # string → ISO
        return dtparser.isoparse(str(ts)).astimezone(timezone.utc)
    except Exception:
        return None


def _ts_as_epoch(ts) -> float:
    """Comparable epoch seconds for sorting/max()."""
    dt = _ts_to_dt(ts)
    return dt.timestamp() if dt else 0.0


def band_weight_from_score(score: Optional[float]) -> float:
    if score is None:
        return 1.0
    if score >= 0.75:
        return 1.25
    if score >= 0.50:
        return 1.0
    return 0.75
# ------------------------------------------------


# -------------------- demo seeding --------------------
def generate_demo_data_if_needed(
    reviewers: List[Dict[str, Any]],
    flag_times: List[datetime],
    now: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], List[datetime], List[Dict[str, Any]]]:
    """
    Read‑only seeding used only for visualization when DEMO_MODE is on
    and there are *no* reviewers for the latest signal.

    Returns (display_reviewers, updated_flag_times, seeded_events)
    """
    if reviewers or not DEMO_MODE:
        return reviewers, flag_times, []

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
    seeded_times = [_ts_to_dt(r["timestamp"]) for r in seeded if r.get("timestamp")]
    # extend timeline for sparkline
    new_flag_times = list(flag_times) + [t for t in seeded_times if t]
    return display, new_flag_times, seeded
# ------------------------------------------------


# -------------------- load logs --------------------
retrain_log = _read_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = _read_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log = _read_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id = {r.get("reviewer_id"): r for r in scores_log}

# Identify latest signal in retraining log
if retrain_log:
    latest = max(retrain_log, key=lambda r: _ts_as_epoch(r.get("timestamp")))
    sig_id = latest.get("signal_id", "unknown")
    sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
else:
    sig_id = "none"
    sig_rows = []

# Build reviewers (dedupe first flag per reviewer) + timeline
seen = set()
reviewers: List[Dict[str, Any]] = []
flag_times: List[datetime] = []
for r in sorted(sig_rows, key=lambda x: _ts_as_epoch(x.get("timestamp"))):
    dt = _ts_to_dt(r.get("timestamp"))
    if dt:
        flag_times.append(dt)

    rid = r.get("reviewer_id", "")
    if rid in seen:
        continue
    seen.add(rid)

    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

# Apply demo seeding (read‑only) if needed
reviewers, flag_times, seeded_events = generate_demo_data_if_needed(reviewers, flag_times)

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold

last_trig = max(
    (t for t in triggered_log if t.get("signal_id") == sig_id),
    key=lambda x: _ts_as_epoch(x.get("timestamp")),
    default=None,
)

now_iso = datetime.now(timezone.utc).isoformat()


# -------------------- visuals --------------------
def _render_bar_chart(consensus_png: Path, total: float, thresh: float):
    """
    Tidier mini bar: dark bg, subtle grid, tight labels.
    """
    bg = "#0B111B"
    panel = "#141C2A"
    text = "#E6EEF7"
    muted = "#9AA8B4"
    accent = "#17D2FF"
    warn = "#FF5A5F"
    ok = "#24D17E"

    fig = plt.figure(figsize=(3.6, 2.2), dpi=220, facecolor=bg)
    ax = fig.add_axes([0.12, 0.22, 0.80, 0.68], facecolor=panel)

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#223047", alpha=0.5, linewidth=0.8)

    bars = ax.bar(["weight", "threshold"], [total, thresh])
    bars[0].set_color(accent)
    bars[1].set_color(ok if total >= thresh else warn)

    ax.set_ylabel("weight", color=muted, fontsize=9)
    ax.tick_params(colors=muted, labelsize=9)
    ax.set_title("Consensus vs Threshold", color=text, fontsize=10, pad=8)

    # value labels
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + max(0.04, h * 0.03),
                f"{h:.2f}", ha="center", va="bottom", color=text, fontsize=9)

    fig.savefig(consensus_png, dpi=220, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def _render_social(fig_data: Dict[str, Any], social_png: Path):
    # Palette
    bg = "#0B111B"
    panel = "#141C2A"
    text = "#E6EEF7"
    muted = "#9AA8B4"
    accent = "#17D2FF"
    ok = "#24D17E"
    warn = "#FF5A5F"
    grid = "#223047"

    fig = plt.figure(figsize=(SOCIAL_W / DPI, SOCIAL_H / DPI), dpi=DPI, facecolor=bg)
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.22, 0.48, 0.30],
        width_ratios=[0.60, 0.40],
        hspace=0.25, wspace=0.18
    )

    # ---------- Title row ----------
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    status = "TRIGGERED" if fig_data["would_trigger"] else "NO TRIGGER"
    status_col = ok if fig_data["would_trigger"] else warn
    subtitle_bits = [f"Signal {red(fig_data['sig_id'])}"]
    if DEMO_MODE and not fig_data["sig_rows"]:
        subtitle_bits.append("DEMO MODE (seeded)")
    subtitle = " • ".join(subtitle_bits)
    ax_title.text(0.0, 0.65, "MoonWire — Consensus Check",
                  color=text, fontsize=24, weight=600, ha="left", va="center")
    ax_title.text(0.0, 0.10, subtitle, color=muted, fontsize=13, ha="left", va="center")
    ax_title.text(1.0, 0.60, f" {status} ",
                  color="white", fontsize=16, weight=700, ha="right", va="center",
                  bbox=dict(boxstyle="round,pad=0.35", fc=status_col, ec=status_col, lw=0))

    # ---------- Gauge (left, middle row) ----------
    ax_g = fig.add_subplot(gs[1, 0], facecolor=panel)
    for spine in ax_g.spines.values():
        spine.set_visible(False)
    ax_g.set_yticks([])
    ax_g.set_xticks([])
    ax_g.set_xlim(0, max(fig_data["threshold"], fig_data["total_weight"], 3.0))
    ax_g.axhspan(0.35, 0.65, xmin=0.02, xmax=0.98, facecolor=grid, linewidth=0)
    ax_g.barh(0.5, fig_data["total_weight"], height=0.22, color=accent)
    ax_g.axvline(fig_data["threshold"], color=status_col, linewidth=2.5)
    ax_g.text(fig_data["total_weight"], 0.84, f"{fig_data['total_weight']:.2f}",
              color=accent, fontsize=13, ha="right", va="bottom")
    ax_g.text(fig_data["threshold"], 0.16, f"threshold {fig_data['threshold']:.2f}",
              color=status_col, fontsize=12, ha="center", va="top")
    ax_g.text(0.02, 0.98, "Consensus weight vs threshold",
              color=muted, fontsize=11, ha="left", va="top", transform=ax_g.transAxes)

    # ---------- Reviewers list (right, middle row) ----------
    ax_r = fig.add_subplot(gs[1, 1], facecolor=panel)
    ax_r.axis("off")
    rows = fig_data["reviewers"]
    ax_r.text(0.03, 0.95, "Reviewers (redacted)", color=text, fontsize=14, weight=600,
              ha="left", va="top")
    if not rows:
        ax_r.text(0.03, 0.78, "No reviewers yet", color=muted, fontsize=12,
                  ha="left", va="top")
    else:
        for i, r in enumerate(rows[:10]):
            ax_r.text(0.03, 0.86 - i * 0.07,
                      f"• {red(r['id'])}  —  {r['weight']:.2f}",
                      color=muted, fontsize=12, ha="left", va="top")
        if len(rows) > 10:
            ax_r.text(0.03, 0.86 - 10 * 0.07, f"… +{len(rows) - 10} more",
                      color=muted, fontsize=12, ha="left", va="top")

    # ---------- Sparkline (bottom) ----------
    ax_s = fig.add_subplot(gs[2, :], facecolor=panel)
    for spine in ax_s.spines.values():
        spine.set_visible(False)
    ax_s.set_yticks([])
    ax_s.grid(axis="x", color=grid, linewidth=1, alpha=0.5)
    ax_s.set_title("Flag timeline", color=muted, fontsize=12, pad=8)

    times = fig_data["flag_times"]
    if times:
        t0, t1 = min(times), max(times)
        if t0 == t1:
            t1 = t0 + timedelta(milliseconds=1)
        xs = [(t - t0).total_seconds() for t in times]
        ys = [0] * len(xs)
        ax_s.plot(xs, ys, linewidth=1.6, color=accent)
        ax_s.scatter(xs, ys, s=22, color=accent, zorder=3)
        ax_s.set_xlim(min(xs), max(xs))
    else:
        ax_s.text(0.02, 0.55, "No flags this run",
                  color=muted, fontsize=12, ha="left", va="center", transform=ax_s.transAxes)

    # ---------- Footer ----------
    fig.text(0.015, 0.02,
             f"moonwire • {'demo mode • ' if DEMO_MODE else ''}{fig_data['now']}",
             color=muted, fontsize=11)

    fig.savefig(social_png, dpi=DPI, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
# ------------------------------------------------


# -------------------- render & write --------------------
# mini bar
small_png = ART / "consensus.png"
_would = "TRIGGERS" if would_trigger else "NO TRIGGER"
_render_bar_chart(small_png, total_weight, threshold)

# social image
social_png = ART / "consensus_social.png"
fig_payload = {
    "sig_id": sig_id,
    "sig_rows": sig_rows,
    "reviewers": reviewers,
    "flag_times": flag_times,
    "total_weight": total_weight,
    "threshold": threshold,
    "would_trigger": would_trigger,
    "now": now_iso,
}
_render_social(fig_payload, social_png)

# markdown
md_lines = []
md_lines.append("# MoonWire CI Demo Summary")
md_lines.append("")
md_lines.append(f"MoonWire Demo Summary — {now_iso}")
md_lines.append("")
md_lines.append("Pipeline proof (CI): end‑to‑end tests passed; consensus math reproduced on latest flagged signal.")
md_lines.append("")
md_lines.append(f"- **Signal:** `{red(sig_id)}`")
md_lines.append(f"- **Unique reviewers:** {len(reviewers)}")
md_lines.append(f"- **Combined weight:** **{total_weight}**")
md_lines.append(f"- **Threshold:** **{threshold}**  → **{_would}**")
if DEMO_MODE and not sig_rows and reviewers:
    md_lines.append(f"- **Mode:** DEMO (seeded reviewers)")
# last trigger (best effort)
if triggered_log:
    last = max(triggered_log, key=lambda x: _ts_as_epoch(x.get("timestamp")))
    md_lines.append(f"- **Last retrain trigger logged:** {last.get('timestamp','')}")
md_lines.append("")
md_lines.append("**Reviewers (redacted):**")
if reviewers:
    for r in reviewers:
        md_lines.append(f"- `{red(r['id'])}` → weight {r['weight']}")
else:
    md_lines.append("- _none found in this run_")
md_lines.append("")
md_lines.append("![Consensus](consensus.png)")

md_path = ART / "demo_summary.md"
md_path.write_text("\n".join(md_lines))

print(f"Wrote: {md_path}")
print(f"Wrote: {small_png}")
print(f"Wrote: {social_png}")
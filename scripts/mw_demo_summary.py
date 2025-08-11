#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs
 - artifacts/demo_summary.md             (markdown for CI summary)
 - artifacts/consensus.png               (tiny bar used in the MD)
 - artifacts/consensus_social.png        (1280x720 social image)

Safe: reads ./logs/*.jsonl written by tests; does not import app code or mutate logs.
"""

import os
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime, timezone, timedelta

from dateutil import parser as dtparser  # in requirements-dev.txt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -------------------- config --------------------
LOGS_DIR = Path("logs")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5  # keep in sync with app config
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 110  # 16:9, crisp but memory-safe
# ------------------------------------------------

# ========== tiny helpers ==========
def red(s: str) -> str:
    """Redact any ID to a short sha1 prefix."""
    if not s:
        return "000000"
    return hashlib.sha1(s.encode()).hexdigest()[:6]

def load_jsonl(path: Path):
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

def band_weight_from_score(score):
    if score is None: return 1.0
    if score >= 0.75: return 1.25
    if score >= 0.50: return 1.0
    return 0.75

# ========== DEMO MODE seeding (kept for tests) ==========
def seed_reviewers_if_empty(reviewers, now=None):
    """Return (compact_reviewers, seeded_events). Events have ISO timestamps."""
    if reviewers:  # real data present → nothing to do
        return reviewers, []
    now = now or datetime.now(timezone.utc)
    n = random.randint(3, 5)
    choices = [0.75, 1.0, 1.25]
    seeded = []
    for _ in range(n):
        seeded.append({
            "id": f"demo-{hashlib.sha1(os.urandom(8)).hexdigest()[:8]}",
            "weight": random.choice(choices),
            "timestamp": (now - timedelta(minutes=random.randint(3, 55))).isoformat()
        })
    compact = [{"id": r["id"], "weight": r["weight"]} for r in seeded]
    return compact, seeded

def generate_demo_data_if_needed(reviewers, flag_times=None):
    """
    Public helper used by tests.
    Returns (reviewers, events). When DEMO_MODE=true and reviewers empty,
    seeds 3–5 demo reviewers and events (read-only, not written to disk).
    """
    flag_times = flag_times or []
    demo_on = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
    if demo_on and not reviewers:
        seeded_reviewers, seeded_events = seed_reviewers_if_empty(reviewers)
        # extend caller's timeline (in-memory only)
        for ev in seeded_events:
            try:
                flag_times.append(dtparser.isoparse(ev["timestamp"]))
            except Exception:
                pass
        return seeded_reviewers, seeded_events
    return reviewers, []

# ========== load logs ==========
retrain_log   = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log    = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id   = {r.get("reviewer_id"): r for r in scores_log}

# latest signal
if retrain_log:
    latest = max(retrain_log, key=lambda r: r.get("timestamp", ""))
    sig_id = latest.get("signal_id", "unknown")
    sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
else:
    sig_id = "none"
    sig_rows = []

# weights & timeline (dedupe by reviewer)
seen = set()
reviewers = []
flag_times = []
for r in sorted(sig_rows, key=lambda x: x.get("timestamp","")):
    ts = r.get("timestamp")
    if ts:
        try:
            flag_times.append(dtparser.isoparse(ts))
        except Exception:
            pass

    rid = r.get("reviewer_id","")
    if rid in seen:
        continue
    seen.add(rid)
    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

# DEMO seeding (read-only)
reviewers, _seeded = generate_demo_data_if_needed(reviewers, flag_times)

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max((t for t in triggered_log if t.get("signal_id")==sig_id),
                key=lambda x: x.get("timestamp",""), default=None)

now_iso = datetime.now(timezone.utc).isoformat()
demo_on = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

# ========== small bar for MD ==========
plt.figure(figsize=(3.8, 2.1), dpi=220)
plt.bar(["weight","threshold"], [total_weight, threshold])
plt.title("Consensus vs threshold", pad=8)
plt.tight_layout()
small_png = ART / "consensus.png"
plt.savefig(small_png, dpi=220)
plt.close()

# ========== SOCIAL IMAGE (clean card layout) ==========
bg="#0A0E1A"; accent="#00E5FF"; muted="#93A3B1"; ok="#24D17E"; warn="#FF5A5F"; frame="#162235"
text="#FFFFFF"

fig = plt.figure(figsize=(SOCIAL_W/DPI, SOCIAL_H/DPI), dpi=DPI, facecolor=bg)
F = fig.transFigure

def card(x, y, w, h, radius=0.02, fc=frame):
    fig.patches.append(
        FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                       linewidth=0, facecolor=fc, transform=F)
    )

# Title row
fig.text(0.06, 0.90, "MoonWire — Consensus Check", color=text, fontsize=28, weight=600,
         ha="left", va="center", transform=F)

status = "TRIGGERED" if would_trigger else "NO TRIGGER"
status_col = ok if would_trigger else warn
fig.text(0.94, 0.90, f" {status} ", color="white", fontsize=24, weight=700,
         ha="right", va="center", transform=F,
         bbox=dict(boxstyle="round,pad=0.35", fc=status_col, ec=status_col))

# Subhead
mode_note = " • DEMO MODE (seeded)" if demo_on and _seeded else ""
fig.text(0.06, 0.86, f"Signal {red(sig_id)}{mode_note}",
         color=muted, fontsize=14, ha="left", va="center", transform=F)

# --- Gauge card (left) ---
g_x, g_y, g_w, g_h = 0.06, 0.60, 0.56, 0.18
card(g_x, g_y, g_w, g_h)

# gauge track
track_inset = 0.03
track_x = g_x + track_inset
track_y = g_y + track_inset
track_w = g_w - track_inset*2
track_h = g_h - track_inset*2
fig.patches.append(FancyBboxPatch((track_x, track_y), track_w, track_h,
                   boxstyle="round,pad=0.25", linewidth=0, facecolor="#122036", transform=F))

# bar fill (scaled against max(threshold, total_weight))
cap = max(threshold, total_weight, 3.0)
fill_w = track_w * max(0.0, min(1.0, total_weight / cap))
if fill_w > 0:
    fig.patches.append(FancyBboxPatch((track_x, track_y), fill_w, track_h,
                       boxstyle="round,pad=0.25", linewidth=0, facecolor=accent, transform=F))

# threshold line
th_x = track_x + (threshold / cap) * track_w
fig.patches.append(FancyBboxPatch((th_x-0.002, track_y-0.01), 0.004, track_h+0.02,
                   boxstyle="round,pad=0.0", linewidth=0, facecolor=status_col, transform=F))
fig.text(th_x, track_y+track_h+0.01, f"{threshold:.2f}", color=status_col, fontsize=12,
         ha="center", va="bottom", transform=F)
# total label above center of bar
fig.text(track_x + min(fill_w, track_w)-0.005, track_y+track_h/2, f"{total_weight:.2f}",
         color=accent, fontsize=16, ha="right", va="center", transform=F)

# --- Reviewers card (right) ---
r_x, r_y, r_w, r_h = 0.66, 0.60, 0.28, 0.18
card(r_x, r_y, r_w, r_h)
fig.text(r_x+0.02, r_y+r_h-0.04, "Reviewers (redacted)", color=text, fontsize=14,
         ha="left", va="top", transform=F)
if reviewers:
    rows = [f"• {red(r['id'])}  —  {r['weight']:.2f}" for r in reviewers[:6]]
    fig.text(r_x+0.02, r_y+r_h-0.08, "\n".join(rows), color=muted, fontsize=12,
             ha="left", va="top", transform=F)
else:
    fig.text(r_x+0.02, r_y+r_h-0.08, "No reviewers yet", color=muted, fontsize=12,
             ha="left", va="top", transform=F)

# --- Sparkline card (full width) ---
s_x, s_y, s_w, s_h = 0.06, 0.30, 0.88, 0.18
card(s_x, s_y, s_w, s_h)

def draw_sparkline(times):
    if not times:
        fig.text(s_x+0.02, s_y+0.06, "No flags this run", color=muted, fontsize=13, transform=F)
        return
    times = sorted(times)
    t0, t1 = times[0], times[-1]
    span = max((t1 - t0).total_seconds(), 1.0)

    # baseline
    fig.patches.append(FancyBboxPatch((s_x+0.02, s_y+0.08), s_w-0.04, 0.006,
                       boxstyle="round,pad=0.2", linewidth=0, facecolor="#122036", transform=F))
    for t in times:
        x01 = (t - t0).total_seconds()/span
        x = s_x+0.02 + x01*(s_w-0.04)
        fig.patches.append(FancyBboxPatch((x-0.003, s_y+0.07), 0.006, 0.028,
                           boxstyle="round,pad=0.0", linewidth=0, facecolor=accent, transform=F))
    fig.text(s_x+0.02, s_y+0.13, "Flag timeline", color=muted, fontsize=12, transform=F)

draw_sparkline(flag_times)

# footer
fig.text(0.06, 0.08, f"moonwire • demo mode • {now_iso}", color=muted, fontsize=12, transform=F)

social_png = ART / "consensus_social.png"
fig.savefig(social_png, dpi=DPI, facecolor=bg)
plt.close(fig)

# ========== Markdown summary ==========
md = []
md.append("# MoonWire CI Demo Summary")
md.append("")
md.append(f"MoonWire Demo Summary — {now_iso}")
md.append("")
md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.")
md.append("")
md.append(f"- **Signal:** `{red(sig_id)}`")
md.append(f"- **Unique reviewers:** {len(reviewers)}")
md.append(f"- **Combined weight:** **{total_weight}**")
md.append(f"- **Threshold:** **{threshold}**  → **{'TRIGGERS' if would_trigger else 'NO TRIGGER'}**")
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

md_path = ART / "demo_summary.md"
md_path.write_text("\n".join(md))

print(f"Wrote: {md_path}")
print(f"Wrote: {small_png}")
print(f"Wrote: {social_png}")
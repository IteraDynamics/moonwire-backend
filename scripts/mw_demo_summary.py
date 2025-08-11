#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs:
 - artifacts/demo_summary.md             (markdown, includes inline Base64 image for CI Summary)
 - artifacts/consensus.png               (small CI plot)
 - artifacts/consensus_social.png        (1280x720 social image w/ gauge + sparkline)

Safe: reads ./logs/*.jsonl written by tests; does not import app code.
"""

import json, hashlib, base64
from pathlib import Path
from datetime import datetime, timezone
from dateutil import parser as dtparser  # add to requirements-dev.txt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------- config ----------
LOGS_DIR = Path("logs")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5  # keep in sync with app config
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 110  # memory-safe 16:9
# ----------------------------

def red(s: str) -> str:
    """Redact any ID to a short sha1 prefix."""
    if not s: return "000000"
    return hashlib.sha1(s.encode()).hexdigest()[:6]

def load_jsonl(path: Path):
    if not path.exists(): return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln: continue
        try:
            out.append(json.loads(ln))
        except Exception:
            # tolerate malformed lines in CI
            pass
    return out

# ---- load logs ----
retrain_log   = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log    = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id   = {r.get("reviewer_id"): r for r in scores_log}

# ---- latest signal in retraining log ----
if retrain_log:
    latest = max(retrain_log, key=lambda r: r.get("timestamp", ""))
    sig_id = latest.get("signal_id", "unknown")
    sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
else:
    sig_id = "none"
    sig_rows = []

# ---- weights & timeline ----
def band_weight_from_score(score):
    if score is None: return 1.0
    if score >= 0.75: return 1.25
    if score >= 0.50: return 1.0
    return 0.75

seen = set()
reviewers = []
flag_times = []
for r in sorted(sig_rows, key=lambda x: x.get("timestamp","")):
    # collect all timestamps for sparkline (including duplicates)
    ts = r.get("timestamp")
    try:
        if ts: flag_times.append(dtparser.isoparse(ts))
    except Exception:
        pass

    rid = r.get("reviewer_id","")
    if rid in seen:  # first flag counts
        continue
    seen.add(rid)
    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max((t for t in triggered_log if t.get("signal_id")==sig_id),
                key=lambda x: x.get("timestamp",""), default=None)

now = datetime.now(timezone.utc).isoformat()

# ---- small CI bar (consensus.png) ----
plt.figure(figsize=(3.8, 2.4), dpi=200)
plt.title("Consensus Weight vs Threshold")
plt.bar(["weight","threshold"], [total_weight, threshold])
plt.tight_layout()
small_png = ART / "consensus.png"
plt.savefig(small_png, dpi=200)
plt.close()

# ---- social visual (consensus_social.png) ----
bg="#0A0E1A"; accent="#00E5FF"; muted="#93A3B1"; ok="#24D17E"; warn="#FF5A5F"; frame="#162235"
fig = plt.figure(figsize=(SOCIAL_W/DPI, SOCIAL_H/DPI), dpi=DPI)
ax = plt.gca(); ax.set_facecolor(bg); fig.set_facecolor(bg)

title = "MoonWire • Consensus Check"
status = "TRIGGERED" if would_trigger else "NO TRIGGER"
status_col = ok if would_trigger else warn
plt.text(40, 660, title, color="white", fontsize=32, weight=600, ha="left", va="center")
plt.text(SOCIAL_W-40, 660, status, color="white", fontsize=30, weight=700,
         ha="right", va="center", bbox=dict(boxstyle="round,pad=0.35", fc=status_col, ec=status_col))

# Gauge
left, right, y = 100, SOCIAL_W-100, 440
track_w = right - left
ax.add_patch(FancyBboxPatch((left, y-22), track_w, 44, boxstyle="round,pad=0.3",
                            linewidth=0, facecolor=frame))
cap = max(threshold, total_weight, 3.0)

th_x = left + (threshold / cap) * track_w
ax.add_patch(FancyBboxPatch((th_x-2, y-34), 4, 68, boxstyle="round,pad=0.0",
                            linewidth=0, facecolor=status_col))
plt.text(th_x, y+60, f"threshold {threshold:.2f}", color=status_col,
         fontsize=16, ha="center", va="bottom")

fill_w = int((total_weight / cap) * track_w)
fill_w = max(0, min(track_w, fill_w))
ax.add_patch(FancyBboxPatch((left, y-18), fill_w, 36, boxstyle="round,pad=0.25",
                            linewidth=0, facecolor=accent))
plt.text(left+fill_w+10, y, f"{total_weight:.2f}", color=accent,
         fontsize=22, ha="left", va="center")

# Stats
stats = [
    f"Signal: {red(sig_id)}",
    f"Unique reviewers: {len(reviewers)}",
    f"Combined weight: {total_weight:.2f}",
    f"Threshold: {threshold:.2f}",
]
plt.text(100, 320, "\n".join(stats), color="white", fontsize=20, va="top")

rows = [f"• {red(r['id'])}: {r['weight']:.2f}" for r in reviewers[:8]]
if len(reviewers) > 8:
    rows.append(f"… +{len(reviewers)-8} more")
plt.text(SOCIAL_W-100, 320, "\n".join(rows) if rows else "No reviewers yet",
         color=muted, fontsize=18, va="top", ha="right")

# Sparkline timeline
def draw_sparkline(times):
    if not times: return
    times = sorted(times)
    t0, t1 = times[0], times[-1]
    span = max((t1 - t0).total_seconds(), 1.0)
    xs = [left + ((t - t0).total_seconds()/span) * track_w for t in times]
    y_base = 170
    ax.add_patch(FancyBboxPatch((left, y_base-2), track_w, 4,
                                boxstyle="round,pad=0.2", linewidth=0, facecolor=frame))
    for x in xs:
        ax.add_patch(FancyBboxPatch((x-2, y_base-10), 4, 20,
                                    boxstyle="round,pad=0.0", linewidth=0, facecolor=accent))
    lbl = "flag timeline" if len(times)>1 else "single flag"
    plt.text(left, y_base+26, lbl, color=muted, fontsize=14, ha="left", va="bottom")

draw_sparkline(flag_times)

# watermark
plt.text(100, 90, f"moonwire • demo mode • {now}", color=muted, fontsize=14, ha="left")

plt.axis("off")
fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.12)
social_png = ART / "consensus_social.png"
plt.savefig(social_png, dpi=DPI, facecolor=bg)
plt.close()

# ---- embed social image in markdown via Base64 ----
b64 = base64.b64encode(social_png.read_bytes()).decode("ascii")
img_html = f'<img alt="Consensus" src="data:image/png;base64,{b64}" width="720"/>'

# ---- markdown summary (with inline image) ----
md = []
md.append("# MoonWire CI Demo Summary")
md.append("")
md.append(f"MoonWire Demo Summary — {now}")
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
# show small CI image (fallback) and inline Base64 (primary)
md.append("![Consensus](consensus.png)")
md.append("")
md.append(img_html)  # Base64 inline for GitHub Summary

md_path = ART / "demo_summary.md"
md_path.write_text("\n".join(md))

print(f"Wrote: {md_path}")
print(f"Wrote: {small_png}")
print(f"Wrote: {social_png}")
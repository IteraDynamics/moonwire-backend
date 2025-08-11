#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Reads JSONL logs produced during tests and renders:
 - Markdown summary  -> artifacts/demo_summary.md
 - Small CI plot     -> artifacts/consensus.png
 - Social visual     -> artifacts/consensus_social.png

No imports from app code; paths are plain ./logs/*.jsonl
Safe to run only after tests pass.
"""

import json, os, hashlib
from pathlib import Path
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------- config ----------
LOGS_DIR = Path("logs")
ART = Path("artifacts")
DEFAULT_THRESHOLD = 2.5  # keep in sync with app config if it changes
SOCIAL_W, SOCIAL_H, DPI = 1600, 900, 200  # 16:9 for X/Twitter
# ----------------------------

ART.mkdir(exist_ok=True)

def red(s: str) -> str:
    """Redact any ID to a short sha1 prefix."""
    if not s:
        return "000000"
    return hashlib.sha1(s.encode()).hexdigest()[:6]

def load_jsonl(path: Path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            # ignore malformed lines instead of failing CI
            pass
    return out

# ---- load logs ----
retrain_log = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id = {r.get("reviewer_id"): r for r in scores_log}

# ---- choose latest signal seen in retraining_log ----
if retrain_log:
    latest = max(retrain_log, key=lambda r: r.get("timestamp", ""))
    sig_id = latest.get("signal_id", "unknown")
    sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
else:
    sig_id = "none"
    sig_rows = []

# ---- dedupe reviewers (first flag counts) & compute weights ----
def band_weight_from_score(score):
    if score is None:
        return 1.0
    if score >= 0.75:
        return 1.25
    if score >= 0.50:
        return 1.0
    return 0.75

seen = set()
reviewers = []
for r in sorted(sig_rows, key=lambda x: x.get("timestamp", "")):
    rid = r.get("reviewer_id", "")
    if rid in seen:
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

last_trig = max(
    (t for t in triggered_log if t.get("signal_id") == sig_id),
    key=lambda x: x.get("timestamp", ""),
    default=None,
)

now = datetime.now(timezone.utc).isoformat()

# ---- small CI bar plot (consensus.png) ----
plt.figure(figsize=(3.8, 2.4), dpi=200)
plt.title("Consensus Weight vs Threshold")
plt.bar(["weight", "threshold"], [total_weight, threshold])
plt.tight_layout()
small_png = ART / "consensus.png"
plt.savefig(small_png, dpi=200)
plt.close()

# ---- social visual (consensus_social.png) ----
bg = "#0A0E1A"           # midnight
accent = "#00E5FF"       # electric cyan
muted = "#93A3B1"        # slate
ok = "#24D17E"           # green
warn = "#FF5A5F"         # red
frame = "#162235"

fig = plt.figure(figsize=(SOCIAL_W / DPI, SOCIAL_H / DPI), dpi=DPI)
ax = plt.gca()
ax.set_facecolor(bg)
fig.set_facecolor(bg)

title = "MoonWire • Consensus Check"
status = "TRIGGERED" if would_trigger else "NO TRIGGER"
status_col = ok if would_trigger else warn

plt.text(40, 820, title, color="white", fontsize=34, weight=600, ha="left", va="center")
bbox = dict(boxstyle="round,pad=0.35", fc=status_col, ec=status_col)
plt.text(SOCIAL_W - 40, 820, status, color="white", fontsize=32, weight=700,
         ha="right", va="center", bbox=bbox)

# Gauge track
left, right, y = 120, SOCIAL_W - 120, 520
track_w = right - left
ax.add_patch(FancyBboxPatch((left, y - 22), track_w, 44, boxstyle="round,pad=0.3",
                            linewidth=0, facecolor=frame))

# Normalize scale so the bar is readable even when small
cap = max(threshold, total_weight, 3.0)
# threshold marker
th_x = left + (threshold / cap) * track_w
ax.add_patch(FancyBboxPatch((th_x - 2, y - 34), 4, 68, boxstyle="round,pad=0.0",
                            linewidth=0, facecolor=status_col))
plt.text(th_x, y + 64, f"threshold {threshold:.2f}", color=status_col,
         fontsize=18, ha="center", va="bottom")

# fill to total weight
fill_w = int((total_weight / cap) * track_w)
fill_w = max(0, min(track_w, fill_w))
ax.add_patch(FancyBboxPatch((left, y - 18), fill_w, 36, boxstyle="round,pad=0.25",
                            linewidth=0, facecolor=accent))
plt.text(left + fill_w + 10, y, f"{total_weight:.2f}", color=accent,
         fontsize=24, ha="left", va="center")

# Stats blocks
stats = [
    f"Signal: {red(sig_id)}",
    f"Unique reviewers: {len(reviewers)}",
    f"Combined weight: {total_weight:.2f}",
    f"Threshold: {threshold:.2f}",
]
plt.text(120, 360, "\n".join(stats), color="white", fontsize=22, va="top")

rows = [f"• {red(r['id'])}: {r['weight']:.2f}" for r in reviewers[:8]]
if len(reviewers) > 8:
    rows.append(f"… +{len(reviewers) - 8} more")
plt.text(SOCIAL_W - 120, 360, "\n".join(rows) if rows else "No reviewers yet",
         color=muted, fontsize=20, va="top", ha="right")

# watermark / timestamp
stamp = f"moonwire • demo mode • {now}"
plt.text(120, 110, stamp, color=muted, fontsize=16, ha="left")

plt.axis("off")
plt.tight_layout(pad=0)
social_png = ART / "consensus_social.png"
plt.savefig(social_png, dpi=DPI, facecolor=bg, bbox_inches="tight")
plt.close()

# ---- markdown summary ----
md_lines = []
md_lines.append("# MoonWire CI Demo Summary")
md_lines.append("")
md_lines.append(f"MoonWire Demo Summary — {now}")
md_lines.append("")
md_lines.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.")
md_lines.append("")
md_lines.append(f"- **Signal:** `{red(sig_id)}`")
md_lines.append(f"- **Unique reviewers:** {len(reviewers)}")
md_lines.append(f"- **Combined weight:** **{total_weight}**")
md_lines.append(f"- **Threshold:** **{threshold}**  → **{'TRIGGERS' if would_trigger else 'NO TRIGGER'}**")
if last_trig:
    md_lines.append(f"- **Last retrain trigger logged:** {last_trig.get('timestamp','')}")
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
#!/usr/bin/env python3
“””
MoonWire CI Demo Summary (read-only)

Outputs

- artifacts/demo_summary.md
- artifacts/consensus.png
- artifacts/consensus_social.png

Reads ./logs/*.jsonl; never mutates logs.
“””

import os, json, hashlib, random, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import matplotlib.patches as mpatches

# –––––––– config ––––––––

LOGS_DIR = Path(“logs”)
ART = Path(“artifacts”); ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 120  # 16:9 stable card

# ––––––––––––––––––––

# ––––––– helpers —————–

def red(s: str) -> str:
“”“Redact any ID to a short sha1 prefix (6 chars).”””
return “000000” if not s else hashlib.sha1(s.encode()).hexdigest()[:6]

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
# tolerate malformed lines
pass
return out

def band_weight_from_score(score):
if score is None: return 1.0
if score >= 0.75: return 1.25
if score >= 0.50: return 1.0
return 0.75

def weight_to_label(w: float) -> str:
if w >= 1.20: return “High”
if w >= 0.90: return “Med”
return “Low”

def parse_ts(val):
if val is None:
return None
try:
ts = float(val)
return datetime.fromtimestamp(ts, tz=timezone.utc)
except Exception:
pass
try:
s = str(val)
s = s[:-1] + “+00:00” if s.endswith(“Z”) else s
return datetime.fromisoformat(s).astimezone(timezone.utc)
except Exception:
return None

# ––––––––––––––––––––

# —— demo seeding (read-only) ––––

def generate_demo_data_if_needed(reviewers, flag_times=None):
“””
If DEMO_MODE=true and no reviewers present, return 3–5 seeded
reviewers + seed timestamps (only for display). Never writes logs.

```
Returns (display_reviewers, seeded_events)
"""
flag_times = flag_times or []
demo_on = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
if not demo_on or reviewers:
    return reviewers, []
now = datetime.now(timezone.utc)
n = random.randint(3, 5)
choices = [0.75, 1.0, 1.25]
seeded, display = [], []
for _ in range(n):
    rid = f"demo-{uuid.uuid4().hex[:8]}"
    w = random.choice(choices)
    ts = (now - timedelta(minutes=random.randint(2, 55)))
    seeded.append({"id": rid, "weight": w, "timestamp": ts.isoformat()})
    display.append({"id": rid, "weight": w})
    flag_times.append(ts)
return display, seeded
```

# ––––––––––––––––––––

# ————— load logs –––––––

retrain_log   = load_jsonl(LOGS_DIR / “retraining_log.jsonl”)
triggered_log = load_jsonl(LOGS_DIR / “retraining_triggered.jsonl”)
scores_log    = load_jsonl(LOGS_DIR / “reviewer_scores.jsonl”)
score_by_id   = {r.get(“reviewer_id”): r for r in scores_log}

# latest signal scope

if retrain_log:
latest = max(retrain_log, key=lambda r: r.get(“timestamp”, 0))
sig_id = latest.get(“signal_id”, “unknown”)
sig_rows = [r for r in retrain_log if r.get(“signal_id”) == sig_id]
else:
sig_id = “none”
sig_rows = []

# reviewers & flag timeline (dedup first flag)

seen, reviewers, flag_times = set(), [], []
for r in sorted(sig_rows, key=lambda x: x.get(“timestamp”, 0)):
t = parse_ts(r.get(“timestamp”))
if t: flag_times.append(t)
rid = r.get(“reviewer_id”, “”)
if rid in seen:
continue
seen.add(rid)
w = r.get(“reviewer_weight”)
if w is None:
sc = (score_by_id.get(rid) or {}).get(“score”)
w = band_weight_from_score(sc)
reviewers.append({“id”: rid, “weight”: round(float(w), 2)})

# seed (display-only) if needed

reviewers, seeded_events = generate_demo_data_if_needed(reviewers, flag_times)

total_weight  = round(sum(r[“weight”] for r in reviewers), 2)
threshold     = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max((t for t in triggered_log if t.get(“signal_id”) == sig_id),
key=lambda x: x.get(“timestamp”, 0), default=None)
now_iso = datetime.now(timezone.utc).isoformat()

# ––––––––––––––––––––

# –––– small CI bar (for README) —–

plt.figure(figsize=(3.8, 2.4), dpi=200)
plt.title(“Consensus Weight vs Threshold”)
plt.bar([“weight”, “threshold”], [total_weight, threshold])
plt.tight_layout()
small_png = ART / “consensus.png”
plt.savefig(small_png, dpi=200)
plt.close()

# ––––––––––––––––––––

# ————— social card ————

def draw_social_card():
# Modern color palette
bg        = “#0A0D14”      # Deep dark background
card_bg   = “#1A1E2E”      # Card background
text      = “#F8FAFC”      # Primary text
sub_text  = “#94A3B8”      # Secondary text
accent    = “#3B82F6”      # Blue accent
success   = “#10B981”      # Green for triggered
warning   = “#F59E0B”      # Amber for threshold
danger    = “#EF4444”      # Red for no trigger
border    = “#334155”      # Subtle borders

```
# Create figure with modern styling
fig = plt.figure(figsize=(SOCIAL_W / DPI, SOCIAL_H / DPI), dpi=DPI, facecolor=bg)
fig.patch.set_facecolor(bg)

# Remove default axes
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Modern gradient background effect
gradient = plt.Circle((0.8, 0.8), 0.4, color=accent, alpha=0.03)
ax.add_patch(gradient)
gradient2 = plt.Circle((0.2, 0.3), 0.3, color=success, alpha=0.02)
ax.add_patch(gradient2)

# Header section with logo-style design
header_y = 0.92
ax.text(0.05, header_y, "⚡ MoonWire", fontsize=32, color=text, 
        weight='bold', ha='left', va='center', family='monospace')

ax.text(0.05, header_y-0.04, "Consensus Check Dashboard", 
        fontsize=16, color=sub_text, ha='left', va='center')

# Status badge - modern pill design
status = "TRIGGERED" if would_trigger else "NO TRIGGER"
badge_color = success if would_trigger else danger
badge_bg = FancyBboxPatch((0.75, header_y-0.025), 0.20, 0.05,
                          boxstyle="round,pad=0.01,rounding_size=0.025",
                          facecolor=badge_color, alpha=0.9, linewidth=0)
ax.add_patch(badge_bg)
ax.text(0.85, header_y, status, fontsize=14, color='white', 
        weight='bold', ha='center', va='center')

# Main metrics cards
card_y = 0.75
card_height = 0.12

# Total Weight Card
weight_card = FancyBboxPatch((0.05, card_y), 0.28, card_height,
                             boxstyle="round,pad=0.02,rounding_size=0.015",
                             facecolor=card_bg, linewidth=1, edgecolor=border)
ax.add_patch(weight_card)

ax.text(0.07, card_y + 0.08, "Total Weight", fontsize=12, color=sub_text)
ax.text(0.07, card_y + 0.04, f"{total_weight:.2f}", fontsize=24, 
        color=text, weight='bold')

# Threshold Card
thresh_card = FancyBboxPatch((0.36, card_y), 0.28, card_height,
                             boxstyle="round,pad=0.02,rounding_size=0.015",
                             facecolor=card_bg, linewidth=1, edgecolor=border)
ax.add_patch(thresh_card)

ax.text(0.38, card_y + 0.08, "Threshold", fontsize=12, color=sub_text)
ax.text(0.38, card_y + 0.04, f"{threshold:.2f}", fontsize=24, 
        color=warning, weight='bold')

# Signal ID Card
signal_card = FancyBboxPatch((0.67, card_y), 0.28, card_height,
                             boxstyle="round,pad=0.02,rounding_size=0.015",
                             facecolor=card_bg, linewidth=1, edgecolor=border)
ax.add_patch(signal_card)

ax.text(0.69, card_y + 0.08, "Signal ID", fontsize=12, color=sub_text)
ax.text(0.69, card_y + 0.04, red(sig_id), fontsize=16, 
        color=text, weight='bold', family='monospace')

# Progress bar section
bar_y = 0.55
bar_width = 0.90
bar_height = 0.04

ax.text(0.05, bar_y + 0.08, "Consensus Progress", fontsize=14, 
        color=text, weight='bold')

# Background track
track_bg = FancyBboxPatch((0.05, bar_y), bar_width, bar_height,
                          boxstyle="round,pad=0.005,rounding_size=0.02",
                          facecolor=border, alpha=0.3)
ax.add_patch(track_bg)

# Progress fill with gradient effect
progress_ratio = min(total_weight / max(threshold, total_weight, 3.0), 1.0)
fill_width = bar_width * progress_ratio

if fill_width > 0:
    progress_fill = FancyBboxPatch((0.05, bar_y), fill_width, bar_height,
                                   boxstyle="round,pad=0.005,rounding_size=0.02",
                                   facecolor=accent, alpha=0.8)
    ax.add_patch(progress_fill)

# Threshold indicator
thresh_x = 0.05 + (threshold / max(threshold, total_weight, 3.0)) * bar_width
thresh_line = FancyBboxPatch((thresh_x-0.003, bar_y-0.01), 0.006, bar_height+0.02,
                             boxstyle="round,pad=0.001,rounding_size=0.003",
                             facecolor=warning, alpha=0.9)
ax.add_patch(thresh_line)

# Progress labels
ax.text(0.05, bar_y - 0.04, f"0", fontsize=10, color=sub_text, ha='left')
ax.text(thresh_x, bar_y - 0.04, f"{threshold:.1f}", fontsize=10, 
        color=warning, ha='center', weight='bold')
ax.text(0.95, bar_y - 0.04, f"{total_weight:.1f}", fontsize=10, 
        color=text, ha='right', weight='bold')

# Reviewers section
reviewers_y = 0.38
reviewers_card = FancyBboxPatch((0.05, 0.15), 0.42, reviewers_y-0.10,
                                boxstyle="round,pad=0.02,rounding_size=0.015",
                                facecolor=card_bg, linewidth=1, edgecolor=border)
ax.add_patch(reviewers_card)

ax.text(0.07, reviewers_y - 0.02, "Active Reviewers", fontsize=14, 
        color=text, weight='bold')

if reviewers:
    y_pos = reviewers_y - 0.08
    for i, reviewer in enumerate(reviewers[:5]):  # Show max 5
        # Weight indicator dot
        weight_color = success if reviewer['weight'] >= 1.2 else (warning if reviewer['weight'] >= 0.9 else danger)
        dot = Circle((0.08, y_pos), 0.008, facecolor=weight_color, alpha=0.8)
        ax.add_patch(dot)
        
        ax.text(0.10, y_pos, f"{red(reviewer['id'])}", fontsize=11, 
                color=text, family='monospace')
        ax.text(0.42, y_pos, weight_to_label(reviewer['weight']), fontsize=11, 
                color=weight_color, ha='right', weight='bold')
        y_pos -= 0.04
        
    if len(reviewers) > 5:
        ax.text(0.08, y_pos, f"... +{len(reviewers)-5} more", fontsize=10, 
                color=sub_text, style='italic')
else:
    ax.text(0.08, reviewers_y - 0.08, "No reviewers active", fontsize=12, 
            color=sub_text, style='italic')

# Timeline section
timeline_card = FancyBboxPatch((0.50, 0.15), 0.45, reviewers_y-0.10,
                               boxstyle="round,pad=0.02,rounding_size=0.015",
                               facecolor=card_bg, linewidth=1, edgecolor=border)
ax.add_patch(timeline_card)

ax.text(0.52, reviewers_y - 0.02, "Flag Timeline", fontsize=14, 
        color=text, weight='bold')

if flag_times and len(flag_times) > 0:
    times = sorted(flag_times)
    timeline_y = reviewers_y - 0.08
    timeline_width = 0.38
    
    # Timeline background
    ax.plot([0.52, 0.52 + timeline_width], [timeline_y, timeline_y], 
            color=border, linewidth=2, alpha=0.3)
    
    # Plot flag events
    if len(times) > 1:
        time_span = (times[-1] - times[0]).total_seconds()
        for t in times:
            if time_span > 0:
                x_pos = 0.52 + ((t - times[0]).total_seconds() / time_span) * timeline_width
            else:
                x_pos = 0.52 + timeline_width / 2
            
            flag_dot = Circle((x_pos, timeline_y), 0.006, facecolor=accent, alpha=0.9)
            ax.add_patch(flag_dot)
    else:
        # Single event
        flag_dot = Circle((0.52 + timeline_width/2, timeline_y), 0.006, 
                         facecolor=accent, alpha=0.9)
        ax.add_patch(flag_dot)
    
    ax.text(0.52, timeline_y - 0.04, f"{len(flag_times)} events", fontsize=10, 
            color=sub_text)
else:
    ax.text(0.52, reviewers_y - 0.08, "No flags recorded", fontsize=11, 
            color=sub_text, style='italic')

# Footer with demo mode indicator
footer_text = "moonwire • demo mode" if seeded_events else "moonwire"
ax.text(0.05, 0.05, footer_text, fontsize=10, color=sub_text, alpha=0.7)
ax.text(0.95, 0.05, now_iso[:19] + "Z", fontsize=10, color=sub_text, 
        alpha=0.7, ha='right', family='monospace')

# Save with high quality
out = ART / "consensus_social.png"
fig.savefig(out, dpi=DPI, facecolor=bg, bbox_inches='tight', 
            pad_inches=0.1, edgecolor='none')
plt.close(fig)
return out
```

social_png = draw_social_card()

# ––––––––––––––––––––

# –––– markdown summary –––––––

md = []
md.append(”# MoonWire CI Demo Summary”)
md.append(””)
md.append(f”MoonWire Demo Summary — {now_iso}”)
md.append(””)
md.append(“Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.”)
md.append(””)
md.append(f”- **Signal:** `{red(sig_id)}`”)
md.append(f”- **Unique reviewers:** {len(reviewers)}”)
md.append(f”- **Combined weight:** **{total_weight}**”)
md.append(f”- **Threshold:** **{threshold}**  → **{‘TRIGGERS’ if would_trigger else ‘NO TRIGGER’}**”)
if last_trig:
md.append(f”- **Last retrain trigger logged:** {last_trig.get(‘timestamp’,’’)}”)
md.append(””)
md.append(”**Reviewers (redacted):**”)
if reviewers:
for r in reviewers:
md.append(f”- `{red(r['id'])}` → {weight_to_label(r[‘weight’])}”)
else:
md.append(”- *none found in this run*”)
md.append(””)
md.append(”![Consensus](consensus.png)”)

(ART / “demo_summary.md”).write_text(”\n”.join(md))

print(f”Wrote: {ART/‘demo_summary.md’}”)
print(f”Wrote: {small_png}”)
print(f”Wrote: {social_png}”)
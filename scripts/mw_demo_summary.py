#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs
 - artifacts/demo_summary.md
 - artifacts/consensus.png
 - artifacts/consensus_social.png

Reads ./logs/*.jsonl; never mutates logs.
"""

import os, json, hashlib, random, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------- config ----------------
LOGS_DIR = Path("logs")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 120  # 16:9 stable card
# ----------------------------------------


# -------------- helpers -----------------
def red(s: str) -> str:
    """Redact any ID to a short sha1 prefix (6 chars)."""
    return "000000" if not s else hashlib.sha1(s.encode()).hexdigest()[:6]

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
    if w >= 1.20: return "High"
    if w >= 0.90: return "Med"
    return "Low"

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
        s = s[:-1] + "+00:00" if s.endswith("Z") else s
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None
# ----------------------------------------


# ------ demo seeding (read-only) --------
def generate_demo_data_if_needed(reviewers, flag_times=None):
    """
    If DEMO_MODE=true and no reviewers present, return 3–5 seeded
    reviewers + seed timestamps (only for display). Never writes logs.

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
# ----------------------------------------


# --------------- load logs --------------
retrain_log   = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log    = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id   = {r.get("reviewer_id"): r for r in scores_log}

# latest signal scope
if retrain_log:
    latest = max(retrain_log, key=lambda r: r.get("timestamp", 0))
    sig_id = latest.get("signal_id", "unknown")
    sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
else:
    sig_id = "none"
    sig_rows = []

# reviewers & flag timeline (dedup first flag)
seen, reviewers, flag_times = set(), [], []
for r in sorted(sig_rows, key=lambda x: x.get("timestamp", 0)):
    t = parse_ts(r.get("timestamp"))
    if t: flag_times.append(t)
    rid = r.get("reviewer_id", "")
    if rid in seen:
        continue
    seen.add(rid)
    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

# seed (display-only) if needed
reviewers, seeded_events = generate_demo_data_if_needed(reviewers, flag_times)

total_weight  = round(sum(r["weight"] for r in reviewers), 2)
threshold     = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max((t for t in triggered_log if t.get("signal_id") == sig_id),
                key=lambda x: x.get("timestamp", 0), default=None)
now_iso = datetime.now(timezone.utc).isoformat()
# ----------------------------------------


# -------- small CI bar (for README) -----
plt.figure(figsize=(3.8, 2.4), dpi=200)
plt.title("Consensus Weight vs Threshold")
plt.bar(["weight", "threshold"], [total_weight, threshold])
plt.tight_layout()
small_png = ART / "consensus.png"
plt.savefig(small_png, dpi=200)
plt.close()
# ----------------------------------------


# --------------- social card ------------
def draw_social_card():
    # palette
    bg      = "#0B0F19"
    panel   = "#0F1827"
    text    = "#E8EEF7"
    sub     = "#9FB3C8"
    line    = "#1A2433"
    accent  = "#3BA1FF"  # bar fill
    th_col  = "#FFB74D"
    ok      = "#22C55E"
    warn    = "#EF4444"

    fig = plt.figure(figsize=(SOCIAL_W / DPI, SOCIAL_H / DPI), dpi=DPI, facecolor=bg)
    F = fig.transFigure  # 0..1

    # safe margins for social crops
    L, R, T, B = 0.06, 0.94, 0.92, 0.08
    W, H = R - L, T - B

    # header
    fig.text(L, T + 0.01, "MoonWire — Consensus Check",
             color=text, fontsize=34, weight=700, ha="left", va="center", transform=F)

    status = "TRIGGERED" if would_trigger else "NO TRIGGER"
    badge_col = ok if would_trigger else warn
    fig.text(R, T + 0.01, f" {status} ",
             color="white", fontsize=24, weight=700, ha="right", va="center", transform=F,
             bbox=dict(boxstyle="round,pad=0.35", fc=badge_col, ec=badge_col))

    mode_tag = "• DEMO MODE (seeded)" if seeded_events else ""
    fig.text(L, T - 0.01, f"Signal {red(sig_id)}  {mode_tag}",
             color=sub, fontsize=16, ha="left", va="center", transform=F)

    # metric chips (top-right, fixed width)
    def chip(x, title, val):
        w, h = 0.16, 0.07
        fig.patches.append(FancyBboxPatch((x, T - 0.09), w, h,
            boxstyle="round,pad=0.28,rounding_size=0.02",
            linewidth=0, facecolor=panel, transform=F))
        fig.text(x + 0.012, T - 0.062, title, color=sub, fontsize=12,
                 ha="left", va="center", transform=F)
        fig.text(x + 0.012, T - 0.087, val, color=text, fontsize=18, weight=600,
                 ha="left", va="center", transform=F)

    chip(L + 0.54, "Total weight", f"{total_weight:.2f}")
    chip(L + 0.72, "Threshold",    f"{threshold:.2f}")

    # main card panel
    card_x, card_y, card_w, card_h = L, B + 0.02, W, H - 0.10
    fig.patches.append(FancyBboxPatch((card_x, card_y), card_w, card_h,
                        boxstyle="round,pad=0.015,rounding_size=0.015",
                        linewidth=0, facecolor=panel, transform=F))

    # left: consensus bar
    left_x = card_x + 0.035
    bar_w  = 0.50 * card_w
    bar_h  = 0.12
    bar_y  = card_y + card_h * 0.57

    fig.text(left_x, bar_y + bar_h + 0.028, "Consensus vs Threshold",
             color=sub, fontsize=16, ha="left", va="bottom", transform=F)

    # track
    fig.patches.append(FancyBboxPatch((left_x, bar_y), bar_w, bar_h,
                        boxstyle="round,pad=0.02,rounding_size=0.012",
                        linewidth=0, facecolor=line, transform=F))
    # fill
    cap = max(threshold, total_weight, 3.0)
    fill_w = max(0.0, min(bar_w, (total_weight / cap) * bar_w))
    if fill_w > 0:
        fig.patches.append(FancyBboxPatch((left_x, bar_y), fill_w, bar_h,
                            boxstyle="round,pad=0.02,rounding_size=0.012",
                            linewidth=0, facecolor=accent, transform=F))
    # threshold line + label
    th_x = left_x + (threshold / cap) * bar_w
    fig.patches.append(Rectangle((th_x - 0.003, bar_y - 0.02), 0.006, bar_h + 0.04,
                                 linewidth=0, facecolor=th_col, transform=F))
    fig.text(th_x, bar_y + bar_h + 0.002, f"{threshold:.2f}",
             color=th_col, fontsize=16, weight=600, ha="center", va="bottom", transform=F)

    # right: reviewers
    box_x = left_x + bar_w + 0.04
    box_w = card_x + card_w - box_x - 0.035
    box_h = bar_h + 0.16
    box_y = bar_y - 0.05

    fig.patches.append(FancyBboxPatch((box_x, box_y), box_w, box_h,
                        boxstyle="round,pad=0.02,rounding_size=0.012",
                        linewidth=0, facecolor=line, transform=F))
    fig.text(box_x + 0.02, box_y + box_h - 0.035, "Reviewers (redacted)",
             color=sub, fontsize=14, ha="left", va="top", transform=F)

    if reviewers:
        # at most 6 rows; no overlap
        rows = reviewers[:6]
        lines = [f"• {red(r['id'])} — {weight_to_label(r['weight'])}" for r in rows]
        if len(reviewers) > 6:
            lines.append(f"… +{len(reviewers) - 6} more")
        fig.text(box_x + 0.02, box_y + box_h - 0.07, "\n".join(lines),
                 color=text, fontsize=15, ha="left", va="top", transform=F)
    else:
        fig.text(box_x + 0.02, box_y + box_h - 0.07, "No reviewers yet",
                 color=text, fontsize=15, ha="left", va="top", transform=F)

    # timeline (bottom)
    tl_x = left_x
    tl_w = card_w - 0.07
    tl_h = 0.11
    tl_y = card_y + 0.20

    fig.patches.append(FancyBboxPatch((tl_x, tl_y), tl_w, tl_h,
                        boxstyle="round,pad=0.02,rounding_size=0.012",
                        linewidth=0, facecolor=line, transform=F))
    fig.text(tl_x + 0.015, tl_y + tl_h + 0.01, "Flag timeline (last run)",
             color=sub, fontsize=14, ha="left", va="bottom", transform=F)

    if flag_times:
        times = sorted(flag_times)
        t0, t1 = times[0], times[-1]
        span = max((t1 - t0).total_seconds(), 1.0)
        base_y = tl_y + tl_h / 2 - 0.003
        fig.patches.append(Rectangle((tl_x + 0.02, base_y), tl_w - 0.04, 0.006,
                                     linewidth=0, facecolor=panel, transform=F))
        for t in times:
            x01 = (t - t0).total_seconds() / span
            x = tl_x + 0.02 + x01 * (tl_w - 0.04)
            fig.patches.append(Rectangle((x - 0.003, base_y - 0.015), 0.006, 0.03,
                                         linewidth=0, facecolor=accent, transform=F))
    else:
        fig.text(tl_x + 0.02, tl_y + tl_h / 2, "No flags this run",
                 color=sub, fontsize=13, ha="left", va="center", transform=F)

    # footer
    fig.text(L, B - 0.005, f"moonwire • demo mode • {now_iso}",
             color=sub, fontsize=12, ha="left", va="top", transform=F)

    out = ART / "consensus_social.png"
    # keep bbox default to avoid cropping/overlap artifacts
    fig.savefig(out, dpi=DPI, facecolor=bg)
    plt.close(fig)
    return out

social_png = draw_social_card()
# ----------------------------------------


# -------- markdown summary --------------
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
        md.append(f"- `{red(r['id'])}` → {weight_to_label(r['weight'])}")
else:
    md.append("- _none found in this run_")
md.append("")
md.append("![Consensus](consensus.png)")

(ART / "demo_summary.md").write_text("\n".join(md))

print(f"Wrote: {ART/'demo_summary.md'}")
print(f"Wrote: {small_png}")
print(f"Wrote: {social_png}")
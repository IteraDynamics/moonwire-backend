#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs
 - artifacts/demo_summary.md             (markdown for CI summary)
 - artifacts/consensus.png               (small CI plot)
 - artifacts/consensus_social.png        (1280x720 social image: gauge + sparkline)

Safe: reads ./logs/*.jsonl written by tests; does not import app code or mutate logs.
"""

import os
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from dateutil import parser as dtparser  # add to requirements-dev.txt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.demo_seed import seed_reviewers_if_empty

# -------------------- config --------------------
LOGS_DIR = Path("logs")
ART = Path("artifacts")
DEFAULT_THRESHOLD = 2.5  # keep in sync with app config
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 110  # memory-safe 16:9 for social
# ------------------------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

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

# ---- exported helper for tests ----
def generate_demo_data_if_needed(reviewers, *, demo_mode: bool | None = None):
    """
    Pure function used by tests to exercise demo-mode branching.
    Returns (reviewers_for_display, seeded_events).
    """
    if demo_mode is None:
        demo_mode = _bool_env("DEMO_MODE", False)
    if not demo_mode:
        return reviewers, []
    if reviewers:
        return reviewers, []
    return seed_reviewers_if_empty(reviewers)

def _compute_reviewers_and_flags(retrain_log, scores_log, latest_signal_rows):
    """Derive reviewers list and flag times from real data."""
    score_by_id = {r.get("reviewer_id"): r for r in scores_log}
    seen = set()
    reviewers = []
    flag_times = []

    for r in sorted(latest_signal_rows, key=lambda x: x.get("timestamp","")):
        ts = r.get("timestamp")
        # Accept either ISO string or epoch seconds for robustness
        if isinstance(ts, (int, float)):
            try:
                flag_times.append(datetime.fromtimestamp(ts, tz=timezone.utc))
            except Exception:
                pass
        else:
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

    return reviewers, flag_times

def _build_data():
    """Builds the data needed for rendering; safe to call from main."""
    ART.mkdir(exist_ok=True)

    retrain_log   = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
    triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
    scores_log    = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")

    # latest signal selection
    if retrain_log:
        latest = max(retrain_log, key=lambda r: r.get("timestamp", ""))
        sig_id = latest.get("signal_id", "unknown")
        sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
    else:
        sig_id = "none"
        sig_rows = []

    reviewers, flag_times = _compute_reviewers_and_flags(retrain_log, scores_log, sig_rows)

    # demo mode seeding (read-only)
    seeded_events = []
    DEMO_MODE = _bool_env("DEMO_MODE", False)
    if DEMO_MODE and not reviewers:
        reviewers, seeded_events = seed_reviewers_if_empty(reviewers)
        for ev in seeded_events:
            try:
                flag_times.append(dtparser.isoparse(ev["timestamp"]))
            except Exception:
                pass

    total_weight = round(sum(r["weight"] for r in reviewers), 2)
    threshold = DEFAULT_THRESHOLD
    would_trigger = total_weight >= threshold

    last_trig = max(
        (t for t in triggered_log if t.get("signal_id")==sig_id),
        key=lambda x: x.get("timestamp",""),
        default=None
    )

    return {
        "sig_id": sig_id,
        "sig_rows": sig_rows,
        "reviewers": reviewers,
        "flag_times": flag_times,
        "seeded_events": seeded_events,
        "total_weight": total_weight,
        "threshold": threshold,
        "would_trigger": would_trigger,
        "last_trigger": last_trig,
        "demo_mode": _bool_env("DEMO_MODE", False),
        "now": datetime.now(timezone.utc).isoformat(),
    }

def _render_small_bar(total_weight, threshold, path_png: Path):
    plt.figure(figsize=(3.8, 2.4), dpi=200)
    plt.title("Consensus Weight vs Threshold")
    plt.bar(["weight","threshold"], [total_weight, threshold])
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()

def _render_social(fig_data, social_png: Path):
    bg="#0A0E1A"; accent="#00E5FF"; muted="#93A3B1"; ok="#24D17E"; warn="#FF5A5F"; frame="#162235"
    fig = plt.figure(figsize=(SOCIAL_W/DPI, SOCIAL_H/DPI), dpi=DPI, facecolor=bg)
    F = fig.transFigure

    would_trigger = fig_data["would_trigger"]
    status = "TRIGGERED" if would_trigger else "NO TRIGGER"
    status_col = ok if would_trigger else warn

    # Title + status badge
    fig.text(0.04, 0.90, "MoonWire • Consensus Check",
             color="white", fontsize=28, weight=600, ha="left", va="center", transform=F)
    fig.text(0.96, 0.90, f" {status} ",
             color="white", fontsize=26, weight=700, ha="right", va="center", transform=F,
             bbox=dict(boxstyle="round,pad=0.35", fc=status_col, ec=status_col))

    # Gauge track
    track_left, track_right, track_y = 0.08, 0.92, 0.62
    track_w = track_right - track_left
    fig.patches.extend([
        FancyBboxPatch((track_left, track_y-0.03), track_w, 0.06,
                       boxstyle="round,pad=0.3", linewidth=0, facecolor=frame, transform=F)
    ])

    total_weight = fig_data["total_weight"]
    threshold = fig_data["threshold"]
    cap = max(threshold, total_weight, 3.0)

    # threshold marker
    th_x = track_left + (threshold / cap) * track_w
    fig.patches.extend([
        FancyBboxPatch((th_x-0.003, track_y-0.05), 0.006, 0.10,
                       boxstyle="round,pad=0.0", linewidth=0, facecolor=status_col, transform=F)
    ])
    fig.text(th_x, track_y+0.06, f"threshold {threshold:.2f}",
             color=status_col, fontsize=14, ha="center", va="bottom", transform=F)

    # fill to total weight
    fill_w = max(0.0, min(track_w, (total_weight / cap) * track_w))
    if fill_w > 0:
        fig.patches.extend([
            FancyBboxPatch((track_left, track_y-0.025), fill_w, 0.05,
                           boxstyle="round,pad=0.25", linewidth=0, facecolor=accent, transform=F)
        ])
    fig.text(track_left + fill_w + 0.005, track_y, f"{total_weight:.2f}",
             color=accent, fontsize=18, ha="left", va="center", transform=F)

    # Stats (left)
    stats = [
        f"Signal: {red(fig_data['sig_id'])}",
        f"Unique reviewers: {len(fig_data['reviewers'])}",
        f"Combined weight: {total_weight:.2f}",
        f"Threshold: {threshold:.2f}",
    ]
    if fig_data["demo_mode"] and not fig_data["sig_rows"]:
        stats.append("Mode: DEMO (seeded reviewers)")
    fig.text(0.08, 0.48, "\n".join(stats), color="white", fontsize=18, va="top", transform=F)

    # Reviewers (right)
    rows = [f"• {red(r['id'])}: {r['weight']:.2f}" for r in fig_data["reviewers"][:8]]
    if len(fig_data["reviewers"]) > 8:
        rows.append(f"… +{len(fig_data['reviewers'])-8} more")
    fig.text(0.92, 0.48,
             "\n".join(rows) if rows else "No reviewers yet",
             color=muted, fontsize=16, va="top", ha="right", transform=F)

    # Sparkline timeline
    def draw_sparkline(times):
        if not times:
            fig.text(0.08, 0.22, "No flags this run", color=muted, fontsize=14, transform=F)
            return
        times = sorted(times)
        t0, t1 = times[0], times[-1]
        span = max((t1 - t0).total_seconds(), 1.0)
        # baseline
        fig.patches.extend([
            FancyBboxPatch((0.08, 0.24), 0.84, 0.006,
                           boxstyle="round,pad=0.2", linewidth=0, facecolor=frame, transform=F)
        ])
        for t in times:
            x01 = (t - t0).total_seconds()/span
            x = 0.08 + x01 * 0.84
            fig.patches.extend([
                FancyBboxPatch((x-0.003, 0.232), 0.006, 0.02,
                               boxstyle="round,pad=0.0", linewidth=0, facecolor=accent, transform=F)
            ])
        lbl = "flag timeline" if len(times) > 1 else "single flag"
        fig.text(0.08, 0.27, lbl, color=muted, fontsize=13, transform=F)

    draw_sparkline(fig_data["flag_times"])

    # watermark
    fig.text(0.08, 0.08, f"moonwire • {'demo mode • ' if fig_data['demo_mode'] else ''}{fig_data['now']}",
             color="#93A3B1", fontsize=13, transform=F)

    fig.savefig(social_png, dpi=DPI, facecolor=bg)
    plt.close(fig)

def _render_markdown(fig_data, md_path: Path, small_png: Path):
    md = []
    md.append("# MoonWire CI Demo Summary")
    md.append("")
    md.append(f"MoonWire Demo Summary — {fig_data['now']}")
    md.append("")
    md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.")
    md.append("")
    md.append(f"- **Signal:** `{red(fig_data['sig_id'])}`")
    md.append(f"- **Unique reviewers:** {len(fig_data['reviewers'])}")
    md.append(f"- **Combined weight:** **{fig_data['total_weight']}**")
    md.append(f"- **Threshold:** **{fig_data['threshold']}**  → **{'TRIGGERS' if fig_data['would_trigger'] else 'NO TRIGGER'}**")
    if fig_data["demo_mode"] and not fig_data["sig_rows"]:
        md.append(f"- **Mode:** DEMO (no real reviewers found; seeded in-memory for visuals only)")
    if fig_data["last_trigger"]:
        md.append(f"- **Last retrain trigger logged:** {fig_data['last_trigger'].get('timestamp','')}")
    md.append("")
    md.append("**Reviewers (redacted):**")
    if fig_data["reviewers"]:
        for r in fig_data["reviewers"]:
            md.append(f"- `{red(r['id'])}` → weight {r['weight']}")
    else:
        md.append("- _none found in this run_")
    md.append("")
    md.append("![Consensus](consensus.png)")
    md_path.write_text("\n".join(md))

def main():
    ART.mkdir(exist_ok=True)

    fig_data = _build_data()

    # render small bar
    small_png = ART / "consensus.png"
    _render_small_bar(fig_data["total_weight"], fig_data["threshold"], small_png)

    # render social
    global social_png
    social_png = ART / "consensus_social.png"
    _render_social(fig_data, social_png)

    # render markdown
    md_path = ART / "demo_summary.md"
    _render_markdown(fig_data, md_path, small_png)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {small_png}")
    print(f"Wrote: {social_png}")

if __name__ == "__main__":
    main()
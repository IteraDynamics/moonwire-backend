# scripts/dashboard/governance_dashboard.py
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

# Reuse common helpers (graceful fallbacks keep CI green)
try:
    from scripts.summary_sections.common import ensure_dir, _iso  # type: ignore
except Exception:  # pragma: no cover
    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Ctx:
    """Minimal context adapter (compatible subset of SummaryContext)."""
    models_dir: Path
    artifacts_dir: Path
    logs_dir: Path | None = None
    is_demo: bool = False


# --------------------------
# Utilities
# --------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _read_json(path: Path) -> Any | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _take_last(seq: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if not isinstance(seq, list):
        return []
    return list(seq[-n:] if len(seq) >= n else seq)


def _mk_snapshot_png(summary: Dict[str, Any], out_png: Path) -> None:
    """
    Draw a compact snapshot image using matplotlib if present.
    Otherwise write a valid 1x1 PNG so artifact is non-empty.
    """
    if out_png.exists():
        return  # never overwrite

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore

        ensure_dir(out_png.parent)
        fig = plt.figure(figsize=(7.2, 3.6), dpi=150)
        ax = fig.add_subplot(111)
        ax.axis("off")

        a = summary["sections"]["apply"]
        b = summary["sections"]["bluegreen"]
        t = summary["sections"]["trend"]
        al = summary["sections"]["alerts"]

        lines = [
            "MoonWire Governance — 72h Snapshot",
            f"Generated: {summary.get('generated_at','—')}",
            f"Apply: mode={a.get('mode','-')} | applied {a.get('applied',0)}, skipped {a.get('skipped',0)}",
            f"Blue-Green: {b.get('current','?')} → {b.get('candidate','?')} "
            f"({b.get('classification','?')}, conf={b.get('confidence','—')})",
            f"Trend: F1 {t.get('f1_trend','—')} | ECE {t.get('ece_trend','—')}",
            f"Alerts: critical {al.get('critical',0)}, info {al.get('info',0)}",
        ]

        y = 0.94
        for ln in lines:
            ax.text(0.02, y, ln, transform=ax.transAxes, va="top")
            y -= 0.14

        fig.tight_layout()
        fig.savefig(str(out_png))
        return
    except Exception:
        pass

    # 1x1 fallback
    _PNG_1x1_BYTES = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
        b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    ensure_dir(out_png.parent)
    out_png.write_bytes(_PNG_1x1_BYTES)


def _compose_manifest(ctx: Ctx, run_url: str | None) -> Dict[str, Any]:
    """
    Read upstream artifacts and compose a normalized manifest with graceful defaults.
    """
    models = ctx.models_dir

    apply_j = _read_json(models / "governance_apply_result.json") or {}
    bg_j = _read_json(models / "bluegreen_promotion.json") or {}
    trend_j = _read_json(models / "model_performance_trend.json") or {}
    notif_j = _read_json(models / "governance_notifications_digest.json") or {}

    # Apply
    apply_mode = apply_j.get("action_mode") or apply_j.get("mode") or "dryrun"
    applied = int(apply_j.get("applied", 0))
    skipped = int(apply_j.get("skipped", 0))
    reversal_plan = apply_j.get("reversal_plan") or {}

    # Blue-Green
    bg_current = bg_j.get("current") or bg_j.get("current_version") or "v?.?.?"
    bg_candidate = bg_j.get("candidate") or bg_j.get("candidate_version") or "v?.?.?"
    bg_class = bg_j.get("classification") or bg_j.get("decision") or "observe"
    try:
        bg_conf = round(float(bg_j.get("confidence", 0.80)), 2)
    except Exception:
        bg_conf = 0.80

    # Trends
    f1_trend = trend_j.get("f1_trend") or trend_j.get("f1_status") or "stable"
    ece_trend = trend_j.get("ece_trend") or trend_j.get("ece_status") or "stable"

    # Alerts
    crit = len((notif_j or {}).get("critical", []))
    info = len((notif_j or {}).get("info", []))
    last3 = _take_last((notif_j or {}).get("critical", []) + (notif_j or {}).get("info", []), 3)

    return {
        "generated_at": _iso(_now_utc()),
        "window_hours": int(os.getenv("MW_DASH_WINDOW_H", "72")),
        "run_url": run_url,
        "sections": {
            "apply": {"mode": apply_mode, "applied": applied, "skipped": skipped, "reversal_plan": reversal_plan},
            "bluegreen": {
                "current": bg_current,
                "candidate": bg_candidate,
                "classification": bg_class,
                "confidence": bg_conf,
            },
            "trend": {"f1_trend": f1_trend, "ece_trend": ece_trend},
            "alerts": {"critical": crit, "info": info, "last": last3},
        },
        "demo": bool(ctx.is_demo),
    }


def _render_html(manifest: Dict[str, Any], png_data_uri: str | None) -> str:
    """Return a self-contained HTML document (inline CSS, no external deps)."""
    run_url = manifest.get("run_url")
    wh = manifest.get("window_hours", 72)
    gen = manifest.get("generated_at", "—")
    a, b, t, al = (
        manifest["sections"]["apply"],
        manifest["sections"]["bluegreen"],
        manifest["sections"]["trend"],
        manifest["sections"]["alerts"],
    )

    css = """
    <style>
      :root { --bg:#0b0f17; --card:#111827; --fg:#e5e7eb; --muted:#9ca3af; --acc:#06b6d4;
              --ok:#10b981; --warn:#f59e0b; --bad:#ef4444; --stroke:#1f2937; }
      * { box-sizing: border-box; }
      body { background: var(--bg); color: var(--fg); font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 0; }
      .wrap { max-width: 980px; margin: 28px auto 48px; padding: 0 16px; }
      h1 { font-size: 22px; margin: 0 0 10px; }
      .meta { color: var(--muted); margin-bottom: 18px; }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 14px; }
      .card { background: var(--card); border: 1px solid var(--stroke); border-radius: 10px; padding: 12px 14px; }
      .title { font-weight: 600; margin-bottom: 8px; }
      .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-left: 8px; background: #1f2937; color: var(--muted); border: 1px solid var(--stroke);}
      .ok   { color: #d1fae5; background: rgba(16,185,129,.14); border-color: rgba(16,185,129,.30);}
      .warn { color: #fef3c7; background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.28);}
      .bad  { color: #fee2e2; background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.28);}
      table { width: 100%; border-collapse: collapse; margin-top: 6px; font-variant-numeric: tabular-nums; }
      td { padding: 4px 6px; border-top: 1px solid var(--stroke); color: var(--fg); }
      td.k { color: var(--muted); width: 46%; }
      ul.events { margin: 6px 0 0 18px; padding: 0; }
      ul.events li { margin: 2px 0; }
      .footer { margin-top: 14px; color: var(--muted); }
      a { color: var(--acc); text-decoration: none; }
      img.snapshot { margin-top: 10px; max-width: 100%; border-radius: 8px; border: 1px solid var(--stroke); display: block; }
    </style>
    """

    def pill(kind: str, text: str) -> str:
        return f'<span class="pill {kind}">{text}</span>'

    cls = {"promote_ready": "ok", "observe": "warn", "rollback_risk": "bad"}.get(b.get("classification", "observe"), "warn")

    # Alerts list (last 3)
    events_html = ""
    if isinstance(al.get("last"), list) and al["last"]:
        items = []
        for ev in al["last"]:
            ver = ev.get("version", "—")
            et = ev.get("type", "event")
            conf = ev.get("conf")
            delta = ev.get("delta") or {}
            df1 = delta.get("F1")
            dece = delta.get("ECE")
            parts = [f"<b>{ver}</b> {et}"]
            if df1 is not None:
                parts.append(f"ΔF1 {df1:+.02f}")
            if dece is not None:
                parts.append(f"ΔECE {dece:+.02f}")
            if conf is not None:
                parts.append(f"conf {conf:.2f}")
            items.append("<li>" + " | ".join(parts) + "</li>")
        events_html = "<ul class='events'>" + "".join(items) + "</ul>"
    else:
        events_html = "<div class='meta'>No recent events.</div>"

    header = f"""
    <div class="wrap">
      <h1>MoonWire Governance Dashboard ({wh}h)</h1>
      <div class="meta">
        Generated <b>{gen}</b>
        {"• <a href='"+run_url+"' target='_blank' rel='noopener'>View CI Run</a>" if run_url else ""}
      </div>

      <div class="grid">
        <div class="card">
          <div class="title">A. Governance Apply {pill('ok' if a.get('mode')=='apply' else 'warn', a.get('mode','-'))}</div>
          <table>
            <tr><td class="k">Applied</td><td>{a.get('applied',0)}</td></tr>
            <tr><td class="k">Skipped</td><td>{a.get('skipped',0)}</td></tr>
          </table>
        </div>

        <div class="card">
          <div class="title">B. Blue-Green Simulation {pill(cls, b.get('classification','observe'))}</div>
          <table>
            <tr><td class="k">Current → Candidate</td><td>{b.get('current','?')} → {b.get('candidate','?')}</td></tr>
            <tr><td class="k">Confidence</td><td>{b.get('confidence','—')}</td></tr>
          </table>
        </div>

        <div class="card">
          <div class="title">C. Performance & Calibration</div>
          <table>
            <tr><td class="k">F1 Trend</td><td>{t.get('f1_trend','—')}</td></tr>
            <tr><td class="k">ECE Trend</td><td>{t.get('ece_trend','—')}</td></tr>
          </table>
        </div>

        <div class="card">
          <div class="title">D. Alerts & Notifications</div>
          <table>
            <tr><td class="k">Critical</td><td>{al.get('critical',0)}</td></tr>
            <tr><td class="k">Info</td><td>{al.get('info',0)}</td></tr>
          </table>
          {events_html}
        </div>
      </div>

      {"<img class='snapshot' alt='dashboard snapshot' src='"+png_data_uri+"'/>" if png_data_uri else ""}
      <div class="footer">Artifacts generated by MoonWire CI • HTML is self-contained (no external assets).</div>
    </div>
    """

    return f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{header}</body></html>"


# --------------------------
# Public API
# --------------------------

def build_dashboard(ctx) -> Dict[str, Any]:
    """
    Build governance dashboard artifacts:
      - HTML: artifacts/governance_dashboard.html
      - PNG : artifacts/governance_dashboard.png
      - JSON: models/governance_dashboard_manifest.json
    Returns a dict with file paths and section summaries.
    """
    # Normalize ctx to our local dataclass if a SummaryContext was passed in
    if not isinstance(ctx, Ctx):
        ctx = Ctx(
            models_dir=Path(ctx.models_dir),
            artifacts_dir=Path(ctx.artifacts_dir),
            logs_dir=getattr(ctx, "logs_dir", None),
            is_demo=getattr(ctx, "is_demo", False),
        )

    arts = ensure_dir(Path(ctx.artifacts_dir))
    models = ensure_dir(Path(ctx.models_dir))

    out_html = arts / "governance_dashboard.html"
    out_png = arts / "governance_dashboard.png"
    out_manifest = models / "governance_dashboard_manifest.json"

    run_url = os.getenv("GITHUB_RUN_URL") or None

    # Compose and persist manifest
    manifest = _compose_manifest(ctx, run_url)
    try:
        ensure_dir(out_manifest.parent)
        out_manifest.write_text(json.dumps(manifest, indent=2))
    except Exception:
        pass  # never fail CI

    # Snapshot PNG
    _mk_snapshot_png(manifest, out_png)

    # Embed PNG (if exists) as data URI
    png_data_uri = None
    try:
        if out_png.exists():
            data = base64.b64encode(out_png.read_bytes()).decode("ascii")
            png_data_uri = f"data:image/png;base64,{data}"
    except Exception:
        png_data_uri = None

    # Render HTML
    html = _render_html(manifest, png_data_uri)
    ensure_dir(out_html.parent)
    out_html.write_text(html)

    return {
        "generated_at": manifest["generated_at"],
        "window_hours": manifest["window_hours"],
        "artifacts": {
            "html": str(out_html),
            "png": str(out_png),
            "manifest": str(out_manifest),
            "run_url": run_url,
        },
        "sections": manifest["sections"],
        "demo": manifest["demo"],
    }
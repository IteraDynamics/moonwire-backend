# scripts/dashboard/governance_dashboard.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso

# Minimal valid 1x1 PNG (black) placeholder
_PNG_1x1_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

# ------------ helpers ------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text() or "{}")
    except Exception:
        pass
    return None

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _sparkline_svg(values: List[float], w: int = 120, h: int = 28) -> str:
    """
    Tiny inline SVG sparkline with a baseline. No deps.
    """
    if not values:
        values = [0.0, 0.0]
    vmin = min(values)
    vmax = max(values)
    span = (vmax - vmin) or 1.0
    # normalize to [2, h-2] with inverted Y for svg
    pts = []
    n = len(values)
    for i, v in enumerate(values):
        x = int(i * (w - 4) / max(1, n - 1)) + 2
        y = int(h - 2 - ((v - vmin) / span) * (h - 4))
        pts.append(f"{x},{y}")
    path = " ".join(pts)
    color = "#4B8BF4"
    baseline_y = int(h - 2)
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="sparkline">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}" />'
        f'<line x1="2" y1="{baseline_y}" x2="{w-2}" y2="{baseline_y}" stroke="#ddd" stroke-width="1"/>'
        "</svg>"
    )

def _trend_label(series: List[float], up_is_good: bool = True) -> str:
    if not series or len(series) < 2:
        return "stable"
    delta = series[-1] - series[0]
    eps = 1e-6
    if up_is_good:
        if delta > eps:
            return "improving"
        if delta < -eps:
            return "declining"
    else:
        if delta < -eps:
            return "improving"
        if delta > eps:
            return "worsening"
    return "stable"

def _format_delta(x: Optional[float], plus_sign: bool = True) -> str:
    if x is None:
        return "0.00"
    s = f"{x:+.2f}" if plus_sign else f"{x:.2f}"
    # Replace leading + with Unicode plus to mirror your summary style
    s = s.replace("+", "+")
    return s

def _write_png_placeholder(path: Path, title_text: str = "MoonWire Governance Dashboard") -> None:
    """
    Attempt a small matplotlib composite; if not available, write 1x1 PNG.
    """
    if path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa
        ensure_dir(path.parent)
        fig = plt.figure(figsize=(6, 3), dpi=160)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.6, title_text, ha="center", va="center", fontsize=10)
        ax.text(0.5, 0.35, "Static snapshot — see HTML for details", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(path))
        plt.close(fig)
        return
    except Exception:
        pass
    ensure_dir(path.parent)
    path.write_bytes(_PNG_1x1_BYTES)

# ------------ core ------------

@dataclass
class _Inputs:
    apply: Optional[Dict[str, Any]]
    bluegreen: Optional[Dict[str, Any]]
    trend: Optional[Dict[str, Any]]
    notif: Optional[Dict[str, Any]]
    run_url: Optional[str]
    window_h: int

def _load_inputs(ctx: SummaryContext) -> _Inputs:
    models = Path(ctx.models_dir)
    apply_path = models / "governance_apply_result.json"
    bg_path = models / "bluegreen_promotion.json"
    trend_path = models / "model_performance_trend.json"
    notif_path = models / "governance_notifications_digest.json"
    run_url = os.getenv("GITHUB_RUN_URL", None)
    window_h = int(os.getenv("MW_DASH_WINDOW_H", "72") or "72")

    return _Inputs(
        apply=_read_json(apply_path),
        bluegreen=_read_json(bg_path),
        trend=_read_json(trend_path),
        notif=_read_json(notif_path),
        run_url=run_url,
        window_h=window_h,
    )

def _trend_series(trend_json: Optional[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """
    Try to extract F1 and ECE series. If not present, return small demo lists.
    Expected shapes seen in earlier tasks: {"versions":[{"version":"vX","F1":..,"ece":..}, ...]}
    """
    f1_series: List[float] = []
    ece_series: List[float] = []
    if trend_json and isinstance(trend_json.get("versions"), list):
        for row in trend_json["versions"]:
            f1 = row.get("F1") or row.get("f1")
            ece = row.get("ECE") or row.get("ece")
            if f1 is not None:
                f1_series.append(_safe_float(f1, default=0.7))
            if ece is not None:
                ece_series.append(_safe_float(ece, default=0.06))
    if not f1_series:
        f1_series = [0.68, 0.70, 0.71]
    if not ece_series:
        ece_series = [0.065, 0.060, 0.055]
    return f1_series, ece_series

def _bluegreen_summary(bg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not bg:
        return {
            "current": "v?.?.?",
            "candidate": "v?.?.?",
            "classification": "observe",
            "confidence": 0.80,
            "delta": {"F1": +0.00, "ECE": -0.00, "precision": 0.00, "recall": 0.00},
            "links": [],
        }
    d = bg.get("delta", {})
    return {
        "current": bg.get("current_model") or bg.get("current") or "v?.?.?",
        "candidate": bg.get("candidate") or "v?.?.?",
        "classification": bg.get("classification") or "observe",
        "confidence": _safe_float(bg.get("confidence"), 0.80),
        "delta": {
            "F1": _safe_float(d.get("F1") or d.get("f1"), 0.00),
            "ECE": _safe_float(d.get("ECE") or d.get("ece"), 0.00),
            "precision": _safe_float(d.get("precision"), 0.00),
            "recall": _safe_float(d.get("recall"), 0.00),
        },
        "links": ["bluegreen_comparison.png", "bluegreen_timeline.png"],
    }

def _apply_summary(apply_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not apply_json:
        return {
            "mode": "dryrun",
            "applied": 0,
            "skipped": 0,
            "reversal_plan": {"window_hours": 12, "trigger": "precision_regression|-0.02|any"},
        }
    return {
        "mode": apply_json.get("mode", "dryrun"),
        "applied": len(apply_json.get("applied", []) or []),
        "skipped": len(apply_json.get("skipped", []) or []),
        "reversal_plan": apply_json.get("reversal_plan")
            or {"window_hours": 12, "trigger": "precision_regression|-0.02|any"},
    }

def _alerts_summary(notif_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not notif_json:
        return {"critical": 0, "info": 0, "last": []}
    return {
        "critical": len(notif_json.get("critical", []) or []),
        "info": len(notif_json.get("info", []) or []),
        "last": (notif_json.get("critical", []) or [])[:1] + (notif_json.get("info", []) or [])[:2],
    }

def _html_template(
    header: Dict[str, Any],
    tile_apply: Dict[str, Any],
    tile_bg: Dict[str, Any],
    tile_perf: Dict[str, Any],
    tile_alerts: Dict[str, Any],
) -> str:
    run_link_html = (
        f'<a href="{header["run_url"]}" target="_blank" rel="noopener">View CI Run</a>'
        if header.get("run_url") else "<em>no run link</em>"
    )
    css = """
    <style>
      :root { --bg:#0b0f1a; --card:#111827; --fg:#e5e7eb; --muted:#9ca3af; --pill:#374151; --ok:#10b981; --warn:#f59e0b; --bad:#ef4444; }
      body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--fg); }
      .wrap { padding: 18px; max-width: 1100px; margin: 0 auto; }
      .hdr { display:flex; align-items:baseline; gap:12px; margin-bottom: 14px; }
      .hdr h1 { font-size: 20px; margin: 0; }
      .muted { color: var(--muted); }
      .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
      .card { background: var(--card); border-radius: 10px; padding: 12px; }
      .pill { display:inline-block; background: var(--pill); border-radius: 999px; padding: 2px 8px; font-size: 12px; }
      .row { display:flex; justify-content: space-between; gap:8px; margin:6px 0; }
      table { width:100%; border-collapse: collapse; }
      td, th { padding: 6px 4px; border-bottom: 1px solid #1f2937; font-size: 13px; }
      .ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); }
      .small { font-size: 12px; }
      .footer { margin-top: 10px; color: var(--muted); font-size: 12px; }
      .links a { color:#60a5fa; text-decoration: none; }
      .links a:hover { text-decoration: underline; }
      .tile-title { font-weight:600; margin-bottom: 6px; }
      .svgbox { margin-top: 6px; }
    </style>
    """
    # Performance tile composition
    perf_html = (
        f'<div class="row"><div>F1 trend</div><div class="muted small">{tile_perf["f1_trend"]}</div></div>'
        f'<div class="svgbox">{tile_perf["f1_svg"]}</div>'
        f'<div class="row" style="margin-top:8px;"><div>ECE trend</div><div class="muted small">{tile_perf["ece_trend"]}</div></div>'
        f'<div class="svgbox">{tile_perf["ece_svg"]}</div>'
    )

    # Alerts tile recent
    last_rows = []
    for e in tile_alerts.get("last", [])[:3]:
        ver = e.get("version", "?")
        typ = e.get("type", "info")
        conf = e.get("conf", e.get("confidence"))
        dF1 = e.get("delta", {}).get("F1")
        dECE = e.get("delta", {}).get("ECE")
        last_rows.append(
            f'<tr><td>{ver}</td><td>{typ}</td><td>{"" if conf is None else f"{conf:.2f}"}</td>'
            f'<td>{"" if dF1 is None else _format_delta(dF1)}</td>'
            f'<td>{"" if dECE is None else _format_delta(dECE)}</td></tr>'
        )
    last_html = "".join(last_rows) or '<tr><td colspan="5" class="muted small">No recent notifications</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8" />
<title>MoonWire Governance Dashboard (72h)</title>
{css}
<body>
  <div class="wrap">
    <div class="hdr">
      <h1>MoonWire Governance Dashboard (72h)</h1>
      <span class="muted small">generated_at {header["generated_at"]}</span>
      <span class="muted small">window {header["window_h"]}h</span>
      <span class="pill small">mode {header["mode"]}</span>
      <span class="muted small">{run_link_html}</span>
    </div>

    <div class="grid">
      <!-- Tile A: Apply -->
      <div class="card">
        <div class="tile-title">Governance Apply</div>
        <div class="row"><div>mode</div><div class="pill small">{tile_apply["mode"]}</div></div>
        <div class="row"><div>applied</div><div class="ok">{tile_apply["applied"]}</div></div>
        <div class="row"><div>skipped</div><div class="muted">{tile_apply["skipped"]}</div></div>
        <div class="small muted">reversal: {tile_apply["reversal_plan"]["window_hours"]}h • {tile_apply["reversal_plan"]["trigger"]}</div>
      </div>

      <!-- Tile B: Blue-Green -->
      <div class="card">
        <div class="tile-title">Blue-Green Simulation</div>
        <div class="row"><div>current → candidate</div><div>{tile_bg["current"]} → {tile_bg["candidate"]}</div></div>
        <div class="row"><div>ΔF1</div><div>{_format_delta(tile_bg["delta"]["F1"])}</div></div>
        <div class="row"><div>ΔECE</div><div>{_format_delta(tile_bg["delta"]["ECE"])}</div></div>
        <div class="row"><div>classification</div><div>{tile_bg["classification"]}</div></div>
        <div class="row"><div>confidence</div><div>{tile_bg["confidence"]:.2f}</div></div>
        <div class="links small">visuals: {", ".join(tile_bg["links"])}</div>
      </div>

      <!-- Tile C: Perf & Cal -->
      <div class="card">
        <div class="tile-title">Performance & Calibration</div>
        {perf_html}
      </div>

      <!-- Tile D: Alerts -->
      <div class="card">
        <div class="tile-title">Alerts & Notifications</div>
        <div class="row"><div>critical</div><div class="bad">{tile_alerts["critical"]}</div></div>
        <div class="row"><div>info</div><div class="muted">{tile_alerts["info"]}</div></div>
        <table class="small" style="margin-top:6px;">
          <thead><tr><th>version</th><th>type</th><th>conf</th><th>ΔF1</th><th>ΔECE</th></tr></thead>
          <tbody>{last_html}</tbody>
        </table>
      </div>
    </div>

    <div class="footer">
      artifacts: governance_dashboard.html • governance_dashboard.png • governance_dashboard_manifest.json
      <br/>provenance: CI-generated static digest
    </div>
  </div>
</body>
</html>"""

def build_dashboard(ctx: SummaryContext) -> Dict[str, Any]:
    """
    Build unified HTML + PNG + manifest for last 72h governance intelligence.
    Returns the manifest dict.
    """
    arts = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    models = Path(ctx.models_dir)

    ensure_dir(arts); ensure_dir(models)

    inputs = _load_inputs(ctx)

    # Build summaries
    apply_s = _apply_summary(inputs.apply)
    bg_s = _bluegreen_summary(inputs.bluegreen)
    f1_series, ece_series = _trend_series(inputs.trend)
    perf_s = {
        "f1_trend": _trend_label(f1_series, up_is_good=True),
        "ece_trend": _trend_label(ece_series, up_is_good=False),
        "f1_svg": _sparkline_svg(f1_series),
        "ece_svg": _sparkline_svg(ece_series),
    }
    alerts_s = _alerts_summary(inputs.notif)

    # Mode badge: prefer governance apply's mode, else demo/dryrun
    mode = apply_s.get("mode") or ("dryrun")

    header = {
        "generated_at": _iso(_now_utc()),
        "run_url": inputs.run_url,
        "window_h": inputs.window_h,
        "mode": mode,
    }

    # Manifest JSON
    manifest = {
        "generated_at": header["generated_at"],
        "window_hours": inputs.window_h,
        "run_url": inputs.run_url,
        "sections": {
            "apply": {"mode": apply_s["mode"], "applied": apply_s["applied"], "skipped": apply_s["skipped"]},
            "bluegreen": {
                "current": bg_s["current"],
                "candidate": bg_s["candidate"],
                "classification": bg_s["classification"],
                "confidence": float(bg_s["confidence"]),
            },
            "trend": {
                "f1_trend": perf_s["f1_trend"],
                "ece_trend": perf_s["ece_trend"],
            },
            "alerts": {
                "critical": alerts_s["critical"],
                "info": alerts_s["info"],
            },
        },
        "demo": bool(getattr(ctx, "is_demo", False)),
    }

    # Write HTML
    html = _html_template(header, apply_s, bg_s, perf_s, alerts_s)
    html_out = arts / "governance_dashboard.html"
    ensure_dir(html_out.parent)
    html_out.write_text(html, encoding="utf-8")

    # Snapshot PNG (best-effort)
    png_out = arts / "governance_dashboard.png"
    _write_png_placeholder(png_out)

    # Manifest path
    manifest_out = models / "governance_dashboard_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    return manifest
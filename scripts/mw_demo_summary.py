# scripts/mw_demo_summary.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# -------- Minimal context used by tests/callers (kept light to avoid import cycles) --------
@dataclass
class _Dirs:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------
# Public API expected by tests
# ------------------------------------------------------------------------------------------
def generate_demo_data_if_needed(
    models_dir: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Seed minimal demo artifacts if DEMO_MODE/MW_DEMO is enabled.
    Returns a report dict with 'seeded' boolean and 'created' file list.
    Safe to call multiple times; only creates files when missing.
    """
    demo_on = _env_bool("DEMO_MODE") or _env_bool("MW_DEMO")
    # Allow callers/tests to pass explicit dirs; fall back to repo defaults
    repo_root = Path(os.getenv("GITHUB_WORKSPACE", Path.cwd()))
    models = Path(models_dir) if models_dir else (repo_root / "models")
    logs = Path(logs_dir) if logs_dir else (repo_root / "logs")

    created: List[str] = []
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    if not demo_on:
        return {"seeded": False, "created": created, "demo": False}

    # --- Minimal, stable demo artifacts that some tests expect to exist ---
    # 1) training_data.jsonl (small, seeded rows)
    td_path = models / "training_data.jsonl"
    if not td_path.exists():
        demo_rows = [
            {"t": _now_iso(), "origin": "reddit", "score": 0.72, "label": 1},
            {"t": _now_iso(), "origin": "twitter", "score": 0.41, "label": 0},
            {"t": _now_iso(), "origin": "rss_news", "score": 0.63, "label": 1},
        ]
        td_path.write_text("\n".join(json.dumps(r) for r in demo_rows) + "\n", encoding="utf-8")
        created.append(str(td_path))

    # 2) accuracy_by_version.json (tiny snapshot used by CI summary uploads)
    abv_path = models / "accuracy_by_version.json"
    if not abv_path.exists():
        abv_payload = {
            "generated_at": _now_iso(),
            "versions": {
                "v0.5.0": {"n": 50, "acc": 0.82},
                "v0.5.1": {"n": 65, "acc": 0.84},
            },
        }
        abv_path.write_text(json.dumps(abv_payload, indent=2), encoding="utf-8")
        created.append(str(abv_path))

    # 3) market_context.json (seed series so market-aware sections can run in demo)
    mc_path = models / "market_context.json"
    if not mc_path.exists():
        # simple synthetic series (hourly) for 24h lookback
        coins = ["bitcoin", "ethereum", "solana"]
        series: Dict[str, List[Dict[str, float]]] = {}
        # seed 24 hourly points per coin
        base = {"bitcoin": 60000.0, "ethereum": 3000.0, "solana": 150.0}
        for coin in coins:
            pts = []
            price = base[coin]
            for h in range(24, 0, -1):
                # tiny deterministic wiggle
                price *= (1.0 + ((h % 5) - 2) * 0.0008)
                t = int((datetime.now(timezone.utc)).timestamp()) - h * 3600
                pts.append({"t": t, "price": round(price, 2)})
            series[coin] = pts
        mc_payload = {
            "generated_at": _now_iso(),
            "vs": "usd",
            "coins": coins,
            "window_hours": 24,
            "series": series,
            "returns": {},  # left empty; downstream can compute
            "demo": True,
            "attribution": "CoinGecko (demo seed)",
        }
        mc_path.write_text(json.dumps(mc_payload, indent=2), encoding="utf-8")
        created.append(str(mc_path))

    # 4) signal_quality_summary.json (harmless tiny stub)
    sq_path = models / "signal_quality_summary.json"
    if not sq_path.exists():
        sq_payload = {
            "generated_at": _now_iso(),
            "window_h": int(os.getenv("METRICS_LOOKBACK_HOURS", "72")),
            "origins": {"reddit": {"n": 12}, "twitter": {"n": 9}, "rss_news": {"n": 7}},
        }
        sq_path.write_text(json.dumps(sq_payload, indent=2), encoding="utf-8")
        created.append(str(sq_path))

    # 5) Append a simple market price line into logs/market_prices.jsonl (tests may assert existence)
    mp_log = logs / "market_prices.jsonl"
    line = {
        "generated_at": _now_iso(),
        "source": "coingecko_demo",
        "vs": "usd",
        "coins": ["bitcoin", "ethereum", "solana"],
        "prices": {"bitcoin": 60790.05, "ethereum": 3047.24, "solana": 151.24},
        "demo": True,
    }
    with mp_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    # only count as "created" if file was just created
    if mp_log.stat().st_size <= len(json.dumps(line)) + 2:
        created.append(str(mp_log))

    return {"seeded": True, "created": created, "demo": True}


# ------------------------------------------------------------------------------------------
# Simple CLI that builds the demo summary markdown used in CI
# ------------------------------------------------------------------------------------------
def _build_demo_summary_md(artifacts_dir: Path, models_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts_dir / "demo_summary.md"

    # Read a couple of models/* artifacts if present to show something meaningful
    abv = {}
    abv_path = models_dir / "accuracy_by_version.json"
    if abv_path.exists():
        try:
            abv = json.loads(abv_path.read_text(encoding="utf-8"))
        except Exception:
            abv = {}

    market_line = "Market Context: demo seed"
    mc_path = models_dir / "market_context.json"
    if mc_path.exists():
        try:
            mc = json.loads(mc_path.read_text(encoding="utf-8"))
            coins = mc.get("coins", [])
            market_line = f"Market Context: {', '.join(coins)} (window={mc.get('window_hours','?')}h)"
        except Exception:
            pass

    summary_md = [
        "# MoonWire CI Demo Summary",
        "",
        f"- Generated at: **{_now_iso()}**",
        f"- {market_line}",
        "",
        "## Accuracy by Version (demo)",
        "```json",
        json.dumps(abv, indent=2),
        "```",
    ]
    summary_path.write_text("\n".join(summary_md) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    repo_root = Path(os.getenv("GITHUB_WORKSPACE", Path.cwd()))
    models = repo_root / "models"
    logs = repo_root / "logs"
    artifacts = Path(os.getenv("ARTIFACTS_DIR", repo_root / "artifacts"))

    # Seed demo if flagged
    generate_demo_data_if_needed(models_dir=models, logs_dir=logs)

    # Always try to produce a summary markdown for CI step that uploads it
    summary = _build_demo_summary_md(artifacts, models)
    print(f"[mw_demo_summary] wrote {summary}")


if __name__ == "__main__":
    main()
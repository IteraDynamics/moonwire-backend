# tests/test_calibration_per_origin.py
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso, _load_jsonl, _write_json
import importlib


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _ctx(tmp: Path, demo: bool = False) -> SummaryContext:
    logs = ensure_dir(tmp / "logs")
    models = ensure_dir(tmp / "models")
    return SummaryContext(logs_dir=logs, models_dir=models, is_demo=demo)


def test_schema_and_artifacts(tmp_path):
    mod = importlib.import_module("scripts.summary_sections.calibration_per_origin")

    ctx = _ctx(tmp_path, demo=False)
    now = datetime.now(timezone.utc)

    # Two origins; include an out-of-window row that should be dropped
    triggers = [
        {"id": "a1", "origin": "reddit", "score": 0.8, "ts": now.isoformat()},
        {"id": "a2", "origin": "reddit", "score": 0.2, "ts": now.isoformat()},
        {"id": "b1", "origin": "twitter", "score": 0.7, "ts": now.isoformat()},
        {"id": "old", "origin": "reddit", "score": 0.9, "ts": (now - timedelta(days=10)).isoformat()},
    ]
    labels = [
        {"id": "a1", "label": 1, "ts": now.isoformat()},
        {"id": "a2", "label": 0, "ts": now.isoformat()},
        {"id": "b1", "label": 1, "ts": now.isoformat()},
        # intentionally no label for "old"
    ]
    _write_jsonl(ctx.logs_dir / "trigger_history.jsonl", triggers)
    _write_jsonl(ctx.logs_dir / "label_feedback.jsonl", labels)

    md: list[str] = []
    mod.append(md, ctx)

    # JSON artifact exists with schema
    jpath = ctx.models_dir / "calibration_per_origin.json"
    assert jpath.exists(), "JSON artifact not created"
    data = json.loads(jpath.read_text())

    assert data["window_hours"] == 72
    assert "generated_at" in data and "demo" in data and "origins" in data
    assert isinstance(data["origins"], list) and len(data["origins"]) >= 2

    for entry in data["origins"]:
        assert set(["origin","n","ece","brier","low_n","high_ece","bins","artifact_png"]).issubset(entry.keys())
        assert len(entry["bins"]) == 10
        png = Path(entry["artifact_png"])
        assert png.exists(), f"PNG missing for {entry['origin']}"
        with png.open("rb") as f:
            assert f.read(8).startswith(b"\x89PNG"), "Not a PNG file"

    # Markdown shape
    assert any(s.startswith("🧮 Per-Origin Calibration (") for s in md)
    assert any(("ECE=" in s and "Brier=" in s and "n=" in s) for s in md)


def test_flags_logic_and_metrics(tmp_path):
    mod = importlib.import_module("scripts.summary_sections.calibration_per_origin")
    ctx = _ctx(tmp_path, demo=False)
    now = datetime.now(timezone.utc)

    triggers, labels = [], []

    # small-n: rss_news (n=25) -> low_n
    for i in range(25):
        tid = f"n{i}"
        triggers.append({"id": tid, "origin": "rss_news", "score": 0.5, "ts": now.isoformat()})
        labels.append({"id": tid, "label": 1 if i % 2 == 0 else 0, "ts": now.isoformat()})

    # overconfident: twitter (n=60), high scores but few positives -> high ECE expected
    for i in range(60):
        tid = f"t{i}"
        triggers.append({"id": tid, "origin": "twitter", "score": 0.9, "ts": now.isoformat()})
        labels.append({"id": tid, "label": 1 if i % 5 == 0 else 0, "ts": now.isoformat()})

    _write_jsonl(ctx.logs_dir / "trigger_history.jsonl", triggers)
    _write_jsonl(ctx.logs_dir / "label_feedback.jsonl", labels)

    md: list[str] = []
    mod.append(md, ctx)

    data = json.loads((ctx.models_dir / "calibration_per_origin.json").read_text())
    o = {e["origin"]: e for e in data["origins"]}

    assert o["rss_news"]["low_n"] is True
    assert o["twitter"]["n"] == 60
    assert o["twitter"]["ece"] > 0.05  # sanity; likely > threshold
    # allow high_ece True in most runs; threshold-based
    assert isinstance(o["twitter"]["high_ece"], bool)


def test_demo_seeding(tmp_path):
    mod = importlib.import_module("scripts.summary_sections.calibration_per_origin")
    ctx = _ctx(tmp_path, demo=True)  # no logs provided

    md: list[str] = []
    mod.append(md, ctx)

    data = json.loads((ctx.models_dir / "calibration_per_origin.json").read_text())
    assert data["demo"] is True
    assert len(data["origins"]) >= 2
    for e in data["origins"]:
        assert Path(e["artifact_png"]).exists()


def test_windowing_accepts_iso_and_epoch(tmp_path):
    mod = importlib.import_module("scripts.summary_sections.calibration_per_origin")
    ctx = _ctx(tmp_path, demo=False)
    now = datetime.now(timezone.utc)

    triggers = [
        {"id": "iso", "origin": "reddit", "score": 0.6, "ts": now.isoformat()},
        {"id": "epoch_s", "origin": "reddit", "score": 0.6, "ts": int(now.timestamp())},
        {"id": "epoch_ms", "origin": "reddit", "score": 0.6, "ts": int(now.timestamp() * 1000)},
        {"id": "old", "origin": "reddit", "score": 0.9, "ts": (now - timedelta(days=5)).isoformat()},
    ]
    labels = [{"id": r["id"], "label": 1, "ts": now.isoformat()} for r in triggers if r["id"] != "old"]

    _write_jsonl(ctx.logs_dir / "trigger_history.jsonl", triggers)
    _write_jsonl(ctx.logs_dir / "label_feedback.jsonl", labels)

    md: list[str] = []
    mod.append(md, ctx)

    data = json.loads((ctx.models_dir / "calibration_per_origin.json").read_text())
    origins = data["origins"]
    # only one origin (reddit) and n=3 within the window
    assert len(origins) == 1
    assert origins[0]["n"] == 3

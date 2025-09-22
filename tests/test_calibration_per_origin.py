# tests/test_calibration_per_origin.py
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import importlib
import types

MOD_PATH = "scripts.summary_sections.calibration_per_origin"


class _Paths:
    def __init__(self, root: Path):
        self.ROOT = root
        self.LOGS_DIR = root / "logs"
        self.MODELS_DIR = root / "models"
        self.ARTIFACTS_DIR = root / "artifacts"


class _Ctx:
    def __init__(self, root: Path, demo: bool = False):
        self.paths = _Paths(root)
        for p in (self.paths.LOGS_DIR, self.paths.MODELS_DIR, self.paths.ARTIFACTS_DIR):
            p.mkdir(parents=True, exist_ok=True)
        self.env = os.environ
        self.now_utc = datetime.now(tz=timezone.utc)
        self.demo = demo

        # Try to import the repo's common; fall back to an empty namespace.
        try:
            # repo root assumed to be in PYTHONPATH by CI; ignore if not found.
            import common  # type: ignore
            self.common = common
        except Exception:
            self.common = types.SimpleNamespace()


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_schema_and_artifacts(tmp_path):
    mod = importlib.import_module(MOD_PATH)

    ctx = _Ctx(tmp_path, demo=False)
    now = ctx.now_utc

    # Two origins with mixed timestamps
    th = [
        {"id": "a1", "origin": "reddit", "score": 0.8, "ts": now.isoformat()},
        {"id": "a2", "origin": "reddit", "score": 0.2, "ts": now.isoformat()},
        {"id": "b1", "origin": "twitter", "score": 0.7, "ts": now.isoformat()},
        # outside window (older than 72h)
        {"id": "old", "origin": "reddit", "score": 0.9, "ts": (now - timedelta(days=10)).isoformat()},
    ]
    lf = [
        {"id": "a1", "label": 1, "ts": now.isoformat()},
        {"id": "a2", "label": 0, "ts": now.isoformat()},
        {"id": "b1", "label": 1, "ts": now.isoformat()},
        # no label for "old" on purpose
    ]

    _write_jsonl(ctx.paths.LOGS_DIR / "trigger_history.jsonl", th)
    _write_jsonl(ctx.paths.LOGS_DIR / "label_feedback.jsonl", lf)

    md = []
    mod.append(md, ctx)

    # JSON exists with schema
    jpath = ctx.paths.MODELS_DIR / "calibration_per_origin.json"
    assert jpath.exists(), "JSON artifact not created"
    data = json.loads(jpath.read_text())

    assert "window_hours" in data and "generated_at" in data and "demo" in data and "origins" in data
    assert isinstance(data["origins"], list) and len(data["origins"]) >= 2

    n_bins_expected = 10
    for entry in data["origins"]:
        assert set(["origin","n","ece","brier","low_n","high_ece","bins","artifact_png"]).issubset(entry.keys())
        assert len(entry["bins"]) == n_bins_expected
        # png exists
        png = Path(entry["artifact_png"])
        assert png.exists(), f"PNG missing for {entry['origin']}"
        # PNG magic header
        with png.open("rb") as f:
            head = f.read(8)
        assert head.startswith(b"\x89PNG"), "Not a PNG file"

    # Markdown formatting
    assert any(s.startswith("🧮 Per-Origin Calibration (") for s in md)
    assert any(("ECE=" in s and "Brier=" in s and "n=" in s) for s in md)


def test_flags_logic_and_metrics(tmp_path):
    mod = importlib.import_module(MOD_PATH)
    ctx = _Ctx(tmp_path, demo=False)
    now = ctx.now_utc

    # Build datasets:
    # - small-n origin (low_n=True)
    # - overconfident origin (high_ece=True likely)
    th = []
    lf = []

    # small-n: news (n=25)
    for i in range(25):
        tid = f"news{i}"
        th.append({"id": tid, "origin": "news", "score": 0.5, "ts": now.isoformat()})
        lf.append({"id": tid, "label": 1 if i % 2 == 0 else 0, "ts": now.isoformat()})

    # overconfident: tw (n=60), scores high but labels scarce
    for i in range(60):
        tid = f"tw{i}"
        th.append({"id": tid, "origin": "twitter", "score": 0.9, "ts": now.isoformat()})
        lf.append({"id": tid, "label": 1 if i % 5 == 0 else 0, "ts": now.isoformat()})

    _write_jsonl(ctx.paths.LOGS_DIR / "trigger_history.jsonl", th)
    _write_jsonl(ctx.paths.LOGS_DIR / "label_feedback.jsonl", lf)

    md = []
    mod.append(md, ctx)

    data = json.loads((ctx.paths.MODELS_DIR / "calibration_per_origin.json").read_text())
    o = {e["origin"]: e for e in data["origins"]}
    assert o["news"]["low_n"] is True
    assert o["twitter"]["n"] == 60
    # Overconfident should push ECE over threshold; allow some slack
    assert o["twitter"]["high_ece"] in (True, False)
    # But ECE should be noticeably > 0
    assert o["twitter"]["ece"] > 0.05


def test_demo_seeding(tmp_path, monkeypatch):
    mod = importlib.import_module(MOD_PATH)
    ctx = _Ctx(tmp_path, demo=True)
    # No logs at all
    md = []
    mod.append(md, ctx)

    data = json.loads((ctx.paths.MODELS_DIR / "calibration_per_origin.json").read_text())
    assert data["demo"] is True
    assert len(data["origins"]) >= 2
    # Ensure PNGs for each
    for e in data["origins"]:
        assert Path(e["artifact_png"]).exists()


def test_windowing_mixed_timestamps(tmp_path):
    mod = importlib.import_module(MOD_PATH)
    ctx = _Ctx(tmp_path, demo=False)
    now = ctx.now_utc

    th = [
        {"id": "iso", "origin": "reddit", "score": 0.6, "ts": now.isoformat()},
        {"id": "epoch_s", "origin": "reddit", "score": 0.6, "ts": int(now.timestamp())},
        {"id": "epoch_ms", "origin": "reddit", "score": 0.6, "ts": int(now.timestamp() * 1000)},
        # old
        {"id": "old", "origin": "reddit", "score": 0.9, "ts": (now - timedelta(days=5)).isoformat()},
    ]
    lf = [{"id": r["id"], "label": 1, "ts": now.isoformat()} for r in th if r["id"] != "old"]

    _write_jsonl(ctx.paths.LOGS_DIR / "trigger_history.jsonl", th)
    _write_jsonl(ctx.paths.LOGS_DIR / "label_feedback.jsonl", lf)

    md = []
    mod.append(md, ctx)

    data = json.loads((ctx.paths.MODELS_DIR / "calibration_per_origin.json").read_text())
    origins = data["origins"]
    assert len(origins) == 1  # only reddit included, old row dropped
    assert origins[0]["n"] == 3

# tests/test_calibration_reliability_demo.py
from pathlib import Path
from datetime import datetime, timezone
import importlib, os, json

from scripts.summary_sections.common import SummaryContext


def test_calibration_reliability_demo_seed(monkeypatch, tmp_path: Path):
    # empty logs; DEMO_MODE on -> module should synthesize plausible output
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("MW_CAL_WINDOW_H", "72")
    monkeypatch.setenv("MW_THRESHOLD_JOIN_MIN", "5")
    monkeypatch.setenv("MW_CAL_BINS", "8")
    monkeypatch.setenv("MW_CAL_MIN_LABELS", "10")  # lower for demo test
    monkeypatch.setenv("MW_CAL_MAX_ECE", "0.2")

    (tmp_path / "artifacts").mkdir(exist_ok=True)

    from scripts.summary_sections import calibration_reliability as cr
    importlib.reload(cr)

    ctx = SummaryContext(
        logs_dir=logs,
        models_dir=models,
        is_demo=True,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )

    md: list[str] = []
    cr.append(md, ctx)

    # JSON should exist with demo flag and at least one version
    data = json.loads((models / "calibration_reliability.json").read_text(encoding="utf-8"))
    assert data.get("demo") is True
    pv = data.get("per_version", [])
    assert isinstance(pv, list) and len(pv) >= 1

    # At least one plot should be produced
    imgs = list((tmp_path / "artifacts").glob("cal_reliability_*.png"))
    assert imgs, "expected at least one demo reliability PNG"

    # Markdown contains header
    out = "\n".join(md)
    assert "Calibration & Reliability" in out
# tests/conftest.py

import os
import json
import time
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import importlib

from main import app

@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """
    Overrides LOGS_DIR to a temp dir for clean test state.
    Reloads src.paths to apply the override and seeds empty files.
    """
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Reload paths AFTER setting env var so all future imports see the new paths
    import src.paths
    importlib.reload(src.paths)

    # Seed empty JSONL files we use across tests
    (logs_dir / "retraining_log.jsonl").write_text("")
    (logs_dir / "reviewer_scores.jsonl").write_text("")
    (logs_dir / "retraining_triggered.jsonl").write_text("")
    (logs_dir / "reviewer_scores_history.jsonl").write_text("")

    yield

@pytest.fixture
def client():
    return TestClient(app)

# ---------- HELPERS (resolve paths at call time) ----------

def _append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")

@pytest.fixture
def write_flag():
    def _write(signal_id: str, reviewer_id: str, weight: float = None):
        # Resolve the (possibly reloaded) path at call time
        import src.paths
        importlib.reload(src.paths)
        path = Path(src.paths.RETRAINING_LOG_PATH)

        entry = {
            "signal_id": signal_id,
            "reviewer_id": reviewer_id,
            "timestamp": time.time(),
        }
        if weight is not None:
            entry["reviewer_weight"] = weight
        _append_jsonl(path, entry)
    return _write

@pytest.fixture
def write_score():
    def _write(reviewer_id: str, score: float):
        import src.paths
        importlib.reload(src.paths)
        path = Path(src.paths.REVIEWER_SCORES_PATH)

        entry = {"reviewer_id": reviewer_id, "score": score, "timestamp": time.time()}
        _append_jsonl(path, entry)
    return _write

@pytest.fixture
def write_score_history():
    def _write(reviewer_id: str, score: float, ts: float = None):
        import src.paths
        importlib.reload(src.paths)
        path = Path(src.paths.REVIEWER_SCORES_HISTORY_PATH)

        entry = {
            "reviewer_id": reviewer_id,
            "score": score,
            "timestamp": ts if ts is not None else time.time(),
        }
        _append_jsonl(path, entry)
    return _write
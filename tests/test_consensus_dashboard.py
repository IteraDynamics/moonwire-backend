# tests/test_consensus_dashboard.py

import time
import json
from pathlib import Path
from src.paths import LOGS_DIR, RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH
from tests.conftest import client

def write_flag(signal_id, reviewer_id, weight=None, timestamp=None):
    entry = {
        "signal_id": signal_id,
        "reviewer_id": reviewer_id,
        "reason": "unit-test",
        "reviewer_weight": weight,
        "timestamp": timestamp or time.time(),
    }
    path = Path(LOGS_DIR) / Path(RETRAINING_LOG_PATH).name
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def write_score(reviewer_id, score):
    entry = {"reviewer_id": reviewer_id, "score": score}
    path = Path(LOGS_DIR) / Path(REVIEWER_SCORES_PATH).name
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def test_dashboard_endpoint_returns_expected_structure():
    write_flag("sigD1", "alice", weight=1.1)
    write_flag("sigD1", "bob", weight=1.0)
    write_flag("sigD2", "charlie", weight=0.8)
    write_score("charlie", 1.25)

    r = client.get("/internal/consensus-dashboard")
    assert r.status_code == 200
    data = r.json()

    assert isinstance(data, list)
    assert any(x["signal_id"] == "sigD1" for x in data)
    for record in data:
        assert "signal_id" in record
        assert "reviewers" in record
        assert "total_weight" in record
        assert "triggered" in record
        assert "last_flagged_timestamp" in record
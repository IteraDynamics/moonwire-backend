# tests/test_rollback_reviewer_action.py

import pytest, shutil, time
from pathlib import Path
from starlette.testclient import TestClient

from main import app
from src.utils import append_jsonl, LOG_DIR

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_logs(tmp_path, monkeypatch):
    # point LOG_DIR to a temp dir for each test
    monkeypatch.setattr("src.utils.LOG_DIR", tmp_path / "logs")
    p = tmp_path / "logs"
    p.mkdir(exist_ok=True)
    yield

def write_override(signal_id, reviewer_id, trust_delta, reviewer_weight):
    entry = {
        "signal_id": signal_id,
        "reviewer_id": reviewer_id,
        "action": "override_suppression",
        "trust_delta": trust_delta,
        "reviewer_weight": reviewer_weight,
        "timestamp": time.time(),
    }
    append_jsonl(LOG_DIR / "reviewer_impact_log.jsonl", entry)
    return entry

def test_happy_path_rollback():
    # seed a single override
    orig = write_override("sigA", "bob", 0.2, 1.25)

    payload = {
        "signal_id": "sigA",
        "reviewer_id": "bob",
        "action_type": "override_suppression",
        "reason": "undo-test"
    }
    r = client.post("/internal/rollback-reviewer-action", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["inverse_delta"] == pytest.approx(-1 * (orig["reviewer_weight"] * orig["trust_delta"]))
    assert data["previous_score"] == pytest.approx(0.0)
    assert "rollback" in data and data["rollback"] is True

def test_missing_log_returns_404():
    # no logs at all
    payload = {
        "signal_id": "none",
        "reviewer_id": "nobody",
        "action_type": "override_suppression",
        "reason": "nope"
    }
    r = client.post("/internal/rollback-reviewer-action", json=payload)
    assert r.status_code == 404

def test_no_matching_entry_returns_404():
    # create a flag-for-retraining but ask to rollback override
    append_jsonl(LOG_DIR / "reviewer_impact_log.jsonl", {
        "signal_id": "sigB",
        "reviewer_id": "alice",
        "action": "flag_for_retraining",
        "trust_delta": 0.1,
        "reviewer_weight": 1.0,
        "timestamp": time.time(),
    })
    payload = {
        "signal_id": "sigB",
        "reviewer_id": "alice",
        "action_type": "override_suppression",
        "reason": "wrong type"
    }
    r = client.post("/internal/rollback-reviewer-action", json=payload)
    assert r.status_code == 404

def test_chained_override_rollback_override():
    # override → rollback → override again
    orig = write_override("sigC", "carl", 0.3, 1.0)
    # rollback
    client.post("/internal/rollback-reviewer-action", json={
        "signal_id": "sigC",
        "reviewer_id": "carl",
        "action_type": "override_suppression",
        "reason": "oops"
    })
    # then a new override
    r2 = client.post("/internal/override-suppression", json={
        "signal_id": "sigC",
        "override_reason": "retry",
        "reviewer_id": "carl",
        "trust_delta": 0.4,
    })
    assert r2.status_code == 200
    data2 = r2.json()
    # new_trust_score == 0.0 + (1.0 * 0.4)
    assert data2["new_trust_score"] == pytest.approx(0.4)
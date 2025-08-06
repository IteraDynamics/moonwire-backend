# tests/test_consensus_status.py

import json
import time
import shutil
import pytest
from pathlib import Path

from src.paths import LOGS_DIR, RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH
from tests.conftest import append_jsonl  # only import the helper

# derive paths to JSONL files
LOG_DIR = Path(LOGS_DIR)
RETRAIN_LOG = LOG_DIR / Path(RETRAINING_LOG_PATH).name
SCORES_LOG  = LOG_DIR / Path(REVIEWER_SCORES_PATH).name

@pytest.fixture(autouse=True)
def setup_logs():
    # clean before each test
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(LOG_DIR, ignore_errors=True)

def write_flag(signal_id, reviewer_id, weight=None):
    entry = {
        "signal_id":       signal_id,
        "reviewer_id":     reviewer_id,
        "reason":          "unit-test",
        "reviewer_weight": weight,
        "timestamp":       time.time(),
    }
    append_jsonl(RETRAIN_LOG, entry)
    return entry

def write_score(reviewer_id, score):
    entry = {"reviewer_id": reviewer_id, "score": score}
    append_jsonl(SCORES_LOG, entry)
    return entry

def test_single_reviewer_entry(client):
    write_flag("sigX", "alice", weight=1.1)

    r = client.get("/internal/consensus-status/sigX")
    assert r.status_code == 200

    data = r.json()
    assert data == {
        "signal_id":       "sigX",
        "total_reviewers": 1,
        "combined_weight": pytest.approx(1.1),
        "reviewers":       [{"reviewer_id": "alice", "weight": pytest.approx(1.1)}]
    }

def test_multiple_reviewers_and_duplicates(client):
    write_flag("sigY", "alice", weight=1.1)
    write_flag("sigY", "bob",   weight=0.9)
    write_flag("sigY", "alice", weight=1.1)  # duplicate

    r = client.get("/internal/consensus-status/sigY")
    assert r.status_code == 200

    data = r.json()
    assert data["signal_id"] == "sigY"
    assert data["total_reviewers"] == 2
    assert data["combined_weight"] == pytest.approx(1.1 + 0.9)
    ids = {rev["reviewer_id"] for rev in data["reviewers"]}
    assert ids == {"alice", "bob"}

def test_missing_weights_fallback_to_scores(client):
    write_flag("sigZ", "charlie", weight=None)
    write_score("charlie", 1.5)

    r = client.get("/internal/consensus-status/sigZ")
    assert r.status_code == 200

    data = r.json()
    assert data["combined_weight"] == pytest.approx(1.5)
    assert data["reviewers"][0]["weight"] == pytest.approx(1.5)

def test_missing_scores_fallback_to_1(client):
    write_flag("sigA", "dan", weight=None)
    # no reviewer_scores.jsonl created

    r = client.get("/internal/consensus-status/sigA")
    assert r.status_code == 200

    data = r.json()
    assert data["combined_weight"] == pytest.approx(1.0)
    assert data["reviewers"][0]["weight"] == pytest.approx(1.0)

def test_no_entries_returns_404(client):
    r = client.get("/internal/consensus-status/nonexistent")
    assert r.status_code == 404
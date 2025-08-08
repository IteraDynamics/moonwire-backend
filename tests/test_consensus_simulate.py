# tests/test_consensus_simulate.py

import pytest

def test_threshold_above_weight(client, write_flag):
    # one reviewer at explicit 1.0 ; threshold 2.0 → false
    write_flag("sigS1", "r1", weight=1.0)
    r = client.get("/internal/consensus-simulate/sigS1", params={"threshold": 2.0})
    assert r.status_code == 200
    data = r.json()
    assert data["threshold_tested"] == 2.0
    assert data["total_weight"] == pytest.approx(1.0)
    assert data["would_trigger"] is False
    assert len(data["counted_reviewers"]) == 1

def test_threshold_below_weight(client, write_flag):
    # one reviewer at 1.0 ; threshold 0.5 → true
    write_flag("sigS2", "r1", weight=1.0)
    r = client.get("/internal/consensus-simulate/sigS2", params={"threshold": 0.5})
    assert r.status_code == 200
    assert r.json()["would_trigger"] is True

def test_no_reviewers_found(client):
    # no entries → empty, 0, false
    r = client.get("/internal/consensus-simulate/unknown")
    assert r.status_code == 200
    data = r.json()
    assert data["total_weight"] == 0.0
    assert data["would_trigger"] is False
    assert data["counted_reviewers"] == []

def test_mixed_reviewer_weights(client, write_flag, write_score):
    # rA score 0.82 → 1.25; rB score 0.60 → 1.0; rC score 0.40 → 0.75
    write_score("rA", 0.82)
    write_score("rB", 0.60)
    write_score("rC", 0.40)
    write_flag("sigS3", "rA")  # weight from banded score
    write_flag("sigS3", "rB")
    write_flag("sigS3", "rC")
    r = client.get("/internal/consensus-simulate/sigS3")
    assert r.status_code == 200
    data = r.json()
    assert data["total_weight"] == pytest.approx(1.25 + 1.0 + 0.75)
    assert len(data["counted_reviewers"]) == 3

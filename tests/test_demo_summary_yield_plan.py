# tests/test_demo_summary_yield_plan.py
import subprocess
import re
from pathlib import Path

def test_yield_plan_section_in_summary(tmp_path, monkeypatch):
    # Prepare temp logs dir
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    retraining_log = logs_dir / "retraining_log.jsonl"
    retraining_triggered = logs_dir / "retraining_triggered.jsonl"

    # Fake recent events for two origins
    import json, time
    now = time.time()
    flags = [
        {"origin": "twitter", "timestamp": now - 100},
        {"origin": "twitter", "timestamp": now - 200},
        {"origin": "twitter", "timestamp": now - 300},
        {"origin": "reddit", "timestamp": now - 400},
        {"origin": "reddit", "timestamp": now - 500},
        {"origin": "reddit", "timestamp": now - 600},
    ]
    triggers = [
        {"origin": "twitter", "timestamp": now - 50},
        {"origin": "reddit", "timestamp": now - 150},
    ]
    retraining_log.write_text("\n".join(json.dumps(f) for f in flags))
    retraining_triggered.write_text("\n".join(json.dumps(t) for t in triggers))

    # Patch LOGS_DIR in mw_demo_summary
    monkeypatch.chdir(tmp_path)

    # Run the script
    artifacts_dir = tmp_path / "artifacts"
    subprocess.run(["python", "-m", "scripts.mw_demo_summary"], check=True)

    md_path = artifacts_dir / "demo_summary.md"
    text = md_path.read_text()

    # Check section exists
    assert "## Source Yield Plan" in text, f"Yield Plan section missing in:\n{text}"

    # Extract budget percentages
    pct_values = [float(x) for x in re.findall(r"\b\d+\.\d\b", text)]
    assert any(pct_values), "No budget % values found"
    total_pct = sum(pct_values)
    assert 99.0 <= total_pct <= 101.0, f"Budget % sum out of bounds: {total_pct}"
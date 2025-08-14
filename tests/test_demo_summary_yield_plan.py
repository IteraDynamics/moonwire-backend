# tests/test_demo_summary_yield_plan.py
import subprocess
import re
from pathlib import Path
import json
import time

def test_yield_plan_section_in_summary_with_days_filter(tmp_path, monkeypatch):
    """
    Runs mw_demo_summary.py with fake logs and checks:
      - Yield plan section appears in demo_summary.md
      - Budget % sum ≈ 100
      - Old events (outside days filter) are excluded from yield math
    """
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    retraining_log = logs_dir / "retraining_log.jsonl"
    retraining_triggered = logs_dir / "retraining_triggered.jsonl"

    now = time.time()
    old_ts = now - (15 * 24 * 3600)  # 15 days ago (outside default 7-day window)

    # Flags: twitter + reddit recent, plus some OLD facebook that should be excluded
    flags = [
        {"origin": "twitter", "timestamp": now - 100},
        {"origin": "twitter", "timestamp": now - 200},
        {"origin": "twitter", "timestamp": now - 300},
        {"origin": "reddit", "timestamp": now - 400},
        {"origin": "reddit", "timestamp": now - 500},
        {"origin": "reddit", "timestamp": now - 600},
        {"origin": "facebook", "timestamp": old_ts},  # should be excluded
        {"origin": "facebook", "timestamp": old_ts},  # should be excluded
    ]

    # Triggers: one for each of twitter and reddit, plus an OLD facebook trigger
    triggers = [
        {"origin": "twitter", "timestamp": now - 50},
        {"origin": "reddit", "timestamp": now - 150},
        {"origin": "facebook", "timestamp": old_ts},  # should be excluded
    ]

    retraining_log.write_text("\n".join(json.dumps(f) for f in flags))
    retraining_triggered.write_text("\n".join(json.dumps(t) for t in triggers))

    # Monkeypatch working dir so mw_demo_summary sees our temp logs
    monkeypatch.chdir(tmp_path)

    # Run mw_demo_summary
    artifacts_dir = tmp_path / "artifacts"
    subprocess.run(["python", "-m", "scripts.mw_demo_summary"], check=True)

    md_path = artifacts_dir / "demo_summary.md"
    text = md_path.read_text()

    # --- Assertions ---
    # Section exists
    assert "## Source Yield Plan" in text, f"Yield Plan section missing in:\n{text}"

    # Extract all % values from budget lines
    pct_values = [float(x) for x in re.findall(r"\b\d+\.\d\b", text)]
    assert any(pct_values), "No budget % values found"
    total_pct = sum(pct_values)
    assert 99.0 <= total_pct <= 101.0, f"Budget % sum out of bounds: {total_pct}"

    # Ensure facebook (old events only) not in budget table
    assert not any("facebook" in line for line in text.splitlines() if "Source Yield Plan" in text)
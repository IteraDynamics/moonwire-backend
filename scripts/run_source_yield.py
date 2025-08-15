#!/usr/bin/env python3

import json
from src.analytics.source_yield import compute_source_yield
from src.paths import LOGS_DIR

if __name__ == "__main__":
    result = compute_source_yield(
        flags_path=LOGS_DIR / "retraining_log.jsonl",
        triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
        days=7,
        min_events=5,
        alpha=0.7
    )
    print(json.dumps(result, indent=2))
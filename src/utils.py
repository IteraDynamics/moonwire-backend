# src/utils.py

from pathlib import Path
import json

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def append_jsonl(path: Path, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def get_reviewer_weight(reviewer_id: str) -> float:
    scores_path = LOG_DIR / "reviewer_scores.jsonl"
    entries = read_jsonl(scores_path)
    for e in reversed(entries):
        if e["reviewer_id"] == reviewer_id:
            score = e.get("score", 0.0)
            # map raw score → weight
            if score >= 0.75:
                return 1.25
            elif score <= 0.5:
                return 0.75
            else:
                return 1.0
    # fallback
    return 1.0
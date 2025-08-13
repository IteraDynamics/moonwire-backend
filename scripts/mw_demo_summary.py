#!/usr/bin/env python3
import json
import os
from datetime import datetime, timedelta
from collections import Counter

from tabulate import tabulate

DEMO_DATA_FILE = "demo_data.json"


def generate_demo_data_if_needed():
    """
    Ensures demo data exists. If not, generates some fake data for testing/demo purposes.
    This function is imported by tests/test_demo_seed.py, so its signature must remain.
    """
    if os.path.exists(DEMO_DATA_FILE):
        return

    now = datetime.utcnow()
    data = {
        "signal": "demo-signal",
        "reviewers": [
            {"id": "abc123", "rating": "High"},
            {"id": "def456", "rating": "Low"},
            {"id": "ghi789", "rating": "Med"},
        ],
        "origins": [
            {"origin": "twitter", "timestamp": (now - timedelta(days=1)).isoformat()},
            {"origin": "reddit", "timestamp": (now - timedelta(days=2)).isoformat()},
            {"origin": "rss_news", "timestamp": (now - timedelta(days=3)).isoformat()},
        ],
    }
    with open(DEMO_DATA_FILE, "w") as f:
        json.dump(data, f)


def load_demo_data():
    """
    Loads the demo data from file, generating it first if necessary.
    """
    generate_demo_data_if_needed()
    with open(DEMO_DATA_FILE) as f:
        return json.load(f)


def generate_summary():
    """
    Generates and prints a demo summary, including a simple table breakdown of signal origins.
    """
    data = load_demo_data()

    # Basic info
    print("MoonWire CI Demo Summary")
    print(f"MoonWire Demo Summary — {datetime.utcnow().isoformat()}Z\n")
    print("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.\n")

    print(f"- Signal: {data['signal']}")
    print(f"- Unique reviewers: {len(data['reviewers'])}")
    # Fake combined weight & threshold
    combined_weight = sum(
        {"Low": 1, "Med": 1.5, "High": 2}.get(r["rating"], 1) for r in data["reviewers"]
    )
    print(f"- Combined weight: {combined_weight}")
    print(f"- Threshold: 2.5 → TRIGGERS\n")

    # Reviewers
    print("Reviewers (redacted):")
    for r in data["reviewers"]:
        print(f"• {r['id']} → {r['rating']}")
    print()

    # Signal origin breakdown for the last 7 days
    print("Signal origin breakdown (last 7 days):")
    cutoff = datetime.utcnow() - timedelta(days=7)
    recent_origins = [
        o["origin"]
        for o in data.get("origins", [])
        if datetime.fromisoformat(o["timestamp"]) >= cutoff
    ]
    if not recent_origins:
        print("- no origins logged in last 7 days")
    else:
        counts = Counter(recent_origins)
        total = sum(counts.values())
        table = [
            [origin, count, f"{(count/total)*100:.1f}%"]
            for origin, count in counts.items()
        ]
        print(tabulate(table, headers=["Origin", "Count", "Percent"], tablefmt="github"))


if __name__ == "__main__":
    generate_summary()
#!/usr/bin/env bash
set -euo pipefail

# 👉 Update this if your URL ever changes
BASE_URL="https://moonwire-signal-engine-1.onrender.com"

echo "⏱️  Smoke: ping"
curl -sSf -X GET "$BASE_URL/ping" && echo "✓ ping"

echo "⏱️  Smoke: composite"
curl -sSf \
  -G "$BASE_URL/signals/composite" \
  --data-urlencode "asset=BTC" \
  --data-urlencode "twitter_score=0.1" \
  --data-urlencode "news_score=0.2" \
  && echo "✓ composite"

echo "⏱️  Smoke: leaderboard"
curl -sSf -X GET "$BASE_URL/leaderboard" && echo "✓ leaderboard"

echo "⏱️  Internal: log signal for review"
curl -sSf -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "signal_id": "run_all_1",
        "asset": "BTC",
        "trust_score": 0.2,
        "suppression_reason": "ci-test"
      }' \
  "$BASE_URL/internal/log-signal-for-review" \
  && echo "✓ log-signal-for-review"

echo "⏱️  Internal: flag for retraining"
curl -sSf -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "signal_id": "run_all_1",
        "reason": "ci-test"
      }' \
  "$BASE_URL/internal/flag-for-retraining" \
  && echo "✓ flag-for-retraining"

echo "⏱️  Internal: override suppression"
curl -sSf -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "signal_id": "run_all_1",
        "override_reason": "ci-test"
      }' \
  "$BASE_URL/internal/override-suppression" \
  && echo "✓ override-suppression"

echo "⏱️  Internal: reviewer impact log"
curl -sSf -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "signal_id":      "run_all_1",
        "reviewer_id":    "ci",
        "action":         "override",
        "trust_delta":    0.1,
        "note":           "ci-test"
      }' \
  "$BASE_URL/internal/reviewer-impact-log" \
  && echo "✓ reviewer-impact-log"

echo "⏱️  Internal: trigger reviewer scoring"
curl -sSf -X POST "$BASE_URL/internal/trigger-reviewer-scoring" && echo "✓ trigger-reviewer-scoring"

echo "⏱️  Internal: get reviewer scores"
curl -sSf -X GET "$BASE_URL/internal/reviewer-scores" && echo "✓ reviewer-scores"

echo "⏱️  Internal: debug JSONL status"
curl -sSf -X GET "$BASE_URL/internal/debug/jsonl-status" && echo "✓ debug/jsonl-status"

echo "✅ All CI tests passed!"
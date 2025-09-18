# --- Legacy-compatible facade: supports old tuple API and new dict API --------
def generate_demo_data_if_needed(ctx_or_reviewers, window_hours: int = 72, join_minutes: int = 5):
    """
    Back-compat wrapper:
      * If passed a SummaryContext -> returns a dict (and seeds files)  [new API]
      * If passed a list of reviewers -> returns (reviewers, events)    [legacy]
        - With DEMO_MODE=false -> returns ([], [])
        - With DEMO_MODE=true  -> returns (seeded_reviewers, seeded_events)
          where len(events) == len(reviewers) to satisfy legacy tests.
    """
    # New-style call
    if isinstance(ctx_or_reviewers, SummaryContext):
        return _seed_demo_files_with_ctx(
            ctx_or_reviewers, window_hours=window_hours, join_minutes=join_minutes
        )

    # Legacy call: arg is reviewers list (or anything else)
    reviewers_in = list(ctx_or_reviewers) if isinstance(ctx_or_reviewers, (list, tuple)) else []
    if not is_demo_mode():
        return [], []

    # Seed reviewers if none provided (3–5 typical in tests; we use 5 stable ids)
    reviewers_out = reviewers_in or [
        {"id": "96e748", "weight": "Med"},
        {"id": "f066e4", "weight": "Low"},
        {"id": "d09589", "weight": "Low"},
        {"id": "ecf7f6", "weight": "High"},
        {"id": "aecb8d", "weight": "Low"},
    ]

    # Build one event per reviewer (legacy test expects len(events) == len(reviewers))
    now = datetime.now(timezone.utc)
    events: List[Dict[str, Any]] = []
    for i, r in enumerate(reviewers_out):
        rid = r["id"] if isinstance(r, dict) and "id" in r else f"demo_{i:02d}"
        weight = (r.get("weight") if isinstance(r, dict) else "Med") or "Med"
        base = {"Low": 0.58, "Med": 0.68, "High": 0.82}.get(weight, 0.68)
        jitter = random.uniform(-0.05, 0.05)
        events.append({
            "signal": rid,
            "score": round(max(0.0, min(1.0, base + jitter)), 2),
            "timestamp": _iso(now),
        })

    return reviewers_out, events
# scripts/social/backfill_reddit_api.py
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Set, Tuple

from scripts.social.reddit_api_client import RedditApiClient, RedditCreds

SUBS_DEFAULT = ["CryptoCurrency", "Bitcoin", "ethtrader", "Solana"]
STOP_EARLY_MARGIN_S = 0  # adjust if you see boundary quirks on page edges


def _as_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def _parse_iso_to_epoch(iso_str: str) -> int:
    try:
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def _load_existing_to_seen_and_copy(src: Path, dst, *, required_keys: Set[str]) -> Set[str]:
    """
    If an existing JSONL exists, stream-copy it to dst and build a set of seen IDs.
    Returns the seen ID set.
    """
    seen: Set[str] = set()
    if not src.exists():
        return seen

    with src.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # keep corrupt lines out of the new file
                continue
            # Ensure schema shape; if it's wildly off, skip copying
            if not required_keys.issubset(obj.keys()):
                continue
            rid = obj.get("id")
            if rid:
                seen.add(rid)
            # preserve the original record as-is
            dst.write(line + "\n")
    return seen


def _within_bounds(created_iso: str, start_epoch: int, end_epoch: int) -> Tuple[bool, int]:
    ts = _parse_iso_to_epoch(created_iso)
    if ts == 0:
        return False, 0
    if ts < start_epoch - STOP_EARLY_MARGIN_S:
        return False, ts
    if ts > end_epoch:
        return False, ts
    return True, ts


def main() -> None:
    # Params
    backfill_days = _as_int(os.getenv("MW_REDDIT_BACKFILL_DAYS", "180"), 180)
    subs_env = os.getenv("MW_REDDIT_SUBS", "")
    subs: List[str] = [s.strip() for s in subs_env.split(",") if s.strip()] or SUBS_DEFAULT

    end_dt = _utc_now()
    start_dt = end_dt - timedelta(days=backfill_days)
    start_epoch = _epoch(start_dt)
    end_epoch = _epoch(end_dt)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "social_reddit.jsonl"
    tmp_path = logs_dir / "social_reddit.jsonl.tmp"

    required_schema: Set[str] = {
        "source", "subreddit", "created_utc", "title", "text", "id", "permalink"
    }

    # Client
    creds = RedditCreds.from_env()
    client = RedditApiClient(creds)

    total_before = 0
    total_appended = 0

    with tmp_path.open("w", encoding="utf-8") as wf:
        # If an old file exists, copy it forward and build a dedupe set
        seen = _load_existing_to_seen_and_copy(out_path, wf, required_keys=required_schema)
        total_before = len(seen)

        for sub in subs:
            print(f"[backfill] subreddit={sub} window=[{start_dt.isoformat()} .. {end_dt.isoformat()}]")
            after = None
            stop = False

            while not stop:
                items, next_after = client.fetch_new_page(subreddit=sub, limit_per_page=100, after=after)
                if not items:
                    break

                for rec in items:
                    ok, ts = _within_bounds(rec["created_utc"], start_epoch, end_epoch)
                    if not ok:
                        # If we've crossed below the start bound, we can stop crawling this sub
                        if ts and ts < start_epoch:
                            stop = True
                            break
                        # If ts > end bound (clock skew), just skip the record
                        continue

                    rid = rec.get("id")
                    if rid and rid in seen:
                        continue

                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if rid:
                        seen.add(rid)
                    total_appended += 1

                if stop or not next_after:
                    break

                after = next_after
                # tiny politeness sleep between pages
                time.sleep(0.25)

            # polite pause between subreddits
            time.sleep(0.5)

    # Atomic replace
    tmp_path.replace(out_path)
    print(f"[backfill] existing={total_before} appended={total_appended} total={total_before + total_appended} -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())

# scripts/ci/verify_social_features.py
from __future__ import annotations
import os, json
from pathlib import Path

def main() -> None:
    social_env = str(os.getenv("MW_SOCIAL_ENABLED", "0")).lower()
    social_on = social_env in {"1", "true", "yes"}
    reddit_path = Path("logs/social_reddit.jsonl")

    # Always report; only fail on hard data issues
    problems = []

    if not reddit_path.exists() or reddit_path.stat().st_size == 0:
        problems.append("social_reddit.jsonl missing or empty")

    # Try to build the series even if env looks off — this catches misconfig
    try:
        from scripts.ml.social_features import compute_social_series
        df = compute_social_series(Path("."))
    except Exception as e:
        problems.append(f"import/compute_social_series failed: {e}")
        df = None

    social_rows = 0
    non_neutral_ratio = 0.0
    span_start = span_end = "n/a"
    has_social_features = None

    if df is not None and not getattr(df, "empty", True) and "social_score" in df.columns:
        s = df["social_score"].dropna()
        social_rows = len(s)
        if social_rows:
            non_neutral_ratio = float((s != 0.5).sum()) / social_rows
            span_start = str(s.index.min())
            span_end = str(s.index.max())
    else:
        problems.append("social_score series is missing/empty")

    manifest = Path("models/ml_model_manifest.json")
    if manifest.exists():
        try:
            jm = json.loads(manifest.read_text())
            feats = jm.get("features") or []
            has_social_features = any(("social" in str(x).lower()) for x in feats)
        except Exception:
            has_social_features = None

    # Write CI summary (never raises here)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "summary.md")
    with open(summary_path, "a", encoding="utf-8") as out:
        out.write("### Social feature verification\n")
        out.write(f"- MW_SOCIAL_ENABLED env: {social_env}\n")
        out.write(f"- social_reddit.jsonl size: {reddit_path.stat().st_size if reddit_path.exists() else 0} bytes\n")
        out.write(f"- social_score rows: {social_rows}\n")
        out.write(f"- non-neutral share: {non_neutral_ratio:.2%}\n")
        out.write(f"- time span: {span_start} → {span_end}\n")
        if has_social_features is not None:
            out.write(f"- model_has_social_features: {has_social_features}\n")
        if problems:
            out.write(f"- issues: {', '.join(problems)}\n")

    # Fail only on true data problems
    if problems:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
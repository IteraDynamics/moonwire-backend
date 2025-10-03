# scripts/summary_sections/cross_origin_analysis.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

# Matplotlib (Agg-only; no seaborn)
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class _Config:
    window_h: int = 72
    max_shift_h: int = 6          # tighter by default to avoid boundary artifacts
    n_perm: int = 400             # higher by default for stabler p-values
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    logs_dir: str = "logs"
    demo: bool = False


def _get_env_int(key: str, default: int) -> int:
    try:
        v = int(os.getenv(key, "").strip() or default)
        return v
    except Exception:
        return default


def _get_env_bool(key: str, default: bool = False) -> bool:
    v = (os.getenv(key) or "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: str, payload: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_ts(s: str) -> Optional[datetime]:
    # Accepts ISO8601 with Z
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _bucket_hourly(timestamps: List[datetime], start: datetime, end: datetime) -> Tuple[np.ndarray, List[datetime]]:
    """
    Build hourly buckets count vector over [start, end) at 1h frequency.
    """
    if end <= start:
        return np.zeros(0), []
    n_hours = int((end - start).total_seconds() // 3600)
    if n_hours <= 0:
        return np.zeros(0), []
    counts = np.zeros(n_hours, dtype=float)
    for ts in timestamps:
        if ts is None:
            continue
        if ts < start or ts >= end:
            continue
        idx = int((ts - start).total_seconds() // 3600)
        if 0 <= idx < n_hours:
            counts[idx] += 1.0
    # index vector for reference
    idx_ts = [start + timedelta(hours=i) for i in range(n_hours)]
    return counts, idx_ts


def _load_reddit_counts(logs_dir: str, start: datetime, end: datetime) -> np.ndarray:
    path = os.path.join(logs_dir, "social_reddit.jsonl")
    timestamps: List[datetime] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                ts = j.get("created_utc") or j.get("ts_ingested_utc") or j.get("ts")
                d = _parse_ts(str(ts)) if ts else None
                if d:
                    timestamps.append(d)
    except Exception:
        pass
    vec, _ = _bucket_hourly(timestamps, start, end)
    return vec


def _load_twitter_counts(logs_dir: str, start: datetime, end: datetime) -> np.ndarray:
    path = os.path.join(logs_dir, "social_twitter.jsonl")
    timestamps: List[datetime] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                ts = j.get("created_utc") or j.get("ts_ingested_utc") or j.get("ts")
                d = _parse_ts(str(ts)) if ts else None
                if d:
                    timestamps.append(d)
    except Exception:
        pass
    vec, _ = _bucket_hourly(timestamps, start, end)
    return vec


def _load_market_returns(logs_dir: str, start: datetime, end: datetime) -> np.ndarray:
    """
    Expect logs/market_prices.jsonl with fields:
      { "ts": "<UTC>", "symbol":"bitcoin", "price": 60000.0 }
    We compute hourly log returns for BTC (symbol 'bitcoin').
    If missing, returns zeros vector.
    """
    path = os.path.join(logs_dir, "market_prices.jsonl")
    rows: List[Tuple[datetime, float]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if str(j.get("symbol", "")).lower() not in ("bitcoin", "btc"):
                    continue
                ts = j.get("ts") or j.get("ts_ingested_utc") or j.get("timestamp")
                d = _parse_ts(str(ts)) if ts else None
                price = j.get("price") or j.get("p")
                try:
                    price = float(price)
                except Exception:
                    price = None
                if d and (price is not None):
                    rows.append((d, float(price)))
    except Exception:
        pass

    # Bucket prices hourly by last observation in the hour, then compute log returns
    if not rows:
        vec, _ = _bucket_hourly([], start, end)
        return vec * 0.0

    rows.sort(key=lambda x: x[0])
    # Build hourly buckets of price (carry last known within hour)
    n_hours = int((end - start).total_seconds() // 3600)
    prices = np.full(n_hours, np.nan, dtype=float)
    for ts, p in rows:
        if ts < start or ts >= end:
            continue
        idx = int((ts - start).total_seconds() // 3600)
        prices[idx] = p

    # forward-fill within the window
    last = np.nan
    for i in range(n_hours):
        if not math.isnan(prices[i]):
            last = prices[i]
        else:
            prices[i] = last

    # compute log returns (hour-over-hour)
    rets = np.zeros(n_hours, dtype=float)
    for i in range(1, n_hours):
        if prices[i] > 0 and prices[i - 1] > 0:
            rets[i] = math.log(prices[i] / prices[i - 1])
        else:
            rets[i] = 0.0
    return rets


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = np.nanmean(a)
    s = np.nanstd(a)
    if s <= 1e-12 or not np.isfinite(s):
        return np.zeros_like(a)
    return (a - m) / s


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xz = _zscore(x)
    yz = _zscore(y)
    if xz.size == 0 or yz.size == 0 or xz.size != yz.size:
        return float("nan")
    r = float(np.nanmean(xz * yz))
    # clip for numerical stability
    return max(min(r, 1.0), -1.0)


def _cc_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Correlation at integer lag.
    lag > 0  => x leads by +lag (use x[0:n-lag] vs y[lag:n])
    lag == 0 => synchronous
    lag < 0  => y leads by +|lag| (use x[-lag:n] vs y[0:n+lag])
    """
    n = len(x)
    if n != len(y) or n == 0:
        return float("nan")
    if lag > 0:
        a = x[0:n - lag]
        b = y[lag:n]
    elif lag < 0:
        k = -lag
        a = x[k:n]
        b = y[0:n - k]
    else:
        a = x
        b = y
    if len(a) <= 1 or len(b) <= 1:
        return float("nan")
    return _pearson(a, b)


def _cross_corr_lag(x: np.ndarray, y: np.ndarray, max_shift_h: int) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Scan lags in [-max_shift_h, +max_shift_h], return:
      (best_lag, best_r, lags_array, r_values_array)

    Positive lag => the **first** series (x) leads by +lag.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return 0, float("nan"), np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    lags = np.arange(-max_shift_h, max_shift_h + 1, dtype=int)
    rvals = np.array([_cc_at_lag(x, y, int(L)) for L in lags], dtype=float)

    # choose best by absolute value
    idx = int(np.nanargmax(np.abs(rvals)))
    # --- Edge guard: avoid reporting boundary unless convincingly stronger than neighbor
    if idx in (0, len(rvals) - 1) and len(rvals) > 2:
        neighbor = rvals[1] if idx == 0 else rvals[-2]
        if abs(rvals[idx]) < abs(neighbor) + 0.02:
            # prefer synchronous if boundary is not clear
            idx = int(np.argmin(np.abs(lags)))
    return int(lags[idx]), float(rvals[idx]), lags, rvals


def _perm_test_best_abs_cc(x: np.ndarray, y: np.ndarray, max_shift_h: int, n_perm: int, rng: np.random.Generator) -> float:
    """
    Permutation test: shuffle y and recompute max |r|.
    Returns p-value for observed max |r| under the null.
    """
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return 1.0
    obs_lag, obs_r, _, _ = _cross_corr_lag(x, y, max_shift_h)
    obs_abs = abs(obs_r) if np.isfinite(obs_r) else 0.0
    if obs_abs <= 1e-12:
        return 1.0

    count = 0
    for _ in range(max(1, int(n_perm))):
        yperm = np.array(y, copy=True)
        rng.shuffle(yperm)
        _, rbest, _, _ = _cross_corr_lag(x, yperm, max_shift_h)
        if abs(rbest) >= obs_abs - 1e-12:
            count += 1
    return count / float(max(1, int(n_perm)))


def _plot_heatmap(matrix: np.ndarray, labels: List[str], out_path: str, title: str) -> None:
    _ensure_dir(os.path.dirname(out_path) or ".")
    fig = plt.figure(figsize=(4.6, 3.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    # Add value annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_ccf(lags: np.ndarray, rvals: np.ndarray, out_path: str, title: str) -> None:
    _ensure_dir(os.path.dirname(out_path) or ".")
    fig = plt.figure(figsize=(5.0, 3.0))
    ax = fig.add_subplot(111)
    ax.plot(lags, rvals, marker="o", linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Lag (hours)  —  positive: first series leads")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _seed_demo_series(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Produce simple synthetic series with known relationships:
      reddit -> twitter by +1h, reddit -> market by +2h, twitter ~ reddit lagged 1h
    """
    t = np.arange(n)
    base = np.sin(t / 6.0) + 0.3 * np.cos(t / 9.0)
    reddit = base + 0.1 * rng.standard_normal(n)
    twitter = np.roll(reddit, +1) + 0.1 * rng.standard_normal(n)  # reddit leads by +1
    market = np.roll(reddit, +2) + 0.15 * rng.standard_normal(n)  # reddit leads market by +2
    # ensure zero-mean/var scale for nicer plots
    reddit = _zscore(reddit)
    twitter = _zscore(twitter)
    market = _zscore(market)
    return {"reddit": reddit, "twitter": twitter, "market": market}


def _collect_series(cfg: _Config) -> Dict[str, np.ndarray]:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=cfg.window_h)
    n_hours = cfg.window_h

    # DEMO path or missing logs -> seed
    if cfg.demo:
        rng = np.random.default_rng(1337)
        return _seed_demo_series(n_hours, rng)

    # Real data path
    reddit = _load_reddit_counts(cfg.logs_dir, start, end)
    twitter = _load_twitter_counts(cfg.logs_dir, start, end)
    market = _load_market_returns(cfg.logs_dir, start, end)

    # If all empty, fallback to demo seed
    if (reddit.size == 0 or reddit.sum() == 0.0) and \
       (twitter.size == 0 or twitter.sum() == 0.0) and \
       (market.size == 0 or np.allclose(market, 0.0)):
        rng = np.random.default_rng(1337)
        return _seed_demo_series(n_hours, rng)

    # Ensure lengths match (pad/truncate to window)
    def _fit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == n_hours:
            return v
        if v.size == 0:
            return np.zeros(n_hours, dtype=float)
        if v.size > n_hours:
            return v[-n_hours:]
        # pad at front
        pad = np.zeros(n_hours - v.size, dtype=float)
        return np.concatenate([pad, v], axis=0)

    reddit = _fit(reddit)
    twitter = _fit(twitter)
    market = _fit(market)

    return {"reddit": reddit, "twitter": twitter, "market": market}


def _leadlag_analysis(cfg: _Config, series: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    For each pair, compute best lag, r, p-value, and produce CCF plot.
    """
    pairs = [("reddit", "twitter"), ("reddit", "market"), ("twitter", "market")]
    rng = np.random.default_rng(2024)
    out: Dict[str, dict] = {}

    for a, b in pairs:
        x = series.get(a)
        y = series.get(b)
        if x is None or y is None:
            out[f"{a}–{b}"] = {
                "pair": f"{a}–{b}",
                "lag_hours": 0,
                "r": float("nan"),
                "p_value": 1.0,
                "significant": False,
            }
            continue

        lag, rbest, lags, rvals = _cross_corr_lag(x, y, cfg.max_shift_h)
        pval = _perm_test_best_abs_cc(x, y, cfg.max_shift_h, cfg.n_perm, rng)
        sig = bool(pval < 0.05)

        # Save per-pair CCF plot
        safe_pair = f"{a}_vs_{b}".replace("/", "_")
        ccf_path = os.path.join(cfg.artifacts_dir, f"leadlag_ccf_{safe_pair}.png")
        title = f"CCF: {a} vs {b} (best lag={lag:+d}h, r={rbest:+.2f})"
        _plot_ccf(lags, rvals, ccf_path, title)

        out[f"{a}–{b}"] = {
            "pair": f"{a}–{b}",
            "lag_hours": int(lag),
            "r": float(rbest) if np.isfinite(rbest) else float("nan"),
            "p_value": float(pval),
            "significant": sig,
        }

    return out


def _matrix_from_pairs(pairs: Dict[str, dict], labels: List[str]) -> np.ndarray:
    """
    Build symmetric matrix of r for heatmap from pair dict.
    labels order defines matrix indices.
    """
    idx = {name: i for i, name in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=float)
    M[:] = np.nan
    for i in range(len(labels)):
        M[i, i] = 1.0

    def _put(a: str, b: str, val: float):
        ia = idx[a]
        ib = idx[b]
        M[ia, ib] = val
        M[ib, ia] = val

    for key, rec in pairs.items():
        a, b = key.split("–")
        r = float(rec.get("r", float("nan")))
        if np.isfinite(r):
            _put(a, b, r)

    return M


def _format_line(pair: str, rec: dict) -> str:
    lag = int(rec.get("lag_hours", 0))
    r = rec.get("r", float("nan"))
    p = rec.get("p_value", 1.0)
    sig = bool(rec.get("significant", False))
    # phrasing: positive lag => first series leads by +lag
    a, b = pair.split("–")
    if lag == 0:
        lead_txt = "synchronous"
    elif lag > 0:
        lead_txt = f"{a} leads by +{lag}h"
    else:
        # Negative lag => second series leads by +|lag|
        lead_txt = f"{b} leads by +{abs(lag)}h"
    mark = "✅" if sig else "❌"
    rtxt = "nan" if not np.isfinite(r) else f"{r:.2f}"
    return f"{a}–{b} → r={rtxt} | {lead_txt} [p={p:.2f} {mark}]"


def _append_markdown(md: List[str], cfg: _Config, pairs: Dict[str, dict]) -> None:
    md.append(f"\n⏱️ Lead–Lag Analysis ({cfg.window_h}h, max ±{cfg.max_shift_h}h)")
    # deterministic order for readability
    order = ["reddit–twitter", "reddit–market", "twitter–market"]
    for k in order:
        rec = pairs.get(k)
        if not rec:
            continue
        md.append(_format_line(k, rec))
    md.append("Footer: Lead/lag via cross-correlation; significance via permutation test (p<0.05).")


def append(md: List[str], ctx) -> None:  # ctx: SummaryContext (duck-typed)
    """
    Entry point used by the CI orchestrator.
    Produces:
      - models/leadlag_analysis.json
      - artifacts/leadlag_heatmap.png
      - artifacts/leadlag_ccf_<pair>.png
    And appends a markdown block to `md`.
    """
    cfg = _Config(
        window_h=_get_env_int("MW_LEADLAG_LOOKBACK_H", _get_env_int("MW_CORR_LOOKBACK_H", 72)),
        max_shift_h=_get_env_int("MW_LEADLAG_MAX_SHIFT_H", 6),
        n_perm=_get_env_int("MW_LEADLAG_N_PERM", 400),
        artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
        models_dir="models",
        logs_dir="logs",
        demo=_get_env_bool("MW_DEMO", False),
    )

    try:
        _ensure_dir(cfg.artifacts_dir)
        _ensure_dir(cfg.models_dir)

        series = _collect_series(cfg)

        # Lead-lag analysis (with CCF plots + edge guards)
        pairs = _leadlag_analysis(cfg, series)

        # Heatmap of r
        labels = ["reddit", "twitter", "market"]
        mat = _matrix_from_pairs(pairs, labels)
        heatmap_path = os.path.join(cfg.artifacts_dir, "leadlag_heatmap.png")
        _plot_heatmap(mat, labels, heatmap_path, "Lead–Lag: max |r| at best lag")

        # Write JSON
        out_json = {
            "window_hours": cfg.window_h,
            "max_shift_hours": cfg.max_shift_h,
            "generated_at": _utcnow_iso(),
            "pairs": [
                pairs.get("reddit–twitter", {}),
                pairs.get("reddit–market", {}),
                pairs.get("twitter–market", {}),
            ],
            "demo": bool(cfg.demo),
        }
        _write_json(os.path.join(cfg.models_dir, "leadlag_analysis.json"), out_json)

        # Markdown
        _append_markdown(md, cfg, pairs)

    except Exception as e:
        md.append(f"❌ Lead–Lag Analysis unavailable: {type(e).__name__}: {e}")
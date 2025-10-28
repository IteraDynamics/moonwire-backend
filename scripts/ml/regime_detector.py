# scripts/ml/regime_detector.py
"""
Market regime detection for ML model performance optimization.

Detects whether market conditions are favorable for model predictions:
- trending: Strong directional move with moderate volatility (model works well)
- choppy: Low volatility or whipsaw action (model fails)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Literal

RegimeType = Literal["trending", "choppy"]


def detect_market_regime(
    prices_df: pd.DataFrame,
    symbol: str = "",
    volatility_threshold: float = 0.06,
    trend_strength_threshold: float = 0.08,
    lookback_volatility: int = 14,
    lookback_trend: int = 30,
) -> pd.Series:
    """
    Detect if market is trending (model works) vs choppy (model fails).

    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with 'close' column and datetime index
    symbol : str, optional
        Symbol name for logging (not used in calculation)
    volatility_threshold : float, default=0.06
        Max relative volatility for trending regime (6% stddev relative to mean)
    trend_strength_threshold : float, default=0.08
        Min absolute price change over lookback_trend period (8%)
    lookback_volatility : int, default=14
        Days/periods to calculate rolling volatility
    lookback_trend : int, default=30
        Days/periods to measure trend strength

    Returns:
    --------
    pd.Series
        Series with same index as prices_df, values are 'trending' or 'choppy'

    Logic:
    ------
    - volatility = price.rolling(14).std() / price.mean()
    - trend_strength = abs(price.pct_change(30))
    - If volatility < 0.06 AND trend_strength > 0.08: 'trending'
    - Else: 'choppy'

    Rationale:
    ----------
    Models work best when there's clear directional movement without excessive noise.
    Low volatility + strong trend = predictable moves = good model performance.
    High volatility OR no clear trend = choppy = bad model performance.
    """
    if "close" not in prices_df.columns:
        raise ValueError("prices_df must have 'close' column")

    close = prices_df["close"].astype(float)

    # Calculate rolling volatility as fraction of mean price
    rolling_std = close.rolling(lookback_volatility, min_periods=max(3, lookback_volatility // 2)).std()
    mean_price = close.rolling(lookback_volatility, min_periods=max(3, lookback_volatility // 2)).mean()
    volatility = (rolling_std / mean_price.replace(0, np.nan)).fillna(0)

    # Calculate trend strength as absolute price change percentage
    trend_strength = close.pct_change(lookback_trend).abs().fillna(0)

    # Regime logic: trending if low volatility AND strong trend
    is_trending = (volatility < volatility_threshold) & (trend_strength > trend_strength_threshold)

    regime = pd.Series("choppy", index=prices_df.index, dtype=str)
    regime[is_trending] = "trending"

    return regime


def add_regime_feature(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    symbol: str = "",
    **regime_kwargs
) -> pd.DataFrame:
    """
    Add regime as a binary feature column to feature DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Feature DataFrame to augment
    prices_df : pd.DataFrame
        Price data with 'close' column
    symbol : str, optional
        Symbol name for logging
    **regime_kwargs :
        Additional arguments passed to detect_market_regime()

    Returns:
    --------
    pd.DataFrame
        Copy of df with added 'regime_trending' column (1=trending, 0=choppy)
    """
    df = df.copy()

    regime = detect_market_regime(prices_df, symbol=symbol, **regime_kwargs)

    # Align regime with feature df index
    regime_aligned = regime.reindex(df.index, fill_value="choppy")

    # Add as binary feature
    df["regime_trending"] = (regime_aligned == "trending").astype(float)

    return df


def filter_by_regime(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    symbol: str = "",
    keep_regime: RegimeType = "trending",
    **regime_kwargs
) -> pd.DataFrame:
    """
    Filter feature DataFrame to only include rows matching specified regime.

    Parameters:
    -----------
    df : pd.DataFrame
        Feature DataFrame to filter
    prices_df : pd.DataFrame
        Price data with 'close' column
    symbol : str, optional
        Symbol name for logging
    keep_regime : 'trending' or 'choppy', default='trending'
        Which regime to keep
    **regime_kwargs :
        Additional arguments passed to detect_market_regime()

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only rows in specified regime
    """
    regime = detect_market_regime(prices_df, symbol=symbol, **regime_kwargs)

    # Align regime with feature df index
    regime_aligned = regime.reindex(df.index, fill_value="choppy")

    # Filter
    mask = regime_aligned == keep_regime
    return df[mask].copy()


def get_regime_stats(
    prices_df: pd.DataFrame,
    symbol: str = "",
    **regime_kwargs
) -> dict:
    """
    Get statistics about regime distribution in price data.

    Returns:
    --------
    dict with keys:
        - total_periods: int
        - trending_periods: int
        - choppy_periods: int
        - trending_pct: float (0-100)
        - choppy_pct: float (0-100)
    """
    regime = detect_market_regime(prices_df, symbol=symbol, **regime_kwargs)

    total = len(regime)
    trending = (regime == "trending").sum()
    choppy = (regime == "choppy").sum()

    return {
        "symbol": symbol,
        "total_periods": int(total),
        "trending_periods": int(trending),
        "choppy_periods": int(choppy),
        "trending_pct": round(100 * trending / total, 2) if total > 0 else 0.0,
        "choppy_pct": round(100 * choppy / total, 2) if total > 0 else 0.0,
    }


# Environment variable support
def regime_filtering_enabled() -> bool:
    """Check if regime filtering is enabled via MW_REGIME_FILTER_ENABLED env var."""
    return str(os.getenv("MW_REGIME_FILTER_ENABLED", "0")).lower() in {"1", "true", "yes"}


def get_regime_config() -> dict:
    """
    Get regime detection configuration from environment variables.

    Environment Variables:
    ----------------------
    MW_REGIME_VOLATILITY_THRESHOLD : float, default=0.06
        Max relative volatility for trending regime
    MW_REGIME_TREND_THRESHOLD : float, default=0.08
        Min trend strength for trending regime
    MW_REGIME_LOOKBACK_VOL : int, default=14
        Periods for volatility calculation
    MW_REGIME_LOOKBACK_TREND : int, default=30
        Periods for trend strength calculation

    Returns:
    --------
    dict with kwargs for detect_market_regime()
    """
    return {
        "volatility_threshold": float(os.getenv("MW_REGIME_VOLATILITY_THRESHOLD", "0.06")),
        "trend_strength_threshold": float(os.getenv("MW_REGIME_TREND_THRESHOLD", "0.08")),
        "lookback_volatility": int(os.getenv("MW_REGIME_LOOKBACK_VOL", "14")),
        "lookback_trend": int(os.getenv("MW_REGIME_LOOKBACK_TREND", "30")),
    }

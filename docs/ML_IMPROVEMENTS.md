# ML Performance Improvements - Implementation Guide

## Problem Statement

Current model performance: **49.1% Win Rate, 0.99 Profit Factor**
- Not profitable (need >50% WR, >1.0 PF)
- Inconsistent across folds (60% in some, 40% in others)
- Regime sensitive (works in trending markets, fails in choppy)

**Goal:** Achieve **52%+ Win Rate, 1.15+ Profit Factor** consistently across all walk-forward folds.

---

## Implemented Approaches

All 4 approaches are fully implemented and ready for testing:

### ✅ Approach 1: Regime Detection + Filtering
**Strategy:** Only train on trending markets where model has edge

**Implementation:**
- New module: `scripts/ml/regime_detector.py`
- Detects trending vs choppy markets using volatility + trend strength
- Filters training data to only include trending periods
- Model learns from clean, predictable data only

**Files Modified:**
- `scripts/ml/regime_detector.py` (NEW)
- `scripts/ml/feature_builder.py` - adds regime feature
- `scripts/ml/train_predict.py` - adds regime filtering logic

**Environment Variables:**
```bash
MW_REGIME_ENABLED=1
MW_REGIME_FILTER_ENABLED=1
MW_REGIME_VOLATILITY_THRESHOLD=0.06  # Max volatility for trending
MW_REGIME_TREND_THRESHOLD=0.08       # Min trend strength
```

**Expected Impact:** Reduce overfitting to choppy markets, improve consistency

---

### ✅ Approach 2: Extended Training Data (365 days)
**Strategy:** Train on full year instead of 180 days

**Implementation:**
- Extended lookback from 180 → 365 days
- Captures more market cycles (bull, bear, ranging)
- Better generalization across conditions

**Configuration:**
```bash
MW_ML_LOOKBACK_DAYS=365
```

**Expected Impact:** More robust model, less sensitive to recent market conditions

---

### ✅ Approach 3: Simplified LR-only Model
**Strategy:** Reduce model complexity to prevent overfitting

**Implementation:**
- Use only Logistic Regression (no ensemble)
- Feature selection: Top 5 most important features only
  - r_6h (6-hour returns)
  - vol_6h (6-hour volatility)
  - social_score (social sentiment)
  - atr_14h (14-hour ATR)
  - sma_gap (SMA deviation)
- Strong regularization (C=0.1)

**Files Modified:**
- `scripts/ml/train_predict.py` - adds feature selection logic
- `models/ml_hyperparameters.json` - adds LR config

**Environment Variables:**
```bash
MW_ML_MODEL=logreg
MW_FEATURE_SELECTION=top5
MW_ML_LOOKBACK_DAYS=365
```

**Expected Impact:** Lower overfitting, more consistent performance across folds

---

### ✅ Approach 4: Per-Regime Models
**Strategy:** Train separate models for different market conditions

**Implementation:**
- Model A: Trained on trending markets only
- Model B: Trained on choppy markets only
- Fallback: Trained on all data (safety net)
- At inference: detect regime → use appropriate model

**Files Modified:**
- `scripts/ml/per_regime_trainer.py` (NEW)
- `scripts/ml/train_predict.py` - integrates per-regime training

**Environment Variables:**
```bash
MW_REGIME_ENABLED=1
MW_PER_REGIME_MODELS=1
MW_ML_LOOKBACK_DAYS=365
```

**Expected Impact:** Best performance - specialized models for each regime

---

## Testing Instructions

### Quick Start

Each approach has a pre-configured test script:

```bash
# Approach 1: Regime Filtering
bash configs/approach_1_regime_filter.sh

# Approach 2: Extended Data
bash configs/approach_2_extended_data.sh

# Approach 3: Simple LR
bash configs/approach_3_simple_lr.sh

# Approach 4: Per-Regime Models
bash configs/approach_4_per_regime.sh
```

### Manual Testing

Set environment variables and run training:

```bash
# Example: Test Approach 1
export MW_ML_LOOKBACK_DAYS=180
export MW_ML_MODEL=hybrid
export MW_REGIME_ENABLED=1
export MW_REGIME_FILTER_ENABLED=1
export MW_ML_SYMBOLS=BTC,ETH

python -m scripts.ml.train_predict
```

### Walk-Forward Validation

To run proper walk-forward cross-validation (GitHub Actions):

1. Push changes to branch
2. Go to Actions → "Walkforward CV"
3. Run workflow with appropriate environment overrides
4. Check `models/walkforward_summary.json` for results

---

## Output & Metrics

### Key Files to Check

**Training Results:**
- `models/ml_model_manifest.json` - Training manifest with fold metrics
- `artifacts/ml_*` - ROC curves, equity plots
- `models/backtest_summary.json` - Backtest performance

**Per-Regime Models (Approach 4):**
- `models/current/{SYMBOL}_trending_model.joblib`
- `models/current/{SYMBOL}_choppy_model.joblib`
- `models/current/{SYMBOL}_fallback_model.joblib`
- `models/current/{SYMBOL}_regime_manifest.json`

### Success Criteria

Compare all 4 approaches on:

| Metric | Target | Notes |
|--------|--------|-------|
| **Win Rate** | >52% | Across all folds |
| **Profit Factor** | >1.15 | Gross profit / gross loss |
| **Sharpe Ratio** | >0.5 | Risk-adjusted returns |
| **Fold Consistency** | StdDev <7% | Low variance across folds |
| **Max Drawdown** | <15% | Downside protection |

**Pick the approach with:**
1. Highest win rate (>52%)
2. Best profit factor (>1.15)
3. Lowest fold variance (most consistent)

---

## Environment Variables Reference

### Core Training
```bash
MW_ML_LOOKBACK_DAYS=180      # Days of training data (180 or 365)
MW_ML_MODEL=hybrid           # Model type: logreg, hybrid, gb, hgb
MW_ML_SYMBOLS=BTC,ETH        # Symbols to train
MW_TRAIN_DAYS=60             # Days per training fold
MW_TEST_DAYS=30              # Days per test fold
MW_HORIZON_H=1               # Prediction horizon in hours
```

### Regime Detection (Approach 1 & 4)
```bash
MW_REGIME_ENABLED=1                    # Enable regime feature
MW_REGIME_FILTER_ENABLED=1             # Filter training data by regime
MW_REGIME_VOLATILITY_THRESHOLD=0.06    # Max volatility for trending
MW_REGIME_TREND_THRESHOLD=0.08         # Min trend strength for trending
MW_REGIME_LOOKBACK_VOL=14              # Periods for volatility calc
MW_REGIME_LOOKBACK_TREND=30            # Periods for trend calc
```

### Feature Selection (Approach 3)
```bash
MW_FEATURE_SELECTION=top5    # Use only top 5 features
MW_FEATURE_SELECTION=all     # Use all features (default)
```

### Per-Regime Training (Approach 4)
```bash
MW_PER_REGIME_MODELS=1       # Train separate models per regime
```

---

## Architecture Details

### Regime Detection Logic

```python
# Trending market criteria:
volatility = price.rolling(14).std() / price.mean()
trend_strength = abs(price.pct_change(30))

is_trending = (volatility < 0.06) AND (trend_strength > 0.08)
```

**Rationale:** Low volatility + strong trend = predictable = model works well

### Feature Selection (Top 5)

Based on feature importance analysis:
1. **r_6h**: 6-hour price returns (momentum)
2. **vol_6h**: 6-hour volatility (risk indicator)
3. **social_score**: Social sentiment (crowd behavior)
4. **atr_14h**: Average True Range (volatility measure)
5. **sma_gap**: Deviation from SMA (trend indicator)

### Per-Regime Training Flow

```
Training:
├─ Detect regime for each period
├─ Split data: trending vs choppy
├─ Train Model A on trending data
├─ Train Model B on choppy data
└─ Train Fallback on all data

Inference:
├─ Detect current regime
├─ If trending → use Model A
├─ If choppy → use Model B
└─ If model missing → use Fallback
```

---

## Troubleshooting

### Issue: "Insufficient trending samples"
**Cause:** Regime filter removed too much training data
**Solution:**
- Relax volatility threshold: `MW_REGIME_VOLATILITY_THRESHOLD=0.08`
- Or relax trend threshold: `MW_REGIME_TREND_THRESHOLD=0.06`

### Issue: "Missing features" warning
**Cause:** Feature not generated in feature_builder
**Solution:** Ensure `MW_REGIME_ENABLED=1` if using regime features

### Issue: Per-regime models failing
**Cause:** Insufficient data in one regime
**Solution:** Falls back to standard training automatically (check logs)

### Issue: 365-day data fetch fails
**Cause:** CoinGecko rate limits
**Solution:**
- Data loader has retry logic with exponential backoff
- Wait and retry, or reduce lookback to 180 days

---

## Next Steps

1. **Test all 4 approaches** using the provided config scripts
2. **Compare results** in walk-forward validation
3. **Pick winner** based on WR, PF, and consistency
4. **Update production config** with winning approach
5. **Monitor live performance** for 1-2 weeks before launch

---

## Backward Compatibility

✅ All changes are additive and backward compatible:
- Default behavior unchanged (all new features disabled by default)
- Environment variables control all new functionality
- Existing models/workflows continue to work
- No breaking changes to APIs or file formats

---

## Files Changed

**New Files:**
- `scripts/ml/regime_detector.py` - Regime detection module
- `scripts/ml/per_regime_trainer.py` - Per-regime training module
- `configs/approach_1_regime_filter.sh` - Test script
- `configs/approach_2_extended_data.sh` - Test script
- `configs/approach_3_simple_lr.sh` - Test script
- `configs/approach_4_per_regime.sh` - Test script

**Modified Files:**
- `scripts/ml/feature_builder.py` - Added regime feature
- `scripts/ml/train_predict.py` - Added all 4 approaches
- `models/ml_hyperparameters.json` - Added approach configs

---

## Questions?

Check the code comments in:
- `scripts/ml/regime_detector.py` - Regime detection details
- `scripts/ml/per_regime_trainer.py` - Per-regime training details
- `scripts/ml/train_predict.py` - Integration logic

All functions have comprehensive docstrings explaining parameters and logic.

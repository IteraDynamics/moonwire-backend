# MoonWire Full-History Validation Workflow Guide

## Overview

The `test-ml-improvements.yml` workflow now includes a new approach: **`itera_full_history`**

This runs MoonWire's signal engine across the **entire 6+ year Itera dataset** (2019-2025, 61k+ bars) with:
- Walk-forward validation (no lookahead bias)
- Price-only features (no social data needed)
- Full backtest with trading metrics
- Complete signal feed export

---

## How to Run

### 1. Via GitHub Actions UI

1. Go to: https://github.com/IteraDynamics/moonwire-backend/actions
2. Select **"ML Improvement Testing"** workflow
3. Click **"Run workflow"** dropdown
4. Configure inputs:

| Input | Value | Notes |
|-------|-------|-------|
| **approach** | `itera_full_history` | ⚠️ Required - this triggers the new job |
| **symbols** | `BTC,ETH` | Comma-separated, no spaces |
| **horizon_h** | `1` | Prediction horizon (1, 3, or 6 hours) |
| **backtest_long_thresh** | `0.65` | Enter LONG when p_long ≥ this |
| **backtest_short_thresh** | `0.35` | Enter SHORT when p_long ≤ this |
| **backtest_debounce_hours** | `10` | Hours between trades (prevents overtrading) |
| **enable_social** | `false` | ⚠️ Leave disabled (ignored anyway) |

5. Click **"Run workflow"**

### 2. Monitor Progress

The workflow will:
1. ✅ Download/verify Itera data from `../iteradynamics/data/`
2. ✅ Run walk-forward validation (~20-30 min per symbol)
3. ✅ Train final models on all data
4. ✅ Export full signal feeds (61k+ signals per symbol)
5. ✅ Run backtests with configurable thresholds
6. ✅ Generate comprehensive summary

**Estimated runtime:** 30-60 minutes (depending on symbols)

### 3. Download Results

When complete, click **"Artifacts"** to download:

```
itera-full-history-results/
├── validation_results/
│   ├── validation_results.json      # Walk-forward metrics by fold
│   ├── final_models.json            # Model training summary
│   ├── training_manifest.json       # Complete config + provenance
│   ├── backtest_BTC.json           # BTC backtest results
│   ├── backtest_ETH.json           # ETH backtest results
│   └── models/
│       ├── BTC_model.joblib        # Trained BTC model
│       └── ETH_model.joblib        # Trained ETH model
└── feeds/
    ├── btc/
    │   ├── signals.jsonl           # 61k+ BTC signals
    │   └── manifest.json           # Signal metadata
    └── eth/
        ├── signals.jsonl           # 61k+ ETH signals
        └── manifest.json
```

---

## Understanding the Output

### Walk-Forward Validation Results

**File:** `validation_results/validation_results.json`

Shows performance across multiple time windows (expanding window, 90-day validation periods):

```json
{
  "BTC": {
    "aggregate": {
      "num_folds": 15,
      "mean_accuracy": 0.5292,
      "mean_precision": 0.5493,
      "mean_recall": 0.6141,
      "mean_f1": 0.5799
    },
    "folds": [
      {
        "fold": 1,
        "train_start": "2019-01-01",
        "train_end": "2020-01-01",
        "val_start": "2020-01-01",
        "val_end": "2020-04-01",
        "accuracy": 0.5234,
        ...
      }
    ]
  }
}
```

**Key Metrics:**
- **Accuracy:** Percentage of correct predictions (> 50% = edge)
- **Precision:** When model predicts UP, how often is it right?
- **Recall:** Of all actual UP moves, how many did we catch?
- **F1:** Harmonic mean of precision + recall

### Backtest Results

**File:** `validation_results/backtest_BTC.json`

Shows actual trading performance with fees:

```json
{
  "aggregate": {
    "cagr": 0.4567,                // 45.67% annualized return
    "max_drawdown": -0.1342,       // -13.42% worst drawdown
    "win_rate": 0.5474,            // 54.74% winning trades
    "n_trades": 137,               // Total trades executed
    "signals_per_day": 0.5076,     // ~15 signals/month
    "profit_factor": 1.41,         // Gross profit / gross loss
    "sharpe_ratio": 1.23           // Risk-adjusted return
  },
  "trades": [
    {
      "ts_entry": "2020-03-12T14:00:00Z",
      "ts_exit": "2020-03-12T17:00:00Z",
      "side": "long",
      "entry_px": 5432.10,
      "exit_px": 5678.90,
      "pnl": 246.80,
      "pnl_pct": 0.0454
    }
  ]
}
```

**Trading Costs:**
- Fee: 10 bps (0.1%)
- Slippage: 5 bps (0.05%)
- Total: ~15 bps per round-trip

### Signal Feeds

**File:** `feeds/btc/signals.jsonl`

One signal per hourly bar (61k+ lines):

```jsonl
{"timestamp": "2019-01-01T00:00:00Z", "product_id": "BTC-USD", "p_long": 0.5234, "horizon_hours": 1, "model_version": "v1.0-itera-full-history", "generated_at": "2026-03-10T01:47:00Z", "close": 3842.10}
{"timestamp": "2019-01-01T01:00:00Z", "product_id": "BTC-USD", "p_long": 0.4876, "horizon_hours": 1, "model_version": "v1.0-itera-full-history", "generated_at": "2026-03-10T01:47:00Z", "close": 3831.25}
```

**Fields:**
- `p_long`: Probability of upward move (0-1)
- `horizon_hours`: Prediction window (1h, 3h, or 6h)
- `close`: Bar close price (for reference)

**Thresholds:**
- `p_long >= 0.65` → LONG signal (bullish)
- `p_long <= 0.35` → SHORT signal (bearish)
- `0.35 < p_long < 0.65` → NEUTRAL (no trade)

---

## Interpreting Results

### Good Results (Production-Ready)

✅ **Validation metrics > 50%** (above random guessing)
✅ **CAGR > 0** (positive returns after fees)
✅ **Win rate ≥ 50%** (at least break-even)
✅ **Max drawdown < -30%** (manageable risk)
✅ **Profit factor > 1.0** (profitable overall)

### Red Flags

❌ Accuracy < 50% (no edge)
❌ CAGR < 0 (loses money after fees)
❌ Win rate < 45% (too many losers)
❌ Max drawdown < -50% (excessive risk)
❌ Very few trades (<20) or too many (>500/year)

### Example Interpretation

**BTC Results:**
- CAGR: 45.67%
- Max DD: -13.42%
- Win Rate: 54.74%
- Trades: 137 (over 7 years = ~20/year)

**Translation:** The model would have delivered solid returns (45% annualized) with manageable risk (13% max drawdown) and a slight edge (54.7% win rate). Low trade frequency (20/year) means it's selective, not overtrading.

---

## Customizing Backtest Thresholds

Want to test different trading strategies? Adjust these inputs:

### Conservative (fewer, higher-confidence trades)

```
backtest_long_thresh: 0.70    # Only enter LONG at 70%+ confidence
backtest_short_thresh: 0.30   # Only enter SHORT at 30%- confidence
backtest_debounce_hours: 24   # Wait 24h between trades
```

Expected: Fewer trades, higher win rate, potentially lower returns

### Aggressive (more frequent trading)

```
backtest_long_thresh: 0.55    # Enter LONG at 55%+ confidence
backtest_short_thresh: 0.45   # Enter SHORT at 45%- confidence
backtest_debounce_hours: 3    # Wait only 3h between trades
```

Expected: More trades, lower win rate, fee drag risk

### Balanced (recommended starting point)

```
backtest_long_thresh: 0.65    # Default
backtest_short_thresh: 0.35   # Default
backtest_debounce_hours: 10   # Default
```

---

## Comparing to Buy & Hold

The backtest results include comparison to **buy-and-hold** strategy:

**BTC Buy & Hold (2019-2025):**
- CAGR: ~56.7%
- Max DD: ~-77.6% (2021-2022 crash)

**Why use signals if B&H is higher?**
1. **Lower drawdowns:** MoonWire can go NEUTRAL/SHORT during crashes
2. **Risk management:** Not always exposed to BTC volatility
3. **Diversification:** Can combine with other strategies
4. **Overlay potential:** Use signals to size positions (bigger during high confidence)

**Key insight:** MoonWire is defensive alpha, not a B&H replacement.

---

## Next Steps After Validation

### If Results Look Good

1. **Review signal feeds:** Check for logical patterns (high p_long during rallies, low during crashes)
2. **Test in Itera:** Import `feeds/btc/signals.jsonl` as external signal source
3. **Paper trade:** Run live for 30 days before real capital
4. **Optimize thresholds:** Use grid search to find best long/short thresholds

### If Results Look Weak

1. **Check validation metrics:** Are models learning anything? (accuracy > 50%?)
2. **Inspect fold results:** Is performance consistent across folds?
3. **Try different horizon:** Test 3h or 6h instead of 1h
4. **Feature engineering:** Add more technical indicators
5. **Consider regime filters:** Only trade during specific market conditions

---

## Technical Notes

### Data Requirements

The workflow expects Itera CSV files at:

```
../iteradynamics/data/
├── btcusd_3600s_2019-01-01_to_2025-12-30.csv
└── ethusd_3600s_2019-01-01_to_2025-12-30.csv
```

**Format:**
```csv
Timestamp,Open,High,Low,Close,Volume
2019-01-01 00:00:00,3842.10,3865.20,3830.45,3852.33,123456.78
```

If these files are missing, the workflow will **fail at the data download step**.

### Walk-Forward Validation Logic

1. **Initial training window:** 365 days (1 year)
2. **Validation window:** 90 days (3 months)
3. **Step size:** 90 days (move forward 3 months each fold)
4. **Window type:** Expanding (use all prior data for training)

This ensures:
- No lookahead bias (never train on future data)
- Realistic performance (test on unseen periods)
- Robust estimates (multiple folds, different market regimes)

### Model Architecture

**Hybrid Ensemble:**
- Logistic Regression (L2 regularization)
- Random Forest (100 trees, max_depth=10)
- Gradient Boosting (100 estimators, max_depth=5)

**Prediction:** Simple average of all model probabilities

**Features (price-only):**
- `r_1h`, `r_3h`, `r_6h` - Returns at multiple horizons
- `vol_6h` - Rolling volatility
- `atr_14h` - Average True Range
- `sma_gap` - 6h/24h SMA ratio
- `high_vol` - High volatility regime flag

---

## Troubleshooting

### "BTC data not found - workflow will fail"

**Solution:** Ensure Itera CSV files are committed to the repo at `../iteradynamics/data/`

If they're too large for Git, you'll need to:
1. Store them in Git LFS, or
2. Modify the workflow to download from external source (S3, GCS, etc.)

### "Training may have failed"

**Check:**
1. Workflow logs for Python errors
2. Data quality (missing values, wrong format)
3. Feature engineering (NaN values after warmup trim?)

### "No trades in backtest"

**Possible causes:**
1. Thresholds too strict (p_long never reaches 0.65 or drops below 0.35)
2. Debounce too long (blocks all trades)
3. Model predictions too neutral (always near 0.5)

**Solution:** Lower thresholds or check model calibration

---

## Support

Questions or issues? Contact:
- GitHub Issues: https://github.com/IteraDynamics/moonwire-backend/issues
- Itera Team: Andrew (@alallos)

---

**Last Updated:** 2026-03-10  
**Workflow Version:** 1.0  
**Branch:** signal-engine-validation

# 📋 Paper Trading Parameters Cheat Sheet

## All Available Parameters in `models/paper_trading_params.json`

| Parameter | Description | Default | Unit/Format |
|-----------|-------------|---------|-------------|
| **`deadband`** | Ignore signals within this distance from 0.5 (filters indecisive signals) | `0.08` | 0.42-0.58 ignored |
| **`min_flip_min`** | Minimum minutes between flipping positions (long→short or short→long) | `360` | Minutes (6 hrs) |
| **`lookback_h`** | Replay window for paper trading simulation | `720` | Hours (30 days) |
| **`conf_min`** | Minimum confidence threshold to accept a signal | `0.60` | 0.0 - 1.0 |
| **`horizon_min`** | Time horizon for evaluating position performance | `60` | Minutes (1 hr) |
| **`slippage_bps`** | Assumed slippage cost per trade | `2` | Basis points (0.02%) |
| **`fees_bps`** | Trading fees per trade | `1` | Basis points (0.01%) |
| **`capital`** | Starting capital for paper trading simulation | `100000` | USD |
| **`position_size_pct`** | Percentage of capital to risk per trade | `1.0` | Decimal (1.0 = 100%, 0.02 = 2%) |
| **`stop_loss_pct`** | Stop loss percentage (exit if loss exceeds this) | `null` | Decimal (0.02 = 2%) or null for none |
| **`risk_free`** | Risk-free rate for Sharpe ratio calculation | `0.0` | Annual rate (0.05 = 5%) |

---

## Workflow Dropdown Overrides (Optional - Per Run)

When running `walkforward-cv.yml`, you can override these 5 parameters **for that specific run only**:

| Dropdown Field | Overrides | Default if Left Blank |
|----------------|-----------|----------------------|
| **`deadband_override`** | `deadband` | Uses config file → `0.08` |
| **`min_flip_min_override`** | `min_flip_min` | Uses config file → `360` |
| **`lookback_h_override`** | `lookback_h` | Uses config file → `720` |
| **`position_size_pct`** | `position_size_pct` | Uses config file → `1.0` |
| **`stop_loss_pct`** | `stop_loss_pct` | Uses config file → `null` |

**All other parameters can only be set in the config file.**

---

## Priority Order

For every parameter, the system checks in this order:

```
1. Workflow dropdown (if filled in)
   ↓ if blank
2. Config file value
   ↓ if missing
3. Hardcoded default
```

**Examples:**

**Deadband:**
```
Workflow: "0.10" → 0.10 ✅
  ↓ blank
Config: "deadband": 0.08 → 0.08 ✅
  ↓ missing
Default: 0.08 ✅
```

**Slippage (no workflow override):**
```
Config: "slippage_bps": 3 → 3 ✅
  ↓ missing
Default: 2 ✅
```

---

## Quick Tuning Guide

| Goal | Parameter to Change | Direction |
|------|---------------------|-----------|
| More selective signals | `conf_min` | 0.60 → 0.65+ |
| Less selective signals | `conf_min` | 0.60 → 0.55 |
| Filter weak signals | `deadband` | 0.08 → 0.12+ |
| Accept more signals | `deadband` | 0.08 → 0.04 |
| Reduce whipsaw | `min_flip_min` | 360 → 720+ |
| More responsive | `min_flip_min` | 360 → 180 |
| Longer backtest | `lookback_h` | 720 → 2160+ (90 days) |
| Shorter backtest | `lookback_h` | 720 → 360 (15 days) |
| Conservative costs | `slippage_bps`, `fees_bps` | Increase both |
| Larger capital | `capital` | 100000 → 500000+ |
| Reduce risk per trade | `position_size_pct` | 1.0 → 0.02 (2%) |
| Full capital per trade | `position_size_pct` | 0.02 → 1.0 (100%) |
| Enable stop loss | `stop_loss_pct` | null → 0.02 (2%) |
| Tighter stop loss | `stop_loss_pct` | 0.05 → 0.02 |
| Wider stop loss | `stop_loss_pct` | 0.02 → 0.05 |

---

## Example Config Files

### Minimal (recommended)
```json
{
  "deadband": 0.08,
  "min_flip_min": 360,
  "lookback_h": 720
}
```
*Other params use defaults*

### Complete
```json
{
  "deadband": 0.08,
  "min_flip_min": 360,
  "lookback_h": 720,
  "conf_min": 0.60,
  "horizon_min": 60,
  "slippage_bps": 2,
  "fees_bps": 1,
  "capital": 100000,
  "position_size_pct": 1.0,
  "stop_loss_pct": null,
  "risk_free": 0.0
}
```
*All params explicit*

### Conservative Trading
```json
{
  "deadband": 0.12,
  "min_flip_min": 720,
  "conf_min": 0.65,
  "slippage_bps": 3,
  "fees_bps": 2,
  "position_size_pct": 0.02,
  "stop_loss_pct": 0.02
}
```
*Fewer, higher-quality trades with realistic costs and risk management*

### Aggressive Trading
```json
{
  "deadband": 0.04,
  "min_flip_min": 180,
  "conf_min": 0.55,
  "position_size_pct": 1.0,
  "stop_loss_pct": null
}
```
*More trades, faster reactions, full capital per trade*

### Risk-Managed Trading
```json
{
  "position_size_pct": 0.02,
  "stop_loss_pct": 0.02,
  "capital": 100000
}
```
*2% risk per trade with 2% stop loss - professional risk management*

---

## What if I Delete a Parameter?

**Safe to delete any parameter - it will use the default value shown in the table above.**

---

## Parameter Details

### `deadband`
- **Range:** 0.0 - 0.5
- **Effect:** Filters signals with confidence between (0.5 - deadband) and (0.5 + deadband)
- **Example:** deadband=0.08 ignores signals with confidence 0.42-0.58
- **Use case:** Reduce noise from weak/indecisive predictions

### `min_flip_min`
- **Range:** 0 - unlimited
- **Effect:** Prevents rapid position changes (whipsaw protection)
- **Example:** 360 = require 6 hours between long→short or short→long
- **Use case:** Reduce trading costs from excessive flipping

### `lookback_h`
- **Range:** 1 - unlimited
- **Effect:** How far back to replay shadow predictions for paper trading
- **Example:** 720 = last 30 days
- **Use case:** Balance between data volume and relevance

### `conf_min`
- **Range:** 0.0 - 1.0
- **Effect:** Minimum ML model confidence to accept a signal
- **Example:** 0.60 = only accept signals with 60%+ confidence
- **Use case:** Filter low-quality predictions

### `horizon_min`
- **Range:** 1 - unlimited
- **Effect:** How long to hold a position before evaluating outcome
- **Example:** 60 = evaluate after 1 hour
- **Use case:** Match your trading timeframe

### `slippage_bps`
- **Range:** 0 - unlimited
- **Effect:** Cost of market impact (price moves against you)
- **Example:** 2 = 0.02% = $20 per $100k trade
- **Use case:** Realistic cost modeling

### `fees_bps`
- **Range:** 0 - unlimited
- **Effect:** Exchange/broker trading fees
- **Example:** 1 = 0.01% = $10 per $100k trade
- **Use case:** Realistic cost modeling

### `capital`
- **Range:** > 0
- **Effect:** Starting capital for simulation (doesn't affect signal logic)
- **Example:** 100000 = $100k
- **Use case:** Match your actual trading capital

### `position_size_pct`
- **Range:** 0.0 - 1.0
- **Effect:** Percentage of current capital to allocate per trade
- **Example:** 0.02 = risk 2% of capital per trade, 1.0 = use 100% of capital
- **Use case:** Professional risk management - most traders risk 1-2% per trade

### `stop_loss_pct`
- **Range:** 0.0 - 1.0 or null
- **Effect:** Exit position early if drawdown exceeds this percentage
- **Example:** 0.02 = exit if position loses 2%, null = no stop loss
- **Use case:** Limit downside risk and protect against large losses
- **Note:** Stop is checked at 10 sample points during the trade period

### `risk_free`
- **Range:** 0.0 - 1.0 (annual rate)
- **Effect:** Used in Sharpe ratio calculation
- **Example:** 0.05 = 5% annual risk-free rate
- **Use case:** Set to current T-bill rate for accurate Sharpe

---

## Files and Locations

| File | Location | Purpose |
|------|----------|---------|
| Config file | `models/paper_trading_params.json` | Permanent defaults |
| Workflow | `.github/workflows/walkforward-cv.yml` | Per-run overrides (UI) |
| Script | `scripts/perf/replay_shadow_to_paper.py` | Reads params and runs simulation |

---

**Last Updated:** 2025-10-28
**Version:** 1.0
**Maintained by:** Claude Code

"""
MOSAIC -- Momentum Signal (Week 4)

Two sub-components blended by regime:
  1. 12-1 Momentum: (price[t-21] / price[t-252]) - 1
     Captures medium-term trend. Works in trending markets.
  2. Mean Reversion: z-score of price vs 20-day moving average
     Captures stretched conditions. Works in choppy/crisis markets.

Regime-adaptive blending:
  Risk-On:       100% momentum, 0% mean reversion
  Transitional:  50% momentum, 50% mean reversion
  Risk-Off:      0% momentum, 100% mean reversion (flipped sign = buy the dip)

Output: one score per ticker per day, clipped to [-1, +1]
  Positive = bullish (upward momentum or oversold bounce expected)
  Negative = bearish (downward momentum or overbought pullback expected)
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# SUB-SIGNAL 1: 12-1 MOMENTUM
# ════════════════════════════════════════════════════════════════

def compute_momentum(prices, lookback=252, skip=21):
    """
    Classic 12-1 momentum factor.

    Formula: (price[t - skip] / price[t - lookback]) - 1

    Why 252 and 21?
      252 trading days = ~12 months
      21 trading days = ~1 month
      We look at the return over 12 months but SKIP the most recent month.

    Why skip the last month?
      The last month exhibits SHORT-TERM REVERSAL (a different effect).
      Stocks that jumped recently tend to pull back next week.
      By skipping it, we isolate the pure medium-term trend signal.

    Output: raw momentum values (not yet normalized).
    Positive = stock has been going up. Negative = going down.
    """
    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices

    # Price 'skip' days ago (21 days = ~1 month)
    price_recent = close.shift(skip)

    # Price 'lookback' days ago (252 days = ~12 months)
    price_old = close.shift(lookback)

    # 12-1 momentum return
    momentum = (price_recent / price_old) - 1

    return momentum


def normalize_momentum(momentum, window=63):
    """
    Cross-sectional z-score normalization of momentum.

    For each day, z-score the momentum values ACROSS all tickers:
      z_i = (mom_i - mean_across_tickers) / std_across_tickers

    Why cross-sectional (across tickers) not time-series (over time)?
      Because we want to know: "Is AAPL's momentum HIGH relative to
      other stocks TODAY?" not "Is AAPL's momentum high relative to
      its own history?" The first is more useful for stock selection.

    Then clip to [-3, +3] and rescale to [-1, +1].
    """
    # Cross-sectional z-score: for each row (date), z-score across columns (tickers)
    row_mean = momentum.mean(axis=1)
    row_std = momentum.std(axis=1)

    # Avoid division by zero
    row_std = row_std.replace(0, np.nan)

    normalized = momentum.sub(row_mean, axis=0).div(row_std, axis=0)

    # Clip and rescale
    normalized = normalized.clip(-3, 3) / 3

    return normalized


# ════════════════════════════════════════════════════════════════
# SUB-SIGNAL 2: MEAN REVERSION
# ════════════════════════════════════════════════════════════════

def compute_mean_reversion(prices, short_window=20, long_window=60):
    """
    Mean-reversion z-score: how far is the price from its moving average?

    Formula: z = (price - MA_20) / rolling_std_20

    Interpretation:
      z = +2.0 → price is 2 std devs ABOVE its 20-day average (overbought)
      z =  0.0 → price is at its average (neutral)
      z = -2.0 → price is 2 std devs BELOW its 20-day average (oversold)

    For mean-reversion trading:
      Very negative z → oversold → expect bounce → BULLISH
      Very positive z → overbought → expect pullback → BEARISH

    So we FLIP the sign: mean_reversion_signal = -z
      Oversold (z=-2) → signal = +2 (bullish)
      Overbought (z=+2) → signal = -2 (bearish)

    We also compute a longer-window (60-day) version and average them
    for robustness. The 20-day catches fast moves, 60-day catches
    slower drifts.
    """
    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices

    # Short-term mean reversion (20-day)
    ma_short = close.rolling(window=short_window).mean()
    std_short = close.rolling(window=short_window).std()
    z_short = (close - ma_short) / std_short.replace(0, np.nan)

    # Longer-term mean reversion (60-day)
    ma_long = close.rolling(window=long_window).mean()
    std_long = close.rolling(window=long_window).std()
    z_long = (close - ma_long) / std_long.replace(0, np.nan)

    # Average the two timeframes and FLIP sign (oversold = bullish)
    mean_rev = -0.5 * (z_short + z_long)

    # Clip to [-3, 3] and rescale to [-1, 1]
    mean_rev = mean_rev.clip(-3, 3) / 3

    return mean_rev


# ════════════════════════════════════════════════════════════════
# REGIME-ADAPTIVE BLENDING
# ════════════════════════════════════════════════════════════════

def blend_signals(momentum_norm, mean_reversion, regime_labels=None):
    """
    Blend momentum and mean-reversion based on the market regime.

    Regime weights:
      Risk-On (0):       momentum=1.0, mean_rev=0.0 (trends persist, ride them)
      Transitional (1):  momentum=0.5, mean_rev=0.5 (mixed, hedge your bets)
      Risk-Off (2):      momentum=0.0, mean_rev=1.0 (panic overshoots, buy dips)

    If no regime labels provided (regime model not built yet), use
    equal weighting as default: 0.5 * momentum + 0.5 * mean_reversion.
    This is a reasonable starting point.

    regime_labels: Series indexed by date with values 0, 1, or 2.
    """
    # Default: equal blend
    if regime_labels is None:
        logger.info("  No regime labels provided -- using 50/50 blend")
        blended = 0.5 * momentum_norm.fillna(0) + 0.5 * mean_reversion.fillna(0)
        return blended.clip(-1, 1)

    # Regime-adaptive weights
    regime_weights = {
        0: (1.0, 0.0),   # Risk-On: pure momentum
        1: (0.5, 0.5),   # Transitional: balanced
        2: (0.0, 1.0),   # Risk-Off: pure mean reversion
    }

    blended = pd.DataFrame(0.0, index=momentum_norm.index, columns=momentum_norm.columns)

    for date in momentum_norm.index:
        regime = regime_labels.get(date, 1)  # Default to transitional if missing
        mom_weight, mr_weight = regime_weights.get(regime, (0.5, 0.5))

        mom_row = momentum_norm.loc[date].fillna(0)
        mr_row = mean_reversion.loc[date].fillna(0)

        blended.loc[date] = mom_weight * mom_row + mr_weight * mr_row

    return blended.clip(-1, 1)


# ════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════

def validate_momentum_signal(momentum_score, momentum_raw=None, mean_rev=None):
    """Quality checks on the momentum signal."""
    results = {}

    results["shape"] = momentum_score.shape
    results["date_range"] = (
        str(momentum_score.index.min().date()),
        str(momentum_score.index.max().date()),
    )

    flat = momentum_score.values.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) > 0:
        results["mean"] = round(np.mean(flat), 4)
        results["std"] = round(np.std(flat), 4)
        results["pct_positive"] = round((flat > 0).mean() * 100, 1)
        results["pct_negative"] = round((flat < 0).mean() * 100, 1)
        results["missing_pct"] = round(
            momentum_score.isnull().sum().sum() / momentum_score.size * 100, 1
        )

    # Check if momentum sub-signal has reasonable values
    if momentum_raw is not None:
        raw_flat = momentum_raw.values.flatten()
        raw_flat = raw_flat[~np.isnan(raw_flat)]
        if len(raw_flat) > 0:
            results["raw_momentum_mean"] = round(np.mean(raw_flat), 4)
            results["raw_momentum_median"] = round(np.median(raw_flat), 4)

    return results


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def run(prices=None, regime_labels=None):
    """
    Main entry point for momentum signal.

    Args:
        prices: Price DataFrame from price_loader (or loads from cache)
        regime_labels: Optional Series of regime states from HMM (0, 1, 2)
                      If None, uses 50/50 default blend.

    Returns: (momentum_score, momentum_raw, mean_reversion, validation)
    """
    # Load prices from cache if not provided
    if prices is None:
        prices_path = RAW_DIR / "prices.parquet"
        if prices_path.exists():
            prices = pd.read_parquet(prices_path)
            logger.info(f"Loaded prices: {prices.shape}")
        else:
            raise FileNotFoundError("No price data found. Run price_loader.py first.")

    # Build sub-signals
    logger.info("Computing 12-1 momentum...")
    momentum_raw = compute_momentum(prices)
    logger.info(f"  ✓ Raw momentum shape: {momentum_raw.shape}")

    logger.info("Normalizing momentum (cross-sectional z-score)...")
    momentum_norm = normalize_momentum(momentum_raw)
    logger.info(f"  ✓ Normalized momentum shape: {momentum_norm.shape}")

    logger.info("Computing mean-reversion signal...")
    mean_rev = compute_mean_reversion(prices)
    logger.info(f"  ✓ Mean reversion shape: {mean_rev.shape}")

    # Blend based on regime
    logger.info("Blending signals...")
    momentum_score = blend_signals(momentum_norm, mean_rev, regime_labels)
    logger.info(f"  ✓ Final momentum signal shape: {momentum_score.shape}")

    # Validate
    validation = validate_momentum_signal(momentum_score, momentum_raw, mean_rev)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    momentum_score.to_parquet(PROCESSED_DIR / "momentum_signal.parquet")
    momentum_raw.to_parquet(PROCESSED_DIR / "momentum_raw.parquet")
    mean_rev.to_parquet(PROCESSED_DIR / "mean_reversion.parquet")
    logger.info(f"Saved momentum signals to {PROCESSED_DIR}")

    return momentum_score, momentum_raw, mean_rev, validation


if __name__ == "__main__":
    score, raw, mr, validation = run()

    print("\n=== Momentum Signal Validation ===")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    print("\n=== Sample: Momentum Score (last 10 days, first 5 tickers) ===")
    print(score.iloc[-10:, :5].round(3).to_string())
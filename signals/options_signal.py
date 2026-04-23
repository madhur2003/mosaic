"""
MOSAIC — Options Anomaly Signal (Week 3)

Detects unusual options market activity that may predict future stock moves.

Two sub-signals:
  1. IV Percentile Signal — is volatility unusually high or low vs past year?
     Used as a CONTRARIAN indicator (high fear → bullish, low fear → bearish)
  2. Put/Call Ratio Signal — is the put/call balance abnormally skewed?
     Also contrarian (extreme put buying → bullish reversal expected)

Combined: options_score = 0.5 * iv_signal + 0.5 * pc_signal, clipped to [-1, +1]

Positive score = bullish options setup (contrarian buy)
Negative score = bearish options setup (contrarian sell)
Near zero = nothing unusual

Validation: Information Coefficient (correlation to next-period returns)
  IC > 0.03 = meaningful, IC > 0.05 = strong, IC > 0.10 = suspicious
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR, TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# ROLLING Z-SCORE UTILITY
# ════════════════════════════════════════════════════════════════

def rolling_zscore(series, window=63):
    """
    Compute rolling z-score for a Series.

    z_t = (x_t - rolling_mean) / rolling_std

    Why rolling (not full-history)?
      - Full-history z-score uses future data = lookahead bias
      - What's "normal" changes over time (2020 vol != 2017 vol)
      - Rolling window adapts to the current regime

    Why 63 days (~3 months)?
      - Short enough to adapt to regime changes
      - Long enough for a stable mean/std estimate
      - Matches roughly one earnings cycle

    Returns a Series of z-scores, NaN for the first 'window' days.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()

    # Avoid division by zero — if std is 0, the series is constant, z-score is 0
    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)

    return zscore


# ════════════════════════════════════════════════════════════════
# SUB-SIGNAL 1: IV PERCENTILE SIGNAL
# ════════════════════════════════════════════════════════════════

def build_iv_signal(iv_percentile):
    """
    Convert IV percentile rank into a contrarian trading signal.

    Raw IV percentile is 0 to 1:
      0.90 = vol is higher than 90% of past year (fear is elevated)
      0.10 = vol is lower than 90% of past year (complacency)

    Contrarian logic (volatility mean-reverts):
      High IV percentile → fear is peaking → expect recovery → BULLISH
      Low IV percentile  → complacency → expect vol expansion → BEARISH

    Steps:
      1. Center the percentile around 0.5: (pctile - 0.5)
         Now: +0.4 = very high vol, -0.4 = very low vol
      2. Flip the sign (contrarian): multiply by -1
         Now: -0.4 = very high vol (bullish), +0.4 = very low vol (bearish)
         Wait — that's backwards. Let me re-think.

    Actually, our convention is:
      Positive score = bullish = expect price to go UP
      Negative score = bearish = expect price to go DOWN

    So:
      High IV percentile → contrarian bullish → POSITIVE score
      Low IV percentile  → contrarian bearish → NEGATIVE score

    This means: iv_signal = (iv_percentile - 0.5) * 2
      pctile=0.9 → signal = +0.8 (bullish)
      pctile=0.5 → signal =  0.0 (neutral)
      pctile=0.1 → signal = -0.8 (bearish)

    Then we z-score to emphasize extremes.
    """
    logger.info("Building IV percentile signal...")

    # Center and scale: 0-1 range → -1 to +1 range
    centered = (iv_percentile - 0.5) * 2

    # Apply rolling z-score to emphasize abnormal readings
    iv_signal = centered.apply(lambda col: rolling_zscore(col, window=63))

    # Clip extreme z-scores to [-3, 3] then rescale to [-1, 1]
    iv_signal = iv_signal.clip(-3, 3) / 3

    logger.info(f"  ✓ IV signal shape: {iv_signal.shape}")
    return iv_signal


# ════════════════════════════════════════════════════════════════
# SUB-SIGNAL 2: PUT/CALL RATIO SIGNAL
# ════════════════════════════════════════════════════════════════

def build_pc_signal(pc_ratio, zscore_threshold=2.0):
    """
    Convert Put/Call ratio into a contrarian anomaly signal.

    Raw PC ratio:
      < 0.7 = more calls (bullish sentiment)
      > 1.0 = more puts (fearful sentiment)

    Contrarian logic:
      Very high PC ratio → extreme fear → expect bounce → BULLISH (positive)
      Very low PC ratio  → extreme greed → expect pullback → BEARISH (negative)

    We only care about ANOMALIES — normal PC ratios give a near-zero signal.
    The rolling z-score naturally handles this: only readings > 2 sigma
    from the rolling mean produce strong signals.

    Steps:
      1. Rolling z-score of PC ratio (63-day window)
      2. Z-score > +2 → extreme put buying → contrarian bullish → positive signal
      3. Z-score < -2 → extreme call buying → contrarian bearish → negative signal
      4. Z-score between -2 and +2 → signal is near zero (nothing unusual)

    Note: We keep the raw z-score (not just a binary flag) because
    a z-score of +3.5 is more meaningful than +2.1. Gradation matters.
    """
    logger.info("Building Put/Call ratio signal...")

    # Rolling z-score
    pc_zscore = pc_ratio.apply(lambda col: rolling_zscore(col, window=63))

    # The z-score IS the signal (positive z = high PC = fear = contrarian bullish)
    # Clip to [-3, 3] then rescale to [-1, 1]
    pc_signal = pc_zscore.clip(-3, 3) / 3

    # Log how many anomalies we detected
    extreme_count = (pc_zscore.abs() > zscore_threshold).sum().sum()
    total_points = pc_zscore.notna().sum().sum()
    logger.info(f"  ✓ PC signal shape: {pc_signal.shape}")
    logger.info(f"  ✓ Anomalies detected (|z| > {zscore_threshold}): "
                f"{extreme_count} / {total_points} "
                f"({extreme_count/total_points*100:.1f}%)" if total_points > 0 else "")

    return pc_signal


# ════════════════════════════════════════════════════════════════
# COMBINE INTO FINAL OPTIONS SCORE
# ════════════════════════════════════════════════════════════════

def build_options_signal(iv_signal, pc_signal, iv_weight=0.5, pc_weight=0.5):
    """
    Combine IV and PC signals into one composite options score.

    options_score = iv_weight * iv_signal + pc_weight * pc_signal

    Equal weighting (0.5/0.5) is the starting point.
    The meta-allocator in Week 5 will learn better weights over time.

    Output is clipped to [-1, +1]:
      Positive = bullish options setup (contrarian signal: fear is high)
      Negative = bearish options setup (contrarian signal: complacency)
      Near zero = nothing unusual in options market
    """
    logger.info("Combining IV and PC signals...")

    # Align indices (should already match, but safety first)
    common_idx = iv_signal.index.intersection(pc_signal.index)
    common_cols = iv_signal.columns.intersection(pc_signal.columns)

    iv_aligned = iv_signal.loc[common_idx, common_cols]
    pc_aligned = pc_signal.loc[common_idx, common_cols]

    # Weighted combination
    options_score = (iv_weight * iv_aligned.fillna(0)
                     + pc_weight * pc_aligned.fillna(0))

    # Final clip to [-1, +1]
    options_score = options_score.clip(-1, 1)

    logger.info(f"  ✓ Final options signal shape: {options_score.shape}")
    return options_score


# ════════════════════════════════════════════════════════════════
# INFORMATION COEFFICIENT (VALIDATION)
# ════════════════════════════════════════════════════════════════

def compute_ic(signal_df, returns_df, forward_days=5):
    """
    Compute Information Coefficient: correlation between
    today's signal and the NEXT forward_days return.

    IC = cross-sectional correlation(signal_t, return_{t+1 to t+5})

    We compute IC for each day, then report the average.

    Why forward_days=5?
      - 1 day is too noisy (single-day returns have huge variance)
      - 5 days (~1 week) smooths out noise while staying responsive
      - Also common in industry for signal evaluation

    Interpretation:
      IC > 0.03 → meaningful signal (yes, 3% is good in finance)
      IC > 0.05 → strong signal
      IC > 0.10 → suspiciously strong (check for bugs/lookahead bias)
      IC ~0.00 → signal is just noise

    Why is 3% correlation considered good?
      Because you apply it across 20 stocks, daily, for years.
      Small edges compound. A fund with consistent 3% IC across a
      broad universe will significantly outperform over time.
    """
    logger.info(f"Computing Information Coefficient (forward={forward_days} days)...")

    # Compute forward returns
    forward_returns = returns_df.shift(-forward_days)  # Shift returns BACK to align with today's signal

    # Align signal and returns
    common_idx = signal_df.index.intersection(forward_returns.index)
    common_cols = signal_df.columns.intersection(forward_returns.columns)

    if len(common_idx) == 0 or len(common_cols) == 0:
        logger.warning("  No overlapping data between signal and returns")
        return pd.Series(dtype=float), {}

    sig = signal_df.loc[common_idx, common_cols]
    ret = forward_returns.loc[common_idx, common_cols]

    # Cross-sectional IC for each day
    # For each day: correlate the signal across all tickers with the forward return across all tickers
    daily_ic = pd.Series(index=common_idx, dtype=float)

    for date in common_idx:
        sig_row = sig.loc[date].dropna()
        ret_row = ret.loc[date].dropna()

        # Need at least 5 stocks with valid data for a meaningful correlation
        overlap = sig_row.index.intersection(ret_row.index)
        if len(overlap) >= 5:
            daily_ic[date] = sig_row[overlap].corr(ret_row[overlap])

    daily_ic = daily_ic.dropna()

    # Summary statistics
    ic_stats = {
        "ic_mean": round(daily_ic.mean(), 4) if len(daily_ic) > 0 else None,
        "ic_std": round(daily_ic.std(), 4) if len(daily_ic) > 0 else None,
        "ic_median": round(daily_ic.median(), 4) if len(daily_ic) > 0 else None,
        "ic_hit_rate": round((daily_ic > 0).mean(), 4) if len(daily_ic) > 0 else None,
        "ic_t_stat": round(
            daily_ic.mean() / (daily_ic.std() / np.sqrt(len(daily_ic))), 2
        ) if len(daily_ic) > 1 and daily_ic.std() > 0 else None,
        "n_days": len(daily_ic),
        "forward_days": forward_days,
    }

    quality = "noise"
    if ic_stats["ic_mean"] is not None:
        abs_ic = abs(ic_stats["ic_mean"])
        if abs_ic > 0.10:
            quality = "suspiciously strong (check for bugs)"
        elif abs_ic > 0.05:
            quality = "strong"
        elif abs_ic > 0.03:
            quality = "meaningful"
        elif abs_ic > 0.01:
            quality = "weak but possibly useful"
    ic_stats["quality"] = quality

    logger.info(f"  ✓ Mean IC: {ic_stats['ic_mean']} ({quality})")
    logger.info(f"  ✓ IC hit rate: {ic_stats['ic_hit_rate']} (fraction of days with positive IC)")

    return daily_ic, ic_stats


# ════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════

def validate_options_signal(options_score, ic_stats=None):
    """Quality checks on the options signal."""
    results = {}

    results["shape"] = options_score.shape
    results["date_range"] = (
        str(options_score.index.min().date()),
        str(options_score.index.max().date()),
    )
    results["tickers"] = len(options_score.columns)

    # Signal distribution
    flat = options_score.values.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) > 0:
        results["mean"] = round(np.mean(flat), 4)
        results["std"] = round(np.std(flat), 4)
        results["pct_positive"] = round((flat > 0).mean() * 100, 1)
        results["pct_negative"] = round((flat < 0).mean() * 100, 1)
        results["pct_near_zero"] = round((np.abs(flat) < 0.1).mean() * 100, 1)
        results["missing_pct"] = round(
            options_score.isnull().sum().sum() / options_score.size * 100, 1
        )

    # Add IC stats if available
    if ic_stats:
        results["ic"] = ic_stats

    return results


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def run(options_features=None, returns=None):
    """
    Main entry point for options signal generation.

    Args:
        options_features: MultiIndex DataFrame from options_loader
                         (or loads from cache)
        returns: Daily returns DataFrame from price_loader
                (or loads from cache). Needed for IC validation.

    Returns: (options_score, iv_signal, pc_signal, daily_ic, validation)
    """
    # Load options features from cache if not provided
    if options_features is None:
        feat_path = RAW_DIR / "options_features.parquet"
        if feat_path.exists():
            options_features = pd.read_parquet(feat_path)
            logger.info(f"Loaded options features: {options_features.shape}")
        else:
            raise FileNotFoundError(
                "No options features found. Run price_loader + options_loader first:\n"
                "  python main.py --stage data --no-sec"
            )

    # Load returns from cache if not provided
    if returns is None:
        prices_path = RAW_DIR / "prices.parquet"
        if prices_path.exists():
            prices = pd.read_parquet(prices_path)
            close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices
            returns = close.pct_change().iloc[1:]
            logger.info(f"Loaded returns: {returns.shape}")
        else:
            logger.warning("No price data found — will skip IC validation")

    # Extract the sub-features from the MultiIndex DataFrame
    iv_percentile = options_features["iv_percentile"]
    pc_ratio = options_features["pc_ratio_proxy"]

    logger.info(f"IV percentile shape: {iv_percentile.shape}")
    logger.info(f"PC ratio shape: {pc_ratio.shape}")

    # Build sub-signals
    iv_signal = build_iv_signal(iv_percentile)
    pc_signal = build_pc_signal(pc_ratio)

    # Combine into final options score
    options_score = build_options_signal(iv_signal, pc_signal)

    # Validate with IC
    daily_ic = pd.Series(dtype=float)
    ic_stats = {}
    if returns is not None:
        daily_ic, ic_stats = compute_ic(options_score, returns, forward_days=5)

        # Also check IC at different horizons for robustness
        logger.info("\n  IC across horizons:")
        for horizon in [1, 5, 10, 21]:
            _, h_stats = compute_ic(options_score, returns, forward_days=horizon)
            logger.info(f"    {horizon:>2}d: IC = {h_stats.get('ic_mean', 'N/A')}")

    # Validate
    validation = validate_options_signal(options_score, ic_stats)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    options_score.to_parquet(PROCESSED_DIR / "options_signal.parquet")
    iv_signal.to_parquet(PROCESSED_DIR / "options_iv_signal.parquet")
    pc_signal.to_parquet(PROCESSED_DIR / "options_pc_signal.parquet")
    if not daily_ic.empty:
        daily_ic.to_frame("ic").to_parquet(PROCESSED_DIR / "options_ic.parquet")
    logger.info(f"Saved options signals to {PROCESSED_DIR}")

    return options_score, iv_signal, pc_signal, daily_ic, validation


if __name__ == "__main__":
    options_score, iv_signal, pc_signal, daily_ic, validation = run()

    print("\n=== Options Signal Validation ===")
    for k, v in validation.items():
        if k != "ic":
            print(f"  {k}: {v}")

    if "ic" in validation:
        print("\n=== Information Coefficient ===")
        for k, v in validation["ic"].items():
            print(f"  {k}: {v}")

    print("\n=== Sample: Options Score (last 10 days, first 5 tickers) ===")
    print(options_score.iloc[-10:, :5].round(3).to_string())
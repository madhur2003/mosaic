"""
MOSAIC -- Meta-Allocator (Week 5)

Dynamically reweights three trading signals using Exponential Gradient Descent.
Combines them into a single composite score, then applies regime scaling.

Inputs:
  - Signal A: NLP tone-shift (daily, from nlp_signal.py)
  - Signal B: Options anomaly (daily, from options_signal.py)
  - Signal C: Momentum (daily, from momentum_signal.py)
  - Regime scalars (daily, from hmm_regime.py)

Process:
  1. Start with equal weights [0.33, 0.33, 0.33]
  2. Every 21 days, compute each signal's rolling 60-day Sharpe contribution
  3. Update weights: w_i = w_i * exp(learning_rate * sharpe_i)
  4. Normalize weights to sum to 1, enforce 10% floor
  5. Composite score = weighted sum of signals
  6. Final score = composite * regime_scalar

Output:
  - Final score per ticker per day [-1, +1]
  - Weight evolution over time (diagnostic)
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

# Allocator parameters
ETA = 0.1               # Learning rate for exponential gradient descent
MIN_WEIGHT = 0.10       # Floor: no signal drops below 10%
REBALANCE_FREQ = 21     # Rebalance every ~1 month (trading days)
SHARPE_WINDOW = 60      # Rolling window for Sharpe contribution


# ════════════════════════════════════════════════════════════════
# LOAD SIGNALS
# ════════════════════════════════════════════════════════════════

def load_signals():
    """
    Load all three signals and returns from disk.

    Each signal is a DataFrame indexed by date, one column per ticker,
    values in [-1, +1].

    Returns from price data are needed to compute Sharpe contributions
    (we need to know if a signal's predictions were correct).
    """
    signals = {}

    # Signal A: NLP (may have sparse coverage -- not all tickers)
    nlp_path = PROCESSED_DIR / "nlp_signals_daily.parquet"
    if nlp_path.exists():
        signals["nlp"] = pd.read_parquet(nlp_path)
        logger.info(f"  Loaded NLP signal: {signals['nlp'].shape}")
    else:
        logger.warning("  NLP signal not found -- will use zeros")
        signals["nlp"] = None

    # Signal B: Options
    opt_path = PROCESSED_DIR / "options_signal.parquet"
    if opt_path.exists():
        signals["options"] = pd.read_parquet(opt_path)
        logger.info(f"  Loaded Options signal: {signals['options'].shape}")
    else:
        raise FileNotFoundError("Options signal not found. Run options_signal.py first.")

    # Signal C: Momentum
    mom_path = PROCESSED_DIR / "momentum_signal.parquet"
    if mom_path.exists():
        signals["momentum"] = pd.read_parquet(mom_path)
        logger.info(f"  Loaded Momentum signal: {signals['momentum'].shape}")
    else:
        raise FileNotFoundError("Momentum signal not found. Run momentum_signal.py first.")

    # Returns (for evaluating signal performance)
    prices_path = RAW_DIR / "prices.parquet"
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices
        returns = close.pct_change().iloc[1:]
        logger.info(f"  Loaded returns: {returns.shape}")
    else:
        raise FileNotFoundError("Price data not found. Run price_loader.py first.")

    # Regime scalars
    regime_path = PROCESSED_DIR / "regime_labels.parquet"
    regime_scalars = None
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_scalars = regime_df["regime_scalar"]
        logger.info(f"  Loaded regime scalars: {len(regime_scalars)} days")
    else:
        logger.warning("  Regime labels not found -- will skip regime scaling")

    return signals, returns, regime_scalars


# ════════════════════════════════════════════════════════════════
# ALIGN SIGNALS
# ════════════════════════════════════════════════════════════════

def align_signals(signals, returns):
    """
    Align all signals and returns to the same date index and ticker columns.

    This is critical because:
      - NLP signal may only cover 6 tickers (sparse SEC filings)
      - Options and momentum cover all 20
      - Date ranges may differ slightly

    We use the intersection of dates and union of tickers.
    Missing signal values are filled with 0 (neutral -- no opinion).
    """
    # Find common dates across all available signals and returns
    date_sets = [set(returns.index)]
    for name, sig in signals.items():
        if sig is not None:
            date_sets.append(set(sig.index))

    common_dates = sorted(set.intersection(*date_sets))
    common_dates = pd.DatetimeIndex(common_dates)

    # Use returns columns as the ticker universe
    tickers = returns.columns.tolist()

    # Align each signal
    aligned = {}
    for name, sig in signals.items():
        if sig is not None:
            # Reindex to common dates and tickers, fill missing with 0
            df = sig.reindex(index=common_dates, columns=tickers).fillna(0)
        else:
            # Signal doesn't exist -- all zeros (no opinion)
            df = pd.DataFrame(0.0, index=common_dates, columns=tickers)
        aligned[name] = df

    aligned_returns = returns.reindex(index=common_dates, columns=tickers).fillna(0)

    logger.info(f"  Aligned to {len(common_dates)} dates x {len(tickers)} tickers")

    return aligned, aligned_returns, common_dates, tickers


# ════════════════════════════════════════════════════════════════
# SHARPE CONTRIBUTION
# ════════════════════════════════════════════════════════════════

def compute_signal_sharpe(signal_df, returns_df, window=SHARPE_WINDOW):
    """
    Compute rolling Sharpe ratio for a single signal.

    Logic: if you traded ONLY this signal (long when positive, short when negative),
    what returns would you have gotten?

      signal_return_t = signal_score_t * actual_return_{t+1}

    Then: sharpe = mean(signal_returns) / std(signal_returns) * sqrt(252)

    This measures how well the signal's predictions aligned with actual moves.

    Returns a Series of rolling Sharpe values indexed by date.
    """
    # Signal return: signal predicts direction, return confirms/denies
    # Use shifted returns (next day) because signal is known today, return happens tomorrow
    next_day_returns = returns_df.shift(-1)

    # Signal return: average across all tickers for each day
    signal_returns = (signal_df * next_day_returns).mean(axis=1)

    # Rolling Sharpe
    rolling_mean = signal_returns.rolling(window=window, min_periods=window).mean()
    rolling_std = signal_returns.rolling(window=window, min_periods=window).std()

    # Annualize
    sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

    return sharpe


# ════════════════════════════════════════════════════════════════
# EXPONENTIAL GRADIENT DESCENT
# ════════════════════════════════════════════════════════════════

def run_egd(aligned_signals, returns, rebalance_freq=REBALANCE_FREQ,
            eta=ETA, min_weight=MIN_WEIGHT, sharpe_window=SHARPE_WINDOW):
    """
    Run Exponential Gradient Descent to dynamically weight signals.

    Process:
      1. Start with equal weights
      2. Every rebalance_freq days:
         a. Compute each signal's rolling Sharpe over the past sharpe_window days
         b. Update: w_i = w_i * exp(eta * sharpe_i)
         c. Enforce minimum weight floor
         d. Normalize so weights sum to 1
      3. Between rebalances, weights stay fixed

    Returns:
      - weights_history: DataFrame with columns [nlp, options, momentum] indexed by date
      - composite_scores: DataFrame of weighted signal combination per ticker per day
    """
    signal_names = list(aligned_signals.keys())
    n_signals = len(signal_names)
    dates = returns.index

    # Initialize equal weights
    weights = np.array([1.0 / n_signals] * n_signals)

    # Track weight evolution
    weight_records = []

    # Compute rolling Sharpe for each signal (full history)
    sharpe_series = {}
    for name in signal_names:
        sharpe_series[name] = compute_signal_sharpe(
            aligned_signals[name], returns, window=sharpe_window
        )

    # Build composite scores day by day
    composite = pd.DataFrame(0.0, index=dates, columns=returns.columns)

    for i, date in enumerate(dates):
        # Record current weights
        weight_records.append({
            "date": date,
            **{name: weights[j] for j, name in enumerate(signal_names)}
        })

        # Compute weighted composite for today
        for j, name in enumerate(signal_names):
            composite.loc[date] += weights[j] * aligned_signals[name].loc[date]

        # Rebalance weights every rebalance_freq days
        if i > 0 and i % rebalance_freq == 0 and i >= sharpe_window:
            # Get each signal's Sharpe at this point
            sharpe_values = []
            for name in signal_names:
                s = sharpe_series[name].loc[date]
                # Handle NaN -- treat as 0 (no information)
                sharpe_values.append(s if not np.isnan(s) else 0.0)

            sharpe_values = np.array(sharpe_values)

            # Exponential gradient update
            weights = weights * np.exp(eta * sharpe_values)

            # Enforce minimum weight floor
            weights = np.maximum(weights, min_weight)

            # Normalize to sum to 1
            weights = weights / weights.sum()

            logger.debug(f"  Rebalance {date.date()}: "
                         f"sharpes={sharpe_values.round(2)}, "
                         f"weights={weights.round(3)}")

    # Build weights DataFrame
    weights_history = pd.DataFrame(weight_records).set_index("date")

    # Clip composite to [-1, 1]
    composite = composite.clip(-1, 1)

    logger.info(f"  Weight evolution: {len(weights_history)} entries")
    logger.info(f"  Final weights: {dict(zip(signal_names, weights.round(3)))}")

    return weights_history, composite


# ════════════════════════════════════════════════════════════════
# REGIME SCALING
# ════════════════════════════════════════════════════════════════

def apply_regime_scaling(composite, regime_scalars):
    """
    Multiply composite scores by regime scalar.

    Risk-On (1.0):      full signal strength
    Transitional (0.5): half strength -- reduce exposure in uncertain markets
    Risk-Off (0.0):     zero -- flat, no new positions during crisis

    This is the system's top-level risk control. Even if all signals
    are screaming "buy," the regime filter can override.
    """
    if regime_scalars is None:
        logger.warning("  No regime scalars -- skipping regime scaling")
        return composite

    # Align regime scalars to composite dates
    scalars_aligned = regime_scalars.reindex(composite.index).ffill().fillna(1.0)

    # Multiply each day's scores by that day's scalar
    final = composite.mul(scalars_aligned, axis=0)

    # Stats
    zero_days = (scalars_aligned == 0).sum()
    half_days = (scalars_aligned == 0.5).sum()
    full_days = (scalars_aligned == 1.0).sum()
    logger.info(f"  Regime scaling applied: {full_days} full / {half_days} half / {zero_days} zero days")

    return final


# ════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════

def validate_allocator(final_scores, weights_history, composite):
    """Quality checks on the meta-allocator output."""
    results = {}

    results["shape"] = final_scores.shape
    results["date_range"] = (
        str(final_scores.index.min().date()),
        str(final_scores.index.max().date()),
    )

    # Final score distribution
    flat = final_scores.values.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) > 0:
        results["final_mean"] = round(np.mean(flat), 4)
        results["final_std"] = round(np.std(flat), 4)
        results["final_pct_positive"] = round((flat > 0).mean() * 100, 1)
        results["final_pct_zero"] = round((flat == 0).mean() * 100, 1)

    # Weight stability
    if not weights_history.empty:
        results["final_weights"] = {
            col: round(weights_history[col].iloc[-1], 3)
            for col in weights_history.columns
        }
        # How much did weights change over time?
        weight_std = weights_history.std()
        results["weight_volatility"] = {
            col: round(weight_std[col], 4)
            for col in weights_history.columns
        }

    return results


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def run():
    """
    Main entry point for the meta-allocator.

    Loads all signals, runs EGD, applies regime scaling.

    Returns: (final_scores, weights_history, composite, validation)
    """
    logger.info("Loading signals...")
    signals, returns, regime_scalars = load_signals()

    logger.info("Aligning signals...")
    aligned, aligned_returns, dates, tickers = align_signals(signals, returns)

    logger.info("Running Exponential Gradient Descent...")
    weights_history, composite = run_egd(aligned, aligned_returns)

    logger.info("Applying regime scaling...")
    final_scores = apply_regime_scaling(composite, regime_scalars)

    validation = validate_allocator(final_scores, weights_history, composite)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_scores.to_parquet(PROCESSED_DIR / "final_scores.parquet")
    composite.to_parquet(PROCESSED_DIR / "composite_scores.parquet")
    weights_history.to_parquet(PROCESSED_DIR / "weight_evolution.parquet")
    logger.info(f"Saved allocator outputs to {PROCESSED_DIR}")

    return final_scores, weights_history, composite, validation


if __name__ == "__main__":
    final, weights, composite, validation = run()

    print("\n=== Meta-Allocator Validation ===")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    print("\n=== Weight Evolution (first and last 5 entries) ===")
    print("First 5:")
    print(weights.head().round(3).to_string())
    print("\nLast 5:")
    print(weights.tail().round(3).to_string())

    print("\n=== Final Scores (last 10 days, first 5 tickers) ===")
    print(final.iloc[-10:, :5].round(3).to_string())
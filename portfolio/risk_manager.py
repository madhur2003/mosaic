# """
# MOSAIC -- Portfolio & Risk Manager (Week 6)

# Converts signal scores into position sizes with hard risk limits.

# Process:
#   1. Fractional Kelly sizing (25%) based on signal score and return variance
#   2. Position cap: 5% max per stock
#   3. Sector cap: 25% max per sector
#   4. Drawdown circuit breaker: halt at -10%, liquidate at -15%
#   5. Transaction cost modeling: 5bps slippage + $0.005/share commission

# Input: final_scores from meta-allocator (per ticker per day, [-1, +1])
# Output: portfolio positions, daily P&L, drawdown tracking
# """

# import sys
# import logging
# from pathlib import Path

# import numpy as np
# import pandas as pd

# sys.path.insert(0, str(Path(__file__).parent.parent))
# from config import RAW_DIR, PROCESSED_DIR

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# # Risk parameters
# KELLY_FRACTION = 0.25
# MAX_POSITION_PCT = 0.05
# MAX_SECTOR_PCT = 0.25
# DRAWDOWN_HALT = -0.10
# DRAWDOWN_LIQUIDATE = -0.15
# SLIPPAGE_BPS = 5
# COMMISSION_PER_SHARE = 0.005
# INITIAL_CAPITAL = 1_000_000  # $1M starting portfolio

# # GICS sector mapping for our 20-ticker universe
# SECTOR_MAP = {
#     "AAPL": "Technology",
#     "MSFT": "Technology",
#     "NVDA": "Technology",
#     "GOOGL": "Technology",
#     "JNJ": "Healthcare",
#     "UNH": "Healthcare",
#     "PFE": "Healthcare",
#     "JPM": "Financials",
#     "GS": "Financials",
#     "BAC": "Financials",
#     "AMZN": "Consumer",
#     "TSLA": "Consumer",
#     "WMT": "Consumer",
#     "XOM": "Energy",
#     "CVX": "Energy",
#     "CAT": "Industrials",
#     "BA": "Industrials",
#     "META": "Communication",
#     "DIS": "Communication",
#     "LIN": "Materials",
# }


# # ════════════════════════════════════════════════════════════════
# # KELLY POSITION SIZING
# # ════════════════════════════════════════════════════════════════

# def compute_kelly_sizes(scores, returns, lookback=60):
#     """
#     Fractional Kelly position sizing.

#     Full Kelly: f* = expected_return / variance
#     We use 25% Kelly: f = 0.25 * (expected_return / variance)

#     For each ticker on each day:
#       - expected_return = signal_score * rolling_mean_return (directional view)
#       - variance = rolling variance of returns

#     The signal score acts as our "conviction" -- stronger score = larger position.
#     Rolling stats ensure we adapt to changing volatility.

#     Output: DataFrame of position weights (fraction of portfolio per ticker).
#     Positive = long, negative = short.
#     """
#     # Rolling return stats
#     rolling_mean = returns.rolling(window=lookback, min_periods=lookback).mean()
#     rolling_var = returns.rolling(window=lookback, min_periods=lookback).var()

#     # Avoid division by zero
#     rolling_var = rolling_var.replace(0, np.nan)

#     # Kelly sizing: score gives direction and conviction, variance scales risk
#     # Simplified: position = kelly_fraction * score * (mean / var)
#     # But mean/var can be noisy, so we use score directly scaled by 1/vol
#     rolling_vol = np.sqrt(rolling_var)

#     # Position weight = kelly_fraction * score / vol (higher vol = smaller position)
#     raw_sizes = KELLY_FRACTION * scores / rolling_vol.replace(0, np.nan)

#     # Replace any infinities or NaN with 0
#     raw_sizes = raw_sizes.replace([np.inf, -np.inf], 0).fillna(0)

#     return raw_sizes


# # ════════════════════════════════════════════════════════════════
# # POSITION LIMITS
# # ════════════════════════════════════════════════════════════════

# def apply_position_limits(positions, max_pos=MAX_POSITION_PCT):
#     """
#     Cap individual positions at max_pos (5%) of portfolio.

#     If Kelly says put 12% in TSLA, we clip to 5%.
#     This prevents any single stock from dominating the portfolio.
#     """
#     clipped = positions.clip(-max_pos, max_pos)

#     n_clipped = ((positions.abs() > max_pos) & (positions != 0)).sum().sum()
#     if n_clipped > 0:
#         logger.info(f"  Position limit: clipped {n_clipped} positions to +/-{max_pos*100}%")

#     return clipped


# def apply_sector_limits(positions, max_sector=MAX_SECTOR_PCT):
#     """
#     Cap total exposure per sector at max_sector (25%).

#     If all 4 tech stocks want 5% each (=20%), that's fine.
#     But if they want 8% each (=32%), we scale them down proportionally
#     so total tech = 25%.
#     """
#     result = positions.copy()

#     # Group tickers by sector
#     sectors = {}
#     for ticker, sector in SECTOR_MAP.items():
#         if sector not in sectors:
#             sectors[sector] = []
#         if ticker in positions.columns:
#             sectors[sector].append(ticker)

#     for date in positions.index:
#         for sector, tickers in sectors.items():
#             if not tickers:
#                 continue

#             sector_exposure = result.loc[date, tickers].abs().sum()

#             if sector_exposure > max_sector:
#                 # Scale down proportionally
#                 scale = max_sector / sector_exposure
#                 result.loc[date, tickers] *= scale

#     return result


# # ════════════════════════════════════════════════════════════════
# # DRAWDOWN CIRCUIT BREAKER
# # ════════════════════════════════════════════════════════════════

# def apply_circuit_breaker(positions, portfolio_value):
#     """
#     Drawdown circuit breaker.

#     Tracks rolling peak of portfolio value and computes drawdown:
#       drawdown = (current - peak) / peak

#     Rules:
#       drawdown > -10%: normal, full trading
#       drawdown <= -10%: HALT -- no new positions (keep existing ones)
#       drawdown <= -15%: LIQUIDATE -- close everything, go to cash

#     Returns: (adjusted_positions, drawdown_series, halt_dates, liquidate_dates)
#     """
#     result = positions.copy()

#     # Track peak and drawdown
#     peak = portfolio_value.expanding().max()
#     drawdown = (portfolio_value - peak) / peak

#     halt_dates = []
#     liquidate_dates = []

#     for date in positions.index:
#         dd = drawdown.get(date, 0)

#         if dd <= DRAWDOWN_LIQUIDATE:
#             # LIQUIDATE: zero everything
#             result.loc[date] = 0
#             liquidate_dates.append(date)
#         elif dd <= DRAWDOWN_HALT:
#             # HALT: no new positions, keep existing
#             # Only allow positions that were already held (same sign as previous day)
#             if date != positions.index[0]:
#                 prev_date = positions.index[positions.index.get_loc(date) - 1]
#                 prev_pos = result.loc[prev_date]
#                 current_pos = result.loc[date]

#                 # Zero out any NEW positions (sign changed or new entry)
#                 for ticker in positions.columns:
#                     if prev_pos[ticker] == 0 and current_pos[ticker] != 0:
#                         result.loc[date, ticker] = 0
#                     elif np.sign(prev_pos[ticker]) != np.sign(current_pos[ticker]):
#                         result.loc[date, ticker] = 0

#             halt_dates.append(date)

#     logger.info(f"  Circuit breaker: {len(halt_dates)} halt days, {len(liquidate_dates)} liquidate days")

#     return result, drawdown, halt_dates, liquidate_dates


# # ════════════════════════════════════════════════════════════════
# # TRANSACTION COSTS
# # ════════════════════════════════════════════════════════════════

# def compute_transaction_costs(positions, prices):
#     """
#     Model transaction costs: slippage + commission.

#     Slippage: 5 basis points on each trade
#       If you trade $10,000 worth, you lose $10,000 * 0.0005 = $5

#     Commission: $0.005 per share
#       Approximated as: position_change_dollars / avg_price * 0.005

#     Costs only apply when positions CHANGE (turnover).
#     Holding a position costs nothing.

#     Returns: DataFrame of daily transaction costs per ticker (as fraction of portfolio).
#     """
#     # Position changes (turnover)
#     turnover = positions.diff().abs()
#     turnover.iloc[0] = positions.iloc[0].abs()  # First day: all positions are new

#     # Slippage cost: turnover * slippage_rate
#     slippage_cost = turnover * (SLIPPAGE_BPS / 10000)

#     # Commission cost (approximate: assume $100 avg price)
#     # Real implementation would use actual prices
#     commission_cost = turnover * COMMISSION_PER_SHARE / 100

#     total_cost = slippage_cost + commission_cost

#     return total_cost


# # ════════════════════════════════════════════════════════════════
# # PORTFOLIO SIMULATION
# # ════════════════════════════════════════════════════════════════

# def simulate_portfolio(positions, returns, initial_capital=INITIAL_CAPITAL):
#     """
#     Simulate portfolio performance day by day.

#     For each day:
#       portfolio_return = sum(position_weight_i * stock_return_i)
#       portfolio_value = previous_value * (1 + portfolio_return - transaction_cost)

#     Returns: (portfolio_value_series, daily_returns_series)
#     """
#     # Align positions and returns
#     common_idx = positions.index.intersection(returns.index)
#     common_cols = positions.columns.intersection(returns.columns)

#     pos = positions.loc[common_idx, common_cols]
#     ret = returns.loc[common_idx, common_cols]

#     # Daily portfolio return = sum of (position * return) across tickers
#     daily_portfolio_return = (pos * ret).sum(axis=1)

#     # Transaction costs
#     turnover = pos.diff().abs()
#     turnover.iloc[0] = pos.iloc[0].abs()
#     daily_cost = (turnover * SLIPPAGE_BPS / 10000).sum(axis=1)

#     # Net return
#     net_return = daily_portfolio_return - daily_cost

#     # Portfolio value
#     portfolio_value = pd.Series(index=common_idx, dtype=float)
#     portfolio_value.iloc[0] = initial_capital * (1 + net_return.iloc[0])

#     for i in range(1, len(common_idx)):
#         portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + net_return.iloc[i])

#     return portfolio_value, net_return


# # ════════════════════════════════════════════════════════════════
# # VALIDATION
# # ════════════════════════════════════════════════════════════════

# def validate_portfolio(positions, portfolio_value, net_returns, drawdown,
#                        halt_dates, liquidate_dates):
#     """Quality checks on the portfolio."""
#     results = {}

#     results["date_range"] = (
#         str(positions.index.min().date()),
#         str(positions.index.max().date()),
#     )
#     results["trading_days"] = len(positions)
#     results["initial_capital"] = INITIAL_CAPITAL
#     results["final_value"] = round(portfolio_value.iloc[-1], 2)
#     results["total_return_pct"] = round(
#         (portfolio_value.iloc[-1] / INITIAL_CAPITAL - 1) * 100, 2
#     )

#     # Risk metrics
#     if len(net_returns) > 0:
#         ann_return = net_returns.mean() * 252
#         ann_vol = net_returns.std() * np.sqrt(252)
#         sharpe = ann_return / ann_vol if ann_vol > 0 else 0

#         results["annualized_return_pct"] = round(ann_return * 100, 2)
#         results["annualized_vol_pct"] = round(ann_vol * 100, 2)
#         results["sharpe_ratio"] = round(sharpe, 3)

#         # Sortino (downside deviation only)
#         downside = net_returns[net_returns < 0]
#         downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
#         sortino = ann_return / downside_vol if downside_vol > 0 else 0
#         results["sortino_ratio"] = round(sortino, 3)

#         # Max drawdown
#         results["max_drawdown_pct"] = round(drawdown.min() * 100, 2)

#         # Win rate
#         results["win_rate_pct"] = round((net_returns > 0).mean() * 100, 1)

#     # Position stats
#     flat = positions.values.flatten()
#     flat = flat[flat != 0]
#     if len(flat) > 0:
#         results["avg_position_size_pct"] = round(np.mean(np.abs(flat)) * 100, 3)
#         results["max_position_size_pct"] = round(np.max(np.abs(flat)) * 100, 3)

#     # Circuit breaker
#     results["halt_days"] = len(halt_dates)
#     results["liquidate_days"] = len(liquidate_dates)

#     # Turnover
#     turnover = positions.diff().abs().sum(axis=1)
#     results["avg_daily_turnover_pct"] = round(turnover.mean() * 100, 2)

#     return results


# # ════════════════════════════════════════════════════════════════
# # ENTRY POINT
# # ════════════════════════════════════════════════════════════════

# def run(final_scores=None, returns=None):
#     """
#     Main entry point for portfolio construction and risk management.

#     Args:
#         final_scores: DataFrame from meta-allocator (or loads from cache)
#         returns: Daily returns DataFrame (or loads from cache)

#     Returns: (positions, portfolio_value, net_returns, drawdown, validation)
#     """
#     # Load from cache if not provided
#     if final_scores is None:
#         scores_path = PROCESSED_DIR / "final_scores.parquet"
#         if scores_path.exists():
#             final_scores = pd.read_parquet(scores_path)
#             logger.info(f"Loaded final scores: {final_scores.shape}")
#         else:
#             raise FileNotFoundError(
#                 "No final scores found. Run meta_allocator.py first:\n"
#                 "  python main.py --stage allocator"
#             )

#     if returns is None:
#         prices_path = RAW_DIR / "prices.parquet"
#         if prices_path.exists():
#             prices = pd.read_parquet(prices_path)
#             close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices
#             returns = close.pct_change().iloc[1:]
#             logger.info(f"Loaded returns: {returns.shape}")
#         else:
#             raise FileNotFoundError("No price data found. Run price_loader.py first.")

#     # Align
#     common_idx = final_scores.index.intersection(returns.index)
#     common_cols = final_scores.columns.intersection(returns.columns)
#     scores = final_scores.loc[common_idx, common_cols]
#     ret = returns.loc[common_idx, common_cols]

#     # Step 1: Kelly sizing
#     logger.info("Step 1: Fractional Kelly position sizing...")
#     raw_positions = compute_kelly_sizes(scores, ret)
#     logger.info(f"  Raw positions: mean abs = {raw_positions.abs().mean().mean():.4f}")

#     # Step 2: Position limits
#     logger.info("Step 2: Applying position limits (5% per stock)...")
#     sized_positions = apply_position_limits(raw_positions)

#     # Step 3: Sector limits
#     logger.info("Step 3: Applying sector limits (25% per sector)...")
#     sector_limited = apply_sector_limits(sized_positions)

#     # Step 4: Initial portfolio simulation (needed for drawdown calculation)
#     logger.info("Step 4: Simulating portfolio for drawdown tracking...")
#     portfolio_value, net_returns = simulate_portfolio(sector_limited, ret)

#     # Step 5: Circuit breaker
#     logger.info("Step 5: Applying drawdown circuit breaker...")
#     final_positions, drawdown, halt_dates, liquidate_dates = apply_circuit_breaker(
#         sector_limited, portfolio_value
#     )

#     # Step 6: Re-simulate with circuit breaker applied
#     logger.info("Step 6: Final portfolio simulation...")
#     portfolio_value, net_returns = simulate_portfolio(final_positions, ret)

#     # Recompute drawdown on final portfolio
#     peak = portfolio_value.expanding().max()
#     drawdown = (portfolio_value - peak) / peak

#     # Validate
#     validation = validate_portfolio(
#         final_positions, portfolio_value, net_returns, drawdown,
#         halt_dates, liquidate_dates
#     )

#     # Save
#     PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
#     final_positions.to_parquet(PROCESSED_DIR / "positions.parquet")

#     portfolio_df = pd.DataFrame({
#         "portfolio_value": portfolio_value,
#         "daily_return": net_returns,
#         "drawdown": drawdown,
#     })
#     portfolio_df.to_parquet(PROCESSED_DIR / "portfolio_results.parquet")
#     logger.info(f"Saved portfolio results to {PROCESSED_DIR}")

#     return final_positions, portfolio_value, net_returns, drawdown, validation


# if __name__ == "__main__":
#     positions, pv, nr, dd, validation = run()

#     print("\n=== Portfolio & Risk Validation ===")
#     for k, v in validation.items():
#         print(f"  {k}: {v}")

#     print(f"\n=== Equity Curve ===")
#     print(f"  Start: ${INITIAL_CAPITAL:,.0f}")
#     print(f"  End:   ${pv.iloc[-1]:,.0f}")
#     print(f"  Peak:  ${pv.max():,.0f}")
#     print(f"  Trough: ${pv.min():,.0f}")

#     print(f"\n=== Positions (last 5 days, first 5 tickers) ===")
#     print(positions.iloc[-5:, :5].round(4).to_string())

"""
MOSAIC -- Portfolio & Risk Manager (Week 6, v3)

Core insight: with proxy-based signals that have weak standalone alpha,
the system's edge comes from REGIME-AWARE BETA EXPOSURE, not from
signal-driven stock picking.

Strategy:
  1. Use signals to TILT positions (not drive them entirely)
  2. Maintain a baseline long exposure that captures market upside
  3. Scale total exposure by regime (reduce in Risk-Off, full in Risk-On)
  4. Hard risk limits prevent catastrophic drawdowns

Position = baseline_weight + signal_tilt
  baseline_weight = 1/N equal weight (captures market beta)
  signal_tilt = small adjustment based on composite signal score
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

INITIAL_CAPITAL = 1_000_000

# Tuned parameters
BASELINE_WEIGHT = 0.04        # 4% equal weight per stock (20 stocks = 80% invested)
SIGNAL_TILT_SCALE = 0.015     # Signal can adjust position by +/-1.5%
MAX_POSITION_PCT = 0.06       # Hard cap at 6% per stock
MAX_SECTOR_PCT = 0.25         # 25% per sector
DRAWDOWN_HALT = -0.18         # Reduce exposure at -18%
DRAWDOWN_LIQUIDATE = -0.25    # Heavy reduction at -25%
SLIPPAGE_BPS = 3              # 3 basis points per trade
REBALANCE_FREQ = 5            # Rebalance every 5 days (weekly) to reduce turnover

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "JPM": "Financials", "GS": "Financials", "BAC": "Financials",
    "AMZN": "Consumer", "TSLA": "Consumer", "WMT": "Consumer",
    "XOM": "Energy", "CVX": "Energy",
    "CAT": "Industrials", "BA": "Industrials",
    "META": "Communication", "DIS": "Communication",
    "LIN": "Materials",
}


def build_positions(scores, regime_scalars, returns):
    """
    Build position weights using baseline + signal tilt approach.

    For each stock each day:
      position = (baseline + signal_tilt) * regime_scalar

    baseline = 0.04 (4% per stock, capturing market beta)
    signal_tilt = signal_score * 0.015 (small adjustment)
    regime_scalar = 1.0 (Risk-On) / 0.6 (Transitional) / 0.3 (Risk-Off)

    This ensures:
    - System is always somewhat long (captures market upside)
    - Signals TILT the portfolio toward stocks with better scores
    - Regime reduces total exposure during crises
    """
    logger.info("  Building positions: baseline + signal tilt...")

    positions = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    # Get regime scalars aligned to score dates
    if regime_scalars is not None:
        scalars = regime_scalars.reindex(scores.index).ffill().fillna(0.7)
    else:
        scalars = pd.Series(0.7, index=scores.index)

    # Build positions with weekly rebalancing
    current_positions = pd.Series(BASELINE_WEIGHT, index=scores.columns)

    for i, date in enumerate(scores.index):
        if i % REBALANCE_FREQ == 0:
            # Rebalance day: update target positions
            signal_row = scores.loc[date].fillna(0)
            scalar = scalars.get(date, 0.7)

            # Position = (baseline + signal_tilt) * regime_scalar
            # All positions are LONG (baseline is positive)
            # Signal tilt can make some stocks more or less weighted
            target = (BASELINE_WEIGHT + signal_row * SIGNAL_TILT_SCALE) * scalar

            # Ensure all positions are non-negative (long-only is safer with weak signals)
            target = target.clip(0, MAX_POSITION_PCT)

            current_positions = target

        positions.loc[date] = current_positions

    # Apply sector limits
    positions = apply_sector_limits(positions)

    total_exposure = positions.sum(axis=1)
    logger.info(f"  Avg total exposure: {total_exposure.mean()*100:.1f}%")
    logger.info(f"  Min exposure: {total_exposure.min()*100:.1f}%, Max: {total_exposure.max()*100:.1f}%")

    return positions


def apply_sector_limits(positions, max_sector=MAX_SECTOR_PCT):
    result = positions.copy()

    sectors = {}
    for ticker, sector in SECTOR_MAP.items():
        if sector not in sectors:
            sectors[sector] = []
        if ticker in positions.columns:
            sectors[sector].append(ticker)

    for date in positions.index:
        for sector, tickers in sectors.items():
            if not tickers:
                continue
            sector_exposure = result.loc[date, tickers].abs().sum()
            if sector_exposure > max_sector:
                scale = max_sector / sector_exposure
                result.loc[date, tickers] *= scale

    return result


def simulate_portfolio(positions, returns, initial_capital=INITIAL_CAPITAL):
    common_idx = positions.index.intersection(returns.index)
    common_cols = positions.columns.intersection(returns.columns)

    pos = positions.loc[common_idx, common_cols]
    ret = returns.loc[common_idx, common_cols]

    # Portfolio return = sum of (position * return) across tickers
    daily_portfolio_return = (pos * ret).sum(axis=1)

    # Transaction costs (only on position changes)
    turnover = pos.diff().abs()
    turnover.iloc[0] = 0  # No cost on day 1 (we're building from cash gradually)
    daily_cost = (turnover * SLIPPAGE_BPS / 10000).sum(axis=1)

    net_return = daily_portfolio_return - daily_cost

    # Build equity curve
    portfolio_value = initial_capital * (1 + net_return).cumprod()

    return portfolio_value, net_return


def apply_circuit_breaker(positions, portfolio_value):
    """
    Soft circuit breaker: reduces exposure gradually, doesn't go to zero.

    At -18% drawdown: reduce all positions by 50%
    At -25% drawdown: reduce all positions by 75%
    This keeps us in the market (capturing recovery) but at reduced risk.
    """
    result = positions.copy()

    peak = portfolio_value.expanding().max()
    drawdown = (portfolio_value - peak) / peak

    halt_days = 0
    liquidate_days = 0

    for date in positions.index:
        dd = drawdown.get(date, 0)
        if dd <= DRAWDOWN_LIQUIDATE:
            result.loc[date] *= 0.25
            liquidate_days += 1
        elif dd <= DRAWDOWN_HALT:
            result.loc[date] *= 0.5
            halt_days += 1

    logger.info(f"  Circuit breaker: {halt_days} reduced-50% days, {liquidate_days} reduced-75% days")
    return result, drawdown, halt_days, liquidate_days


def validate_portfolio(positions, portfolio_value, net_returns, drawdown,
                       halt_days, liquidate_days):
    results = {}

    results["date_range"] = (
        str(positions.index.min().date()),
        str(positions.index.max().date()),
    )
    results["trading_days"] = len(positions)
    results["initial_capital"] = INITIAL_CAPITAL
    results["final_value"] = round(portfolio_value.iloc[-1], 2)
    results["total_return_pct"] = round(
        (portfolio_value.iloc[-1] / INITIAL_CAPITAL - 1) * 100, 2
    )

    if len(net_returns) > 0:
        ann_return = net_returns.mean() * 252
        ann_vol = net_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        results["annualized_return_pct"] = round(ann_return * 100, 2)
        results["annualized_vol_pct"] = round(ann_vol * 100, 2)
        results["sharpe_ratio"] = round(sharpe, 3)

        downside = net_returns[net_returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0
        results["sortino_ratio"] = round(sortino, 3)

        results["max_drawdown_pct"] = round(drawdown.min() * 100, 2)
        results["win_rate_pct"] = round((net_returns > 0).mean() * 100, 1)

    flat = positions.values.flatten()
    flat = flat[flat != 0]
    if len(flat) > 0:
        results["avg_position_size_pct"] = round(np.mean(np.abs(flat)) * 100, 3)
        results["max_position_size_pct"] = round(np.max(np.abs(flat)) * 100, 3)

    results["halt_days"] = halt_days
    results["liquidate_days"] = liquidate_days

    turnover = positions.diff().abs().sum(axis=1)
    results["avg_daily_turnover_pct"] = round(turnover.mean() * 100, 2)

    return results


def run(final_scores=None, returns=None):
    if final_scores is None:
        scores_path = PROCESSED_DIR / "final_scores.parquet"
        if scores_path.exists():
            final_scores = pd.read_parquet(scores_path)
            logger.info(f"Loaded final scores: {final_scores.shape}")
        else:
            raise FileNotFoundError("No final scores. Run meta_allocator.py first.")

    if returns is None:
        prices_path = RAW_DIR / "prices.parquet"
        if prices_path.exists():
            prices = pd.read_parquet(prices_path)
            close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices
            returns = close.pct_change().iloc[1:]
            logger.info(f"Loaded returns: {returns.shape}")
        else:
            raise FileNotFoundError("No price data.")

    # Load regime scalars
    regime_scalars = None
    regime_path = PROCESSED_DIR / "regime_labels.parquet"
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_scalars = regime_df["regime_scalar"]
        logger.info(f"Loaded regime scalars: {len(regime_scalars)} days")

    # Align
    common_idx = final_scores.index.intersection(returns.index)
    common_cols = final_scores.columns.intersection(returns.columns)
    scores = final_scores.loc[common_idx, common_cols]
    ret = returns.loc[common_idx, common_cols]

    # Step 1: Build positions (baseline + tilt)
    logger.info("Step 1: Building positions (baseline + signal tilt)...")
    raw_positions = build_positions(scores, regime_scalars, ret)

    # Step 2: Initial simulation
    logger.info("Step 2: Simulating portfolio...")
    portfolio_value, net_returns = simulate_portfolio(raw_positions, ret)

    # Step 3: Circuit breaker
    logger.info("Step 3: Applying circuit breaker...")
    final_positions, drawdown, halt_days, liquidate_days = apply_circuit_breaker(
        raw_positions, portfolio_value
    )

    # Step 4: Final simulation with circuit breaker
    logger.info("Step 4: Final simulation...")
    portfolio_value, net_returns = simulate_portfolio(final_positions, ret)

    # Recompute drawdown
    peak = portfolio_value.expanding().max()
    drawdown = (portfolio_value - peak) / peak

    # Validate
    validation = validate_portfolio(
        final_positions, portfolio_value, net_returns, drawdown,
        halt_days, liquidate_days
    )

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_positions.to_parquet(PROCESSED_DIR / "positions.parquet")

    portfolio_df = pd.DataFrame({
        "portfolio_value": portfolio_value,
        "daily_return": net_returns,
        "drawdown": drawdown,
    })
    portfolio_df.to_parquet(PROCESSED_DIR / "portfolio_results.parquet")
    logger.info(f"Saved portfolio results to {PROCESSED_DIR}")

    return final_positions, portfolio_value, net_returns, drawdown, validation


if __name__ == "__main__":
    positions, pv, nr, dd, validation = run()

    print("\n=== Portfolio & Risk Validation ===")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    print(f"\n=== Equity Curve ===")
    print(f"  Start: ${INITIAL_CAPITAL:,.0f}")
    print(f"  End:   ${pv.iloc[-1]:,.0f}")
    print(f"  Peak:  ${pv.max():,.0f}")
    print(f"  Trough: ${pv.min():,.0f}")
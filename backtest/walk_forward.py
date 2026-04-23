"""
MOSAIC -- Walk-Forward Backtesting Engine (Week 7)

Tests the full system across 2018-2024 using rolling windows:
  - Train window: 24 months (expanding)
  - Test window: 6 months
  - Step: 6 months forward

Compares against SPY benchmark on all key metrics.
Includes stress test analysis for COVID (2020) and rate hike bear (2022).

CRITICAL: No lookahead bias. At every point, the system only uses
data available up to that date. This is enforced by the walk-forward
structure and expanding-window HMM training.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR, BACKTEST_START

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ════════════════════════════════════════════════════════════════

def compute_metrics(returns, name="Strategy"):
    """
    Compute all key performance metrics from a return series.

    Returns a dict with:
      - Annualized return, volatility
      - Sharpe, Sortino, Calmar ratios
      - Max drawdown (and date)
      - Win rate
      - Best/worst day
    """
    if len(returns) == 0 or returns.std() == 0:
        return {"name": name, "error": "Insufficient data"}

    # Strip NaN
    returns = returns.dropna()

    # Annualized return and vol
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino ratio (uses downside deviation)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    # Cumulative return and drawdown
    cum_return = (1 + returns).cumprod()
    peak = cum_return.expanding().max()
    drawdown = (cum_return - peak) / peak

    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    # Calmar ratio (return / max drawdown)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (returns > 0).mean()

    # Total return
    total_return = cum_return.iloc[-1] - 1

    return {
        "name": name,
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_date": str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
        "win_rate_pct": round(win_rate * 100, 1),
        "best_day_pct": round(returns.max() * 100, 2),
        "worst_day_pct": round(returns.min() * 100, 2),
        "trading_days": len(returns),
    }


# ════════════════════════════════════════════════════════════════
# BENCHMARK
# ════════════════════════════════════════════════════════════════

def load_benchmark_returns():
    """Load SPY daily returns for comparison."""
    bench_path = RAW_DIR / "benchmark.parquet"
    if not bench_path.exists():
        raise FileNotFoundError("No benchmark data. Run price_loader.py first.")

    bench = pd.read_parquet(bench_path)
    bench_col = bench.iloc[:, 0]
    bench_returns = bench_col.pct_change().dropna()

    return bench_returns


# ════════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ════════════════════════════════════════════════════════════════

def generate_walk_forward_windows(dates, train_months=24, test_months=6, step_months=6):
    """
    Generate train/test date windows for walk-forward validation.

    Starting from BACKTEST_START:
      Window 1: train = [start - 24mo, start], test = [start, start + 6mo]
      Window 2: train = [start - 24mo, start + 6mo], test = [start + 6mo, start + 12mo]
      ...

    Each window returns (train_start, train_end, test_start, test_end) dates.
    """
    backtest_start = pd.Timestamp(BACKTEST_START)
    all_dates = pd.DatetimeIndex(dates)

    windows = []
    test_start = backtest_start

    while test_start < all_dates.max():
        train_start = test_start - pd.DateOffset(months=train_months)
        test_end = test_start + pd.DateOffset(months=test_months)

        # Clip to available data
        train_start = max(train_start, all_dates.min())
        test_end = min(test_end, all_dates.max())

        if test_start < all_dates.max():
            windows.append({
                "train_start": train_start,
                "train_end": test_start,
                "test_start": test_start,
                "test_end": test_end,
            })

        test_start = test_end

    return windows


def run_walk_forward(strategy_returns, benchmark_returns):
    """
    Run walk-forward analysis on pre-computed strategy returns.

    Since our signals are already generated with expanding-window
    (no lookahead bias in HMM, no future data in signals), the
    walk-forward here is about EVALUATING performance in non-overlapping
    test windows, not re-training.

    Returns per-window metrics and aggregate metrics.
    """
    windows = generate_walk_forward_windows(strategy_returns.index)

    logger.info(f"Walk-forward: {len(windows)} windows")

    window_results = []

    for i, w in enumerate(windows):
        # Get returns for this test window
        mask = (strategy_returns.index >= w["test_start"]) & \
               (strategy_returns.index < w["test_end"])
        window_strat = strategy_returns[mask]

        bench_mask = (benchmark_returns.index >= w["test_start"]) & \
                     (benchmark_returns.index < w["test_end"])
        window_bench = benchmark_returns[bench_mask]

        if len(window_strat) < 10:
            continue

        strat_metrics = compute_metrics(window_strat, f"Window {i+1} Strategy")
        bench_metrics = compute_metrics(window_bench, f"Window {i+1} SPY")

        result = {
            "window": i + 1,
            "test_start": str(w["test_start"].date()),
            "test_end": str(w["test_end"].date()),
            "days": len(window_strat),
            "strat_return_pct": strat_metrics.get("total_return_pct", 0),
            "bench_return_pct": bench_metrics.get("total_return_pct", 0),
            "strat_sharpe": strat_metrics.get("sharpe", 0),
            "bench_sharpe": bench_metrics.get("sharpe", 0),
            "strat_max_dd_pct": strat_metrics.get("max_drawdown_pct", 0),
            "bench_max_dd_pct": bench_metrics.get("max_drawdown_pct", 0),
        }

        # Did strategy beat benchmark?
        result["outperformed"] = result["strat_return_pct"] > result["bench_return_pct"]

        window_results.append(result)

        logger.info(f"  Window {i+1} ({result['test_start']} to {result['test_end']}): "
                     f"strat={result['strat_return_pct']}% vs bench={result['bench_return_pct']}% "
                     f"{'✓' if result['outperformed'] else '✗'}")

    return pd.DataFrame(window_results)


# ════════════════════════════════════════════════════════════════
# STRESS TESTS
# ════════════════════════════════════════════════════════════════

def run_stress_tests(strategy_returns, benchmark_returns):
    """
    Analyze performance during specific crisis periods.

    COVID crash: Feb 19 - Mar 23, 2020 (peak to trough)
    2022 bear: Jan 3 - Oct 12, 2022 (rate hike driven decline)
    """
    stress_periods = {
        "COVID Crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
        "COVID Recovery (Mar-Jun 2020)": ("2020-03-23", "2020-06-08"),
        "2022 Bear Market (Jan-Oct)": ("2022-01-03", "2022-10-12"),
        "2023 Recovery (Jan-Jul)": ("2023-01-03", "2023-07-31"),
    }

    results = []

    for name, (start, end) in stress_periods.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        strat_mask = (strategy_returns.index >= start_dt) & (strategy_returns.index <= end_dt)
        bench_mask = (benchmark_returns.index >= start_dt) & (benchmark_returns.index <= end_dt)

        strat_period = strategy_returns[strat_mask]
        bench_period = benchmark_returns[bench_mask]

        if len(strat_period) < 5:
            continue

        strat_total = ((1 + strat_period).cumprod().iloc[-1] - 1) * 100
        bench_total = ((1 + bench_period).cumprod().iloc[-1] - 1) * 100

        results.append({
            "period": name,
            "days": len(strat_period),
            "strategy_return_pct": round(strat_total, 2),
            "benchmark_return_pct": round(bench_total, 2),
            "outperformed": strat_total > bench_total,
        })

        logger.info(f"  {name}: strat={strat_total:.1f}% vs bench={bench_total:.1f}%")

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════
# EQUITY CURVE
# ════════════════════════════════════════════════════════════════

def build_equity_curves(strategy_returns, benchmark_returns, initial=1_000_000):
    """Build cumulative equity curves for strategy and benchmark."""
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)

    strat = strategy_returns.reindex(common_idx).fillna(0)
    bench = benchmark_returns.reindex(common_idx).fillna(0)

    strat_equity = initial * (1 + strat).cumprod()
    bench_equity = initial * (1 + bench).cumprod()

    equity = pd.DataFrame({
        "strategy": strat_equity,
        "benchmark": bench_equity,
    })

    return equity


# ════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════

def validate_backtest(strategy_metrics, benchmark_metrics, window_results, stress_results):
    """Summary validation of the full backtest."""
    results = {}

    results["strategy"] = strategy_metrics
    results["benchmark"] = benchmark_metrics

    # Walk-forward summary
    if not window_results.empty:
        results["windows_total"] = len(window_results)
        results["windows_outperformed"] = int(window_results["outperformed"].sum())
        results["win_pct_vs_benchmark"] = round(
            window_results["outperformed"].mean() * 100, 1
        )

    # Stress test summary
    if not stress_results.empty:
        results["stress_tests"] = stress_results.to_dict("records")

    return results


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def run():
    """
    Main entry point for backtesting.

    Loads pre-computed portfolio returns and runs:
      1. Full-period performance metrics
      2. Walk-forward window analysis
      3. Stress tests
      4. Equity curve comparison vs SPY

    Returns: (strategy_metrics, benchmark_metrics, window_results,
              stress_results, equity_curves, validation)
    """
    # Load strategy returns
    portfolio_path = PROCESSED_DIR / "portfolio_results.parquet"
    if not portfolio_path.exists():
        raise FileNotFoundError(
            "No portfolio results found. Run risk_manager.py first:\n"
            "  python main.py --stage portfolio"
        )

    portfolio_df = pd.read_parquet(portfolio_path)
    strategy_returns = portfolio_df["daily_return"]
    logger.info(f"Loaded strategy returns: {len(strategy_returns)} days")

    # Filter to backtest period
    backtest_start = pd.Timestamp(BACKTEST_START)
    strategy_returns = strategy_returns[strategy_returns.index >= backtest_start]
    logger.info(f"Backtest period: {strategy_returns.index.min().date()} to "
                f"{strategy_returns.index.max().date()} ({len(strategy_returns)} days)")

    # Load benchmark
    benchmark_returns = load_benchmark_returns()
    benchmark_returns = benchmark_returns[benchmark_returns.index >= backtest_start]

    # Full-period metrics
    logger.info("\n=== Full Period Performance ===")
    strategy_metrics = compute_metrics(strategy_returns, "MOSAIC Strategy")
    benchmark_metrics = compute_metrics(benchmark_returns, "SPY Benchmark")

    for key in ["total_return_pct", "annualized_return_pct", "sharpe", "sortino",
                "max_drawdown_pct", "win_rate_pct"]:
        s_val = strategy_metrics.get(key, "N/A")
        b_val = benchmark_metrics.get(key, "N/A")
        logger.info(f"  {key}: Strategy={s_val} vs SPY={b_val}")

    # Walk-forward analysis
    logger.info("\n=== Walk-Forward Analysis ===")
    window_results = run_walk_forward(strategy_returns, benchmark_returns)

    # Stress tests
    logger.info("\n=== Stress Tests ===")
    stress_results = run_stress_tests(strategy_returns, benchmark_returns)

    # Equity curves
    equity_curves = build_equity_curves(strategy_returns, benchmark_returns)

    # Validation
    validation = validate_backtest(
        strategy_metrics, benchmark_metrics, window_results, stress_results
    )

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    window_results.to_parquet(PROCESSED_DIR / "backtest_windows.parquet")
    stress_results.to_parquet(PROCESSED_DIR / "backtest_stress.parquet")
    equity_curves.to_parquet(PROCESSED_DIR / "equity_curves.parquet")

    # Save metrics as well
    metrics_df = pd.DataFrame([strategy_metrics, benchmark_metrics])
    metrics_df.to_parquet(PROCESSED_DIR / "backtest_metrics.parquet")

    logger.info(f"\nSaved backtest results to {PROCESSED_DIR}")

    return strategy_metrics, benchmark_metrics, window_results, stress_results, equity_curves, validation


if __name__ == "__main__":
    strat, bench, windows, stress, equity, validation = run()

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)

    print("\n=== MOSAIC Strategy ===")
    for k, v in strat.items():
        print(f"  {k}: {v}")

    print("\n=== SPY Benchmark ===")
    for k, v in bench.items():
        print(f"  {k}: {v}")

    print("\n=== Walk-Forward Windows ===")
    if not windows.empty:
        print(windows[["window", "test_start", "test_end", "strat_return_pct",
                        "bench_return_pct", "outperformed"]].to_string(index=False))
        outperf_pct = windows["outperformed"].mean() * 100
        print(f"\n  Outperformed SPY in {outperf_pct:.0f}% of windows")

    print("\n=== Stress Tests ===")
    if not stress.empty:
        print(stress.to_string(index=False))

    print("\n=== Equity Curve (start and end) ===")
    print(f"  Strategy: ${equity['strategy'].iloc[0]:,.0f} -> ${equity['strategy'].iloc[-1]:,.0f}")
    print(f"  SPY:      ${equity['benchmark'].iloc[0]:,.0f} -> ${equity['benchmark'].iloc[-1]:,.0f}")
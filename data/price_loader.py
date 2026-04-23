"""
MOSAIC — Price Data Loader
Downloads OHLCV + VIX + SPY from Yahoo Finance.
Computes daily returns. Caches everything to Parquet.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root to Python's import path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TICKERS, BENCHMARK, VIX_TICKER, START_DATE, END_DATE, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_price_data(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """
    Download adjusted OHLCV for all tickers in one API call.
    
    Returns a DataFrame with MultiIndex columns:
      Level 0 = Field: Open, High, Low, Close, Volume
      Level 1 = Ticker: AAPL, MSFT, ...
    
    Access examples:
      prices["Close"]           → all tickers' closes (DataFrame)
      prices["Close"]["AAPL"]   → Apple closes only (Series)
      prices["Volume"]["TSLA"]  → Tesla volume (Series)
    """
    logger.info(f"Fetching OHLCV for {len(tickers)} tickers: {start} → {end}")

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # Adjust for splits & dividends — CRITICAL
        threads=True,        # Parallel download — faster for 20 tickers
        progress=True,
    )

    if df.empty:
        raise ValueError("yfinance returned no data — check tickers and date range")

    logger.info(f"Price data shape: {df.shape}")
    return df


def fetch_vix(start=START_DATE, end=END_DATE):
    """
    Download VIX (fear index) separately.
    
    VIX measures expected S&P 500 volatility over the next 30 days,
    derived from options prices. Our regime model (Layer 3) uses this
    to classify whether markets are calm, transitional, or panicking.
    """
    logger.info("Fetching VIX data")
    vix = yf.download(VIX_TICKER, start=start, end=end, auto_adjust=True, progress=False)

    if vix.empty:
        raise ValueError("No VIX data returned")

    # yfinance quirk: single-ticker downloads sometimes return MultiIndex columns
    # Flatten to simple column names
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # We only need the closing value of VIX each day
    vix = vix[["Close"]].rename(columns={"Close": "VIX_Close"})

    logger.info(f"VIX: {len(vix)} days, range {vix['VIX_Close'].min():.1f} – {vix['VIX_Close'].max():.1f}")
    return vix


def fetch_benchmark(start=START_DATE, end=END_DATE):
    """
    Download SPY — the S&P 500 ETF we benchmark against.
    
    In quant finance, raw returns don't matter. What matters is 
    RISK-ADJUSTED returns vs a benchmark. If your system returns 15% 
    but SPY returned 20%, you underperformed. If your system returned 
    12% with half the volatility of SPY's 15%, you outperformed on a 
    risk-adjusted basis (higher Sharpe ratio).
    """
    logger.info(f"Fetching benchmark ({BENCHMARK})")
    bench = yf.download(BENCHMARK, start=start, end=end, auto_adjust=True, progress=False)

    if bench.empty:
        raise ValueError("No benchmark data returned")

    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)

    bench = bench[["Close"]].rename(columns={"Close": f"{BENCHMARK}_Close"})
    logger.info(f"Benchmark: {len(bench)} trading days")
    return bench


def compute_returns(prices):
    """
    Compute daily simple returns: r_t = (P_t / P_{t-1}) - 1
    
    Why returns instead of raw prices?
    1. Scale-invariant — 2% is 2% whether stock is $10 or $1000
    2. Roughly stationary — prices trend up forever, returns bounce around zero
       Most statistical models (regression, correlation) assume stationarity
    3. Portfolio math works — portfolio return = weighted sum of stock returns
    4. Comparable across stocks — you can't compare AAPL at $180 vs JPM at $200
       but you can compare their 2% vs 1.5% daily returns
    """
    # Extract Close prices from the MultiIndex DataFrame
    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices

    returns = close.pct_change()   # (P_today / P_yesterday) - 1
    returns = returns.iloc[1:]     # First row is NaN (no previous day), drop it

    return returns


def validate_data(prices, vix):
    """
    Quality checks. In quant finance, bad data = bad backtest = false confidence.
    
    A single missing day can shift momentum calculations.
    An unadjusted split makes signals see phantom crashes.
    This function catches the obvious problems early.
    """
    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices

    results = {}

    # Did we get all 20 tickers?
    results["tickers_loaded"] = list(close.columns)
    results["tickers_missing"] = [t for t in TICKERS if t not in close.columns]

    # Date coverage
    results["date_range"] = (str(close.index.min().date()), str(close.index.max().date()))
    results["trading_days"] = len(close)

    # Missing data — over 5% for any ticker is a red flag
    missing_pct = (close.isnull().sum() / len(close) * 100).round(2)
    results["missing_pct_per_ticker"] = missing_pct.to_dict()
    results["total_missing_pct"] = round(close.isnull().sum().sum() / close.size * 100, 2)

    # VIX coverage
    results["vix_days"] = len(vix)
    results["vix_missing"] = int(vix.isnull().sum().sum())

    return results


def save_data(prices, vix, benchmark):
    """Save to Parquet. First run = slow download. Every run after = instant load."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(RAW_DIR / "prices.parquet")
    vix.to_parquet(RAW_DIR / "vix.parquet")
    benchmark.to_parquet(RAW_DIR / "benchmark.parquet")
    logger.info(f"All data saved to {RAW_DIR}")


def load_cached():
    """Try loading from Parquet cache. Returns None if cache doesn't exist."""
    paths = [RAW_DIR / f for f in ["prices.parquet", "vix.parquet", "benchmark.parquet"]]
    if all(p.exists() for p in paths):
        logger.info("Loading cached data from Parquet")
        return tuple(pd.read_parquet(p) for p in paths)
    return None


def run(use_cache=True):
    """
    Main entry point. Other modules call this.
    
    First call: downloads from Yahoo Finance (~30 sec), saves to Parquet.
    Subsequent calls: loads from Parquet (~0.5 sec).
    
    Returns: (prices, vix, benchmark, returns, validation_dict)
    """
    # Try cache first
    if use_cache:
        cached = load_cached()
        if cached:
            prices, vix, benchmark = cached
            returns = compute_returns(prices)
            validation = validate_data(prices, vix)
            return prices, vix, benchmark, returns, validation

    # Fetch fresh from Yahoo Finance
    prices = fetch_price_data()
    vix = fetch_vix()
    benchmark = fetch_benchmark()
    returns = compute_returns(prices)
    validation = validate_data(prices, vix)

    # Log summary
    logger.info(f"Date range: {validation['date_range']}")
    logger.info(f"Trading days: {validation['trading_days']}")
    logger.info(f"Missing data: {validation['total_missing_pct']:.2f}%")
    if validation["tickers_missing"]:
        logger.warning(f"Missing tickers: {validation['tickers_missing']}")

    # Cache for next time
    save_data(prices, vix, benchmark)

    return prices, vix, benchmark, returns, validation


# Run directly: python data/price_loader.py
if __name__ == "__main__":
    prices, vix, benchmark, returns, validation = run(use_cache=False)

    print("\n=== Validation Summary ===")
    for k, v in validation.items():
        if k != "missing_pct_per_ticker":
            print(f"  {k}: {v}")

    print("\n=== Missing Data Per Ticker ===")
    for ticker, pct in validation["missing_pct_per_ticker"].items():
        flag = " ⚠️" if pct > 5 else ""
        print(f"  {ticker}: {pct}%{flag}")

    print(f"\n=== Sample Returns (last 5 days) ===")
    print(returns.tail())
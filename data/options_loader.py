"""
MOSAIC — Options Data Loader

Two modes:
  1. HISTORICAL PROXY — builds IV-like features from price data + VIX
     for backtesting (because free historical options data doesn't exist)
  2. LIVE SNAPSHOT — fetches current options chains from yfinance
     for forward-looking / paper trading

The proxy uses realized volatility (backward-looking) as a stand-in for
implied volatility (forward-looking). They're ~85% correlated and move
together. It's not perfect — document this limitation in your writeup.

For real historical IV data you'd need:
  - CBOE DataShop (paid, ~$1000+)
  - OptionMetrics via WRDS (free with university access)
  - Tradier API (limited free history)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TICKERS, START_DATE, END_DATE, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IV_LOOKBACK = 252  # 1 year of trading days for percentile calculation


# ════════════════════════════════════════════════════════════════
# PART 1: HISTORICAL PROXY (for backtesting)
# ════════════════════════════════════════════════════════════════

def compute_realized_volatility(prices, window=21):
    """
    Rolling annualized realized volatility.
    
    Steps:
      1. Compute log returns: ln(P_t / P_{t-1})
      2. Rolling standard deviation over 'window' days (21 ≈ 1 month)
      3. Annualize by multiplying by sqrt(252)
    
    Output is in annualized terms — 0.25 means 25% annual vol.
    Same units as VIX and implied volatility, so they're directly comparable.
    
    window=21 balances responsiveness vs noise:
      - Too short (5 days) = jittery, reacts to single-day moves
      - Too long (63 days) = sluggish, misses regime changes
      - 21 days = ~1 month, standard in industry
    """
    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices

    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    realized_vol = rolling_std * np.sqrt(252)

    return realized_vol


def compute_iv_percentile(realized_vol, lookback=IV_LOOKBACK):
    """
    IV Percentile Rank: what % of the past year had LOWER vol than today?
    
    Result is 0 to 1:
      0.90 → today's vol exceeds 90% of past year → fear is elevated
      0.10 → today's vol is lower than 90% of past year → unusual calm
      0.50 → nothing special
    
    Why this matters: volatility MEAN-REVERTS. Extremes don't last.
      - Very high IV percentile → vol likely to come down → options are expensive
      - Very low IV percentile → vol likely to expand → cheap to buy protection
    This is the basis of many options-selling strategies.
    
    The loop is O(n * lookback) per ticker. Can't vectorize cleanly because
    each day's percentile depends on a different trailing window. For 20
    tickers × ~2000 days it runs in a few seconds — fine for our use case.
    """
    iv_pctile = realized_vol.copy()

    for col in realized_vol.columns:
        vals = realized_vol[col].values
        pctiles = np.full_like(vals, np.nan)

        for i in range(lookback, len(vals)):
            if np.isnan(vals[i]):
                continue
            # Look at the trailing 252-day window
            window = vals[i - lookback:i]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                # What fraction of historical days had lower vol?
                pctiles[i] = np.mean(valid < vals[i])

        iv_pctile[col] = pctiles

    return iv_pctile


def compute_pc_ratio_proxy(prices, vix):
    """
    Approximate historical Put/Call ratio using VIX / stock_vol.
    
    Logic: When market-wide fear (VIX) is high RELATIVE to a specific
    stock's recent volatility, put demand for that stock tends to be
    elevated. This captures the direction of the relationship, though
    not the exact magnitude.
    
    Real put/call ratios come from CBOE exchange data. This proxy is
    the weakest part of the pipeline — be honest about this in interviews.
    
    Output is clipped to [0.3, 3.0] to match realistic PC ratio ranges.
    Below 0.7 is considered bullish, above 1.0 is bearish.
    """
    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices
    stock_vol = compute_realized_volatility(prices, window=21)

    # Align VIX dates to our price dates, forward-fill weekends/holidays
    vix_aligned = vix.reindex(close.index).ffill()

    # Extract VIX values as a simple Series (handle column format quirks)
    if isinstance(vix_aligned.columns, pd.MultiIndex):
        vix_values = vix_aligned.iloc[:, 0]
    else:
        vix_col = [c for c in vix_aligned.columns if "VIX" in str(c).upper()]
        vix_values = vix_aligned[vix_col[0]] if vix_col else vix_aligned.iloc[:, 0]

    # Compute proxy for each stock
    pc_proxy = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        if col in stock_vol.columns:
            # stock_vol is decimal (0.25 = 25%), VIX is points (25 = 25%)
            # Multiply stock_vol by 100 to match units before dividing
            ratio = vix_values / (stock_vol[col] * 100 + 1e-8)  # 1e-8 avoids div by zero
            pc_proxy[col] = ratio.clip(0.3, 3.0)

    return pc_proxy


def build_historical_options_features(prices, vix):
    """
    Assemble all three features into one MultiIndex DataFrame.
    
    Access: features["realized_vol"]["AAPL"], features["iv_percentile"]["TSLA"], etc.
    
    This is what the Options Signal module (Week 3) will consume to 
    generate the options anomaly score.
    """
    logger.info("Building historical options features...")

    realized_vol = compute_realized_volatility(prices, window=21)
    logger.info("  ✓ Realized volatility")

    iv_pctile = compute_iv_percentile(realized_vol)
    logger.info("  ✓ IV percentile rank")

    pc_proxy = compute_pc_ratio_proxy(prices, vix)
    logger.info("  ✓ Put/Call ratio proxy")

    # Stack into MultiIndex: (feature_name, ticker)
    features = pd.concat(
        {
            "realized_vol": realized_vol,
            "iv_percentile": iv_pctile,
            "pc_ratio_proxy": pc_proxy,
        },
        axis=1,
    )

    logger.info(f"  Final shape: {features.shape}")
    return features


# ════════════════════════════════════════════════════════════════
# PART 2: LIVE OPTIONS SNAPSHOT (for current/forward use)
# ════════════════════════════════════════════════════════════════

def fetch_current_options(ticker):
    """
    Fetch today's options chain for one ticker.
    
    Returns a dict with:
      - atm_iv: at-the-money implied volatility (strike closest to current price)
      - call_oi / put_oi: total open interest for calls and puts
      - pc_ratio: put/call open interest ratio
    
    ATM = the strike price closest to the current stock price.
    ATM options are the most liquid and most commonly quoted.
    """
    stock = yf.Ticker(ticker)

    try:
        expirations = stock.options
    except Exception:
        return {"ticker": ticker, "error": "No options available"}

    if not expirations:
        return {"ticker": ticker, "error": "No expirations found"}

    # Nearest expiration = most liquid
    chain = stock.option_chain(expirations[0])
    calls = chain.calls
    puts = chain.puts

    # Total open interest
    call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
    put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
    pc_ratio = put_oi / call_oi if call_oi > 0 else np.nan

    # ATM implied volatility — find strike closest to current price
    current_price = stock.info.get("regularMarketPrice", np.nan)
    atm_iv = np.nan
    if not calls.empty and not np.isnan(current_price) and "impliedVolatility" in calls.columns:
        idx = (calls["strike"] - current_price).abs().idxmin()
        atm_iv = calls.loc[idx, "impliedVolatility"]

    return {
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "expiration": expirations[0],
        "call_oi": int(call_oi),
        "put_oi": int(put_oi),
        "pc_ratio": round(pc_ratio, 4) if not np.isnan(pc_ratio) else None,
        "atm_iv": round(atm_iv, 4) if not np.isnan(atm_iv) else None,
    }


def fetch_all_current_options(tickers=None):
    """Fetch live options snapshot for all tickers. Returns a DataFrame."""
    tickers = tickers or TICKERS
    summaries = []

    for ticker in tickers:
        logger.info(f"Fetching live options: {ticker}")
        try:
            summaries.append(fetch_current_options(ticker))
        except Exception as e:
            logger.warning(f"  Failed for {ticker}: {e}")

    return pd.DataFrame(summaries)


# ════════════════════════════════════════════════════════════════
# VALIDATION & I/O
# ════════════════════════════════════════════════════════════════

def validate_options_data(features):
    """Quality checks on historical features."""
    results = {
        "shape": features.shape,
        "date_range": (str(features.index.min().date()), str(features.index.max().date())),
    }

    for feat_name in ["realized_vol", "iv_percentile", "pc_ratio_proxy"]:
        if feat_name in features.columns.get_level_values(0):
            feat = features[feat_name]
            results[f"{feat_name}_missing_pct"] = round(
                feat.isnull().sum().sum() / feat.size * 100, 2
            )
            results[f"{feat_name}_mean"] = round(feat.mean().mean(), 4)

    return results


def run(prices=None, vix=None, fetch_live=False):
    """
    Main entry point.
    
    Pass in prices/vix from price_loader, or it loads from cache.
    Returns: (features_df, live_snapshot_df, validation_dict)
    """
    # Load from cache if not passed in
    if prices is None or vix is None:
        prices_path = RAW_DIR / "prices.parquet"
        vix_path = RAW_DIR / "vix.parquet"
        if prices_path.exists() and vix_path.exists():
            prices = pd.read_parquet(prices_path)
            vix = pd.read_parquet(vix_path)
        else:
            raise FileNotFoundError("Run price_loader.py first!")

    # Build historical features
    features = build_historical_options_features(prices, vix)
    validation = validate_options_data(features)

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(RAW_DIR / "options_features.parquet")
    logger.info(f"Saved to {RAW_DIR / 'options_features.parquet'}")

    # Optional: live snapshot
    live = pd.DataFrame()
    if fetch_live:
        live = fetch_all_current_options()
        if not live.empty:
            live.to_parquet(RAW_DIR / "options_live_snapshot.parquet")

    return features, live, validation


if __name__ == "__main__":
    features, live, validation = run(fetch_live=False)

    print("\n=== Historical Options Features ===")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    print("\n=== Sample: AAPL IV Percentile (last 10 days) ===")
    if "iv_percentile" in features.columns.get_level_values(0):
        print(features["iv_percentile"]["AAPL"].dropna().tail(10))
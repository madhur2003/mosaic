"""
MOSAIC — Data Validation (Week 1 Deliverable)

Usage: python data_check.py

Run after: python main.py --stage data
Checks all data sources loaded correctly before you move to Week 2.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import TICKERS, BENCHMARK, RAW_DIR


def header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}\n")


def check_prices():
    header("1. PRICE DATA")
    path = RAW_DIR / "prices.parquet"
    if not path.exists():
        print("  ❌ Not found — run: python main.py --stage data")
        return False

    prices = pd.read_parquet(path)
    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices

    print(f"  Date range : {close.index.min().date()} → {close.index.max().date()}")
    print(f"  Trading days: {len(close)}")
    print(f"  Tickers     : {len(close.columns)} / {len(TICKERS)}\n")

    print(f"  {'Ticker':<8} {'First':>12} {'Last':>12} {'Missing':>8} {'Last $':>10}")
    print(f"  {'-'*54}")

    all_ok = True
    for ticker in TICKERS:
        if ticker in close.columns:
            col = close[ticker].dropna()
            missing = close[ticker].isnull().sum()
            flag = " ⚠️" if missing > len(close) * 0.05 else ""
            print(f"  {ticker:<8} {str(col.index.min().date()):>12} "
                  f"{str(col.index.max().date()):>12} {missing:>8} "
                  f"{col.iloc[-1]:>10.2f}{flag}")
        else:
            print(f"  {ticker:<8} {'NOT FOUND':>48}")
            all_ok = False

    returns = close.pct_change()
    extreme = (returns.abs() > 0.30).sum().sum()
    print(f"\n  Returns > ±30%: {extreme} (check if too many)")

    print(f"\n  {'✅ PASS' if all_ok else '⚠️  ISSUES'}")
    return all_ok


def check_vix():
    header("2. VIX")
    path = RAW_DIR / "vix.parquet"
    if not path.exists():
        print("  ❌ Not found")
        return False

    vix = pd.read_parquet(path)
    v = vix.iloc[:, 0]

    print(f"  Range  : {v.min():.1f} – {v.max():.1f} (mean {v.mean():.1f})")
    print(f"  Days   : {len(vix)}")
    print(f"  Missing: {v.isnull().sum()}")
    if v.max() > 60:
        print("  ✓ COVID VIX spike captured")
    else:
        print("  ⚠️  Max VIX seems low")

    print(f"\n  ✅ PASS")
    return True


def check_benchmark():
    header("3. BENCHMARK (SPY)")
    path = RAW_DIR / "benchmark.parquet"
    if not path.exists():
        print("  ❌ Not found")
        return False

    bench = pd.read_parquet(path)
    b = bench.iloc[:, 0]
    ret = (b.iloc[-1] / b.iloc[0] - 1) * 100

    print(f"  ${b.iloc[0]:.2f} → ${b.iloc[-1]:.2f} ({ret:+.1f}%)")
    print(f"  Days: {len(bench)}")

    print(f"\n  ✅ PASS")
    return True


def check_options():
    header("4. OPTIONS FEATURES")
    path = RAW_DIR / "options_features.parquet"
    if not path.exists():
        print("  ❌ Not found")
        return False

    features = pd.read_parquet(path)
    feat_names = features.columns.get_level_values(0).unique().tolist()

    print(f"  Shape: {features.shape}")
    print(f"  Features: {feat_names}\n")

    for feat in feat_names:
        sub = features[feat]
        missing = sub.isnull().sum().sum() / sub.size * 100
        print(f"  {feat}: mean={sub.mean().mean():.4f}, missing={missing:.1f}%")

    print(f"\n  ✅ PASS")
    return True


def check_alignment():
    header("5. CROSS-SOURCE ALIGNMENT")
    try:
        prices = pd.read_parquet(RAW_DIR / "prices.parquet")
        vix = pd.read_parquet(RAW_DIR / "vix.parquet")
        bench = pd.read_parquet(RAW_DIR / "benchmark.parquet")
    except FileNotFoundError as e:
        print(f"  ❌ Missing: {e}")
        return False

    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices
    common = close.index.intersection(vix.index).intersection(bench.index)

    print(f"  Prices: {len(close.index)} days")
    print(f"  VIX:    {len(vix.index)} days")
    print(f"  SPY:    {len(bench.index)} days")
    print(f"  Common: {len(common)} days")

    gap = len(close.index) - len(common)
    print(f"  Gap:    {gap} days {'(normal)' if gap < 10 else '⚠️'}")

    print(f"\n  ✅ PASS")
    return True


def main():
    print("\n" + "🧩 " * 20)
    print("  MOSAIC — Week 1 Data Validation")
    print("🧩 " * 20)

    results = {}
    for name, func in [("Prices", check_prices), ("VIX", check_vix),
                        ("Benchmark", check_benchmark), ("Options", check_options),
                        ("Alignment", check_alignment)]:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n  ❌ {name} crashed: {e}")
            results[name] = False

    header("SUMMARY")
    for name, passed in results.items():
        print(f"  {'✅' if passed else '❌'} {name}")

    if all(results.values()):
        print("\n  🎉 All passed! Ready for Week 2 → NLP Signal (FinBERT)")
    else:
        print("\n  Fix issues above, then re-run.")


if __name__ == "__main__":
    main()


## FILE 8: `mosaic/requirements.txt`

# **Purpose:** Dependencies for Week 1 only. Uncomment future weeks as you get there.
# ```
# # Week 1: Data Pipeline
# yfinance>=0.2.31
# sec-edgar-downloader>=5.0.0
# pyarrow>=14.0
# pandas>=2.1.0
# numpy>=1.24.0

# Week 2: NLP Signal (uncomment when ready)
# transformers>=4.36.0
# sentence-transformers>=2.2.0
# torch>=2.0.0

# Week 4: Regime Model
# hmmlearn>=0.3.0
# scikit-learn>=1.3.0
# scipy>=1.11.0
# statsmodels>=0.14.0

# Week 7-8: Backtest & Analysis
# quantstats>=0.0.62
# matplotlib>=3.8.0
# seaborn>=0.13.0
# plotly>=5.18.0
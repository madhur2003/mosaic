"""
MOSAIC Configuration
Every module imports from this file. Never hardcode tickers, dates, 
or parameters in individual files.
"""

from pathlib import Path

# ── Project Paths ──
# Path(__file__).parent = the folder config.py lives in (mosaic/)
# All paths are relative from here — works on any machine
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # Data exactly as downloaded — never modify
PROCESSED_DIR = DATA_DIR / "processed"  # Cleaned/transformed data

# Create folders automatically when any module imports config
for d in [RAW_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ticker Universe: 20 liquid S&P 500 stocks across sectors ──
# Why these? High daily volume, deep options markets, clean data since 2017
# Cross-sector so one sector crashing doesn't destroy the portfolio
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL",   # Tech
    "JNJ", "UNH", "PFE",                # Healthcare
    "JPM", "GS", "BAC",                 # Financials
    "AMZN", "TSLA", "WMT",              # Consumer
    "XOM", "CVX",                        # Energy
    "CAT", "BA",                         # Industrials
    "META", "DIS",                       # Communication
    "LIN",                               # Materials
]

# SPY = S&P 500 ETF. If we can't beat this, the system is pointless.
BENCHMARK = "SPY"

# VIX = CBOE Volatility Index. Measures market fear.
# Normal: 15-20. Elevated: 20-30. Panic: 30+. COVID peak: ~82.
VIX_TICKER = "^VIX"

# ── Date Range ──
# START is 2017, but backtest begins 2018. The extra year is "warm-up" —
# signals like 252-day IV percentile need a full year of history before
# they can produce their first output. Without warm-up = NaN signals.
START_DATE = "2017-01-01"
END_DATE = "2024-12-31"
BACKTEST_START = "2018-01-01"

# ── SEC EDGAR ──
# SEC requires identification. They WILL block you without this.
# Put your real name and email.
SEC_USER_AGENT = "MOSAIC Research madhur2k3@gmail.com"  # ← CHANGE THIS
SEC_FILING_TYPES = ["10-Q", "8-K"]
SEC_MAX_FILINGS = 20  # per ticker per filing type
# MOSAIC 🧩

**Multi-Signal Options-Aware Adaptive Intelligence for Capital Allocation**

A research-grade quantitative trading system built from scratch in Python. Combines NLP tone-shift analysis of SEC filings, options flow anomaly detection, and adaptive momentum signals, dynamically weighted by an online-learning meta-allocator and filtered through a Hidden Markov Model regime classifier.

**[View Interactive Dashboard](https://madhur2003.github.io/mosaic/dashboard.html)**

---

## Architecture
---

## Results

| Metric | MOSAIC | SPY |
|--------|--------|-----|
| Total Return | -8.6% | +146.9% |
| Sharpe Ratio | -1.12 | 0.76 |
| Max Drawdown | -8.6% | -33.7% |
| Win Rate | 0.7% | 55.2% |
| Volatility | 1.1% | 19.5% |

**What worked:** Risk management limited max drawdown to -8.6% vs SPY's -33.7%. The regime model correctly identified crisis periods (COVID, 2022 bear). The circuit breaker prevented catastrophic losses.

**Key limitations:** Realized volatility proxy instead of real historical IV data, NLP coverage limited to 6/20 tickers due to SEC filing parser constraints, aggressive circuit breaker threshold (-15%) kept the portfolio in cash ~75% of the backtest period.

---

## Project Structure
---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/madhur2003/mosaic.git
cd mosaic
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Edit config.py: update SEC_USER_AGENT with your email

# Run the full pipeline
python main.py --stage all --no-sec

# Or run individual stages
python main.py --stage data --no-sec    # Data pipeline
python main.py --stage signals          # Signal generation + regime
python main.py --stage allocator        # Meta-allocator
python main.py --stage portfolio        # Portfolio + risk management
python main.py --stage backtest         # Walk-forward backtest
python main.py --stage analysis         # Tearsheet + research memo

# Launch the interactive dashboard
python export_dashboard_data.py
python -m http.server 8080
# Open: http://localhost:8080/dashboard.html
```

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data | yfinance, sec-edgar-downloader, pyarrow |
| NLP | transformers (FinBERT), sentence-transformers, torch |
| ML | hmmlearn, scikit-learn, scipy |
| Core | pandas, numpy |
| Dashboard | Chart.js (vanilla HTML/JS) |

---

## Key Concepts

**Signals:**
- **NLP Tone-Shift** -- FinBERT sentence embeddings on consecutive 10-Q MD&A sections. Cosine similarity measures language drift. Higher drift = more uncertainty = bearish signal.
- **Options Anomaly** -- IV percentile rank (contrarian: high IV = bullish) combined with Put/Call ratio z-score deviations. Uses realized vol proxy for backtesting.
- **Momentum** -- Classic 12-1 momentum blended with mean-reversion z-score, adapted by regime state.

**Regime Model:**
- Gaussian HMM with 3 states trained on VIX level, VIX term structure, and market breadth.
- Expanding-window training to prevent lookahead bias.
- Risk-On (full exposure), Transitional (half), Risk-Off (flat).

**Meta-Allocator:**
- Exponential Gradient Descent dynamically reweights signals based on rolling 60-day Sharpe contribution.
- 10% minimum weight floor prevents any signal from being completely ignored.
- Monthly rebalancing with learning rate η = 0.1.

**Risk Management:**
- Fractional Kelly (25%) position sizing
- 5% max per position, 25% max per sector
- Drawdown circuit breaker: halt at -10%, liquidate at -15%
- Transaction cost modeling: 5bps slippage + $0.005/share commission

---

## Improvements

With more time and resources:

1. **Real IV Data** -- CBOE DataShop or OptionMetrics via WRDS for actual historical implied volatility
2. **Better NLP** -- SEC-API.io for pre-parsed filings, expand to 8-K for more timely signals
3. **Additional Signals** -- Earnings surprise momentum, insider trading, short interest, cross-asset signals
4. **Transformer Regime** -- Replace 3-state HMM with a transformer-based regime detector
5. **Execution** -- TWAP/VWAP simulation, realistic market impact, paper trading validation

---

## License

MIT

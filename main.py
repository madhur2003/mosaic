"""
MOSAIC -- Main Entry Point

Usage:
    python main.py --stage data           # Data pipeline
    python main.py --stage data --no-sec  # Skip SEC download
    python main.py --stage signals        # All signals + regime
    python main.py --stage allocator      # Meta-allocator
    python main.py --stage portfolio      # Portfolio & risk management
    python main.py --stage backtest       # Walk-forward backtest
    python main.py --stage analysis       # Tearsheet & research memo
    python main.py --stage all            # Run everything
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mosaic")


def run_data_pipeline(skip_sec=False):
    from data.price_loader import run as load_prices
    from data.options_loader import run as load_options

    print("\n" + "=" * 60)
    print("  MOSAIC -- Data Pipeline (Week 1)")
    print("=" * 60)

    print("\n📈 Step 1/3: Price data...")
    prices, vix, benchmark, returns, price_val = load_prices(use_cache=False)
    print(f"   ✓ {price_val['trading_days']} days, {len(price_val['tickers_loaded'])} tickers")

    print("\n📊 Step 2/3: Options features...")
    options, live, opt_val = load_options(prices=prices, vix=vix, fetch_live=False)
    print(f"   ✓ Shape: {opt_val['shape']}")

    if skip_sec:
        print("\n📄 Step 3/3: SEC filings -- SKIPPED")
    else:
        print("\n📄 Step 3/3: SEC filings...")
        try:
            from data.sec_loader import run as load_sec
            for ft in ["10-Q", "8-K"]:
                df, sec_val = load_sec(download=True, filing_type=ft)
                print(f"   ✓ {ft}: {sec_val['total_filings']} filings")
        except Exception as e:
            print(f"   ⚠ {e}")

    print("\n  ✅ Data pipeline complete!\n")


def run_signals():
    print("\n" + "=" * 60)
    print("  MOSAIC -- Signals (Weeks 2-4)")
    print("=" * 60)

    print("\n🧠 Signal A: NLP...")
    try:
        from signals.nlp_signal import run as f
        _, _, _, v = f()
        print(f"   ✓ {v.get('tickers', 0)} tickers, {v.get('total_records', 0)} signals")
    except Exception as e:
        print(f"   ⚠ {e}")

    print("\n📊 Signal B: Options...")
    try:
        from signals.options_signal import run as f
        _, _, _, _, v = f()
        print(f"   ✓ {v.get('shape', '?')}")
    except Exception as e:
        print(f"   ❌ {e}")

    print("\n🔄 Regime Model...")
    regime_labels = None
    try:
        from regime.hmm_regime import run as f
        labels, _, _, v = f(use_expanding=True)
        regime_labels = labels
        for s, n in {0: "Risk-On", 1: "Transitional", 2: "Risk-Off"}.items():
            if n in v:
                print(f"   ✓ {n}: {v[n]['pct']}%")
    except Exception as e:
        print(f"   ❌ {e}")

    print("\n📈 Signal C: Momentum...")
    try:
        from signals.momentum_signal import run as f
        _, _, _, v = f(regime_labels=regime_labels)
        print(f"   ✓ {v.get('shape', '?')}")
    except Exception as e:
        print(f"   ❌ {e}")

    print("\n  ✅ Signals complete!\n")


def run_allocator():
    print("\n" + "=" * 60)
    print("  MOSAIC -- Meta-Allocator (Week 5)")
    print("=" * 60)
    try:
        from allocation.meta_allocator import run as f
        _, w, _, v = f()
        print(f"\n   ✓ Weights: {v.get('final_weights', '?')}")
    except Exception as e:
        print(f"   ❌ {e}")
    print("\n  ✅ Allocator complete!\n")


def run_portfolio():
    print("\n" + "=" * 60)
    print("  MOSAIC -- Portfolio & Risk (Week 6)")
    print("=" * 60)
    try:
        from portfolio.risk_manager import run as f
        _, pv, _, _, v = f()
        print(f"\n   ✓ ${pv.iloc[-1]:,.0f} | Sharpe: {v.get('sharpe_ratio', '?')} | Max DD: {v.get('max_drawdown_pct', '?')}%")
    except Exception as e:
        print(f"   ❌ {e}")
    print("\n  ✅ Portfolio complete!\n")


def run_backtest():
    print("\n" + "=" * 60)
    print("  MOSAIC -- Backtest (Week 7)")
    print("=" * 60)
    try:
        from backtest.walk_forward import run as f
        strat, bench, windows, _, _, _ = f()
        print(f"\n   Strategy: {strat.get('total_return_pct')}% | Sharpe {strat.get('sharpe')}")
        print(f"   SPY:      {bench.get('total_return_pct')}% | Sharpe {bench.get('sharpe')}")
        if not windows.empty:
            print(f"   Walk-forward: beat SPY in {windows['outperformed'].mean()*100:.0f}% of windows")
    except Exception as e:
        print(f"   ❌ {e}")
    print("\n  ✅ Backtest complete!\n")


def run_analysis():
    print("\n" + "=" * 60)
    print("  MOSAIC -- Analysis & Tearsheet (Week 8)")
    print("=" * 60)
    try:
        from analysis.tearsheet import run as f
        html, memo = f()
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback
        traceback.print_exc()
    print("")


def run_regime():
    try:
        from regime.hmm_regime import run as f
        labels, _, _, v = f(use_expanding=True)
        for k, val in v.items():
            print(f"  {k}: {val}")
    except Exception as e:
        print(f"   ❌ {e}")


STAGES = {
    "data": run_data_pipeline,
    "signals": run_signals,
    "allocator": run_allocator,
    "portfolio": run_portfolio,
    "backtest": run_backtest,
    "analysis": run_analysis,
    "regime": run_regime,
}


def main():
    parser = argparse.ArgumentParser(description="MOSAIC Quant System")
    parser.add_argument("--stage", choices=list(STAGES.keys()) + ["all"], default="data")
    parser.add_argument("--no-sec", action="store_true", help="Skip SEC download")
    args = parser.parse_args()

    if args.stage == "all":
        for name in ["data", "signals", "allocator", "portfolio", "backtest", "analysis"]:
            func = STAGES[name]
            if name == "data":
                func(skip_sec=args.no_sec)
            else:
                func()
    elif args.stage == "data":
        run_data_pipeline(skip_sec=args.no_sec)
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
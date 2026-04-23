"""
MOSAIC -- Export Dashboard Data
Reads all Parquet results and exports to a single JSON file
that the HTML dashboard can consume.

Usage: python export_dashboard_data.py
Output: dashboard_data.json (in project root)
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR, TICKERS

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "JPM": "Financials", "GS": "Financials", "BAC": "Financials",
    "AMZN": "Consumer", "TSLA": "Consumer", "WMT": "Consumer",
    "XOM": "Energy", "CVX": "Energy",
    "CAT": "Industrials", "BA": "Industrials",
    "META": "Comm", "DIS": "Comm",
    "LIN": "Materials",
}


def safe_load(path):
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def export():
    print("Exporting MOSAIC data for dashboard...\n")
    output = {"tickers": TICKERS, "sectors": SECTOR_MAP}

    # 1. Equity Curves
    equity = safe_load(PROCESSED_DIR / "equity_curves.parquet")
    if not equity.empty:
        eq = equity.iloc[::3]
        output["equity"] = [
            {"date": str(idx.date()) if hasattr(idx, "date") else str(idx),
             "strategy": round(row.get("strategy", 0)),
             "benchmark": round(row.get("benchmark", 0))}
            for idx, row in eq.iterrows()
        ]
        print(f"  ✓ Equity curves: {len(output['equity'])} points")
    else:
        output["equity"] = []
        print("  ⚠ No equity curves found")

    # 2. Portfolio Results
    portfolio = safe_load(PROCESSED_DIR / "portfolio_results.parquet")
    if not portfolio.empty:
        pr = portfolio.iloc[::3]
        output["portfolio"] = [
            {"date": str(idx.date()) if hasattr(idx, "date") else str(idx),
             "daily_return": round(row.get("daily_return", 0), 5),
             "drawdown": round(row.get("drawdown", 0), 4)}
            for idx, row in pr.iterrows()
        ]
        print(f"  ✓ Portfolio results: {len(output['portfolio'])} points")

    # 3. Backtest Metrics
    metrics = safe_load(PROCESSED_DIR / "backtest_metrics.parquet")
    if not metrics.empty:
        output["metrics"] = {}
        for _, row in metrics.iterrows():
            name = "strategy" if "Strategy" in str(row.get("name", "")) else "benchmark"
            output["metrics"][name] = {}
            for k, v in row.to_dict().items():
                if isinstance(v, (float, np.floating)):
                    if not np.isnan(v):
                        output["metrics"][name][k] = round(v, 4)
                elif isinstance(v, (int, np.integer)):
                    output["metrics"][name][k] = int(v)
                elif isinstance(v, str):
                    output["metrics"][name][k] = v
                elif not pd.isna(v):
                    output["metrics"][name][k] = str(v)
        print(f"  ✓ Backtest metrics loaded")

    # 4. Walk-Forward Windows
    windows = safe_load(PROCESSED_DIR / "backtest_windows.parquet")
    if not windows.empty:
        output["windows"] = []
        for _, row in windows.iterrows():
            output["windows"].append({
                "id": int(row.get("window", 0)),
                "period": f"{str(row.get('test_start', ''))[:7]} to {str(row.get('test_end', ''))[:7]}",
                "strat": round(float(row.get("strat_return_pct", 0)), 2),
                "bench": round(float(row.get("bench_return_pct", 0)), 2),
                "outperformed": bool(row.get("outperformed", False)),
            })
        print(f"  ✓ Walk-forward: {len(output['windows'])} windows")

    # 5. Stress Tests
    stress = safe_load(PROCESSED_DIR / "backtest_stress.parquet")
    if not stress.empty:
        output["stress"] = []
        for _, row in stress.iterrows():
            output["stress"].append({
                "period": str(row.get("period", "")),
                "days": int(row.get("days", 0)),
                "strat": round(float(row.get("strategy_return_pct", 0)), 2),
                "bench": round(float(row.get("benchmark_return_pct", 0)), 2),
                "outperformed": bool(row.get("outperformed", False)),
            })
        print(f"  ✓ Stress tests: {len(output['stress'])} periods")

    # 6. Weight Evolution
    weights = safe_load(PROCESSED_DIR / "weight_evolution.parquet")
    if not weights.empty:
        w = weights.iloc[::21]
        output["weights"] = []
        for idx, row in w.iterrows():
            point = {"date": str(idx.date()) if hasattr(idx, "date") else str(idx)}
            for col in weights.columns:
                point[col] = round(float(row[col]), 3)
            output["weights"].append(point)
        print(f"  ✓ Weights: {len(output['weights'])} points")

    # 7. Regime Labels
    regime = safe_load(PROCESSED_DIR / "regime_labels.parquet")
    regime_features = safe_load(PROCESSED_DIR / "regime_features.parquet")
    if not regime.empty and not regime_features.empty:
        merged = regime.join(regime_features, how="inner")
        m = merged.iloc[::3]
        output["regime"] = []
        for idx, row in m.iterrows():
            output["regime"].append({
                "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                "regime": int(row.get("regime", 1)),
                "vix": round(float(row.get("vix_level", 0)), 1),
                "breadth": round(float(row.get("market_breadth", 0)), 2),
                "scalar": round(float(row.get("regime_scalar", 0.5)), 1),
            })
        counts = regime["regime"].value_counts().to_dict()
        total = len(regime)
        output["regime_dist"] = {}
        for k, v in counts.items():
            output["regime_dist"][str(int(k))] = {
                "count": int(v),
                "pct": round(v / total * 100, 1),
            }
        print(f"  ✓ Regime: {len(output['regime'])} points")

    # 8. Signal Data (per ticker)
    options_sig = safe_load(PROCESSED_DIR / "options_signal.parquet")
    momentum_sig = safe_load(PROCESSED_DIR / "momentum_signal.parquet")
    nlp_sig = safe_load(PROCESSED_DIR / "nlp_signals_daily.parquet")
    final_scores = safe_load(PROCESSED_DIR / "final_scores.parquet")

    output["signals"] = {}
    for ticker in TICKERS:
        ticker_data = []

        # Use the most complete signal as base dates
        if not options_sig.empty:
            base_dates = options_sig.index
        elif not momentum_sig.empty:
            base_dates = momentum_sig.index
        else:
            output["signals"][ticker] = []
            continue

        for i in range(0, len(base_dates), 5):
            date = base_dates[i]
            point = {"date": str(date.date()) if hasattr(date, "date") else str(date)}

            # Options signal
            try:
                if not options_sig.empty and ticker in options_sig.columns and date in options_sig.index:
                    v = float(options_sig.at[date, ticker])
                    point["options"] = round(v, 3) if not np.isnan(v) else 0
                else:
                    point["options"] = 0
            except Exception:
                point["options"] = 0

            # Momentum signal
            try:
                if not momentum_sig.empty and ticker in momentum_sig.columns and date in momentum_sig.index:
                    v = float(momentum_sig.at[date, ticker])
                    point["momentum"] = round(v, 3) if not np.isnan(v) else 0
                else:
                    point["momentum"] = 0
            except Exception:
                point["momentum"] = 0

            # NLP signal
            try:
                if not nlp_sig.empty and ticker in nlp_sig.columns and date in nlp_sig.index:
                    v = float(nlp_sig.at[date, ticker])
                    point["nlp"] = round(v, 3) if not np.isnan(v) else 0
                else:
                    point["nlp"] = 0
            except Exception:
                point["nlp"] = 0

            # Composite (final scores)
            try:
                if not final_scores.empty and ticker in final_scores.columns and date in final_scores.index:
                    v = float(final_scores.at[date, ticker])
                    point["composite"] = round(v, 3) if not np.isnan(v) else 0
                else:
                    point["composite"] = 0
            except Exception:
                point["composite"] = 0

            ticker_data.append(point)

        output["signals"][ticker] = ticker_data

    print(f"  ✓ Signals: {len(TICKERS)} tickers")

    # 9. Position Stats
    positions = safe_load(PROCESSED_DIR / "positions.parquet")
    if not positions.empty:
        active_days = int((positions.abs().sum(axis=1) > 0).sum())
        total_days = len(positions)
        non_zero = positions[positions != 0].abs()
        avg_size = float(non_zero.mean().mean()) if not non_zero.empty else 0
        output["position_stats"] = {
            "active_days": active_days,
            "total_days": total_days,
            "active_pct": round(active_days / total_days * 100, 1),
            "avg_position_size": round(avg_size * 100, 3),
        }
        print(f"  ✓ Position stats: {active_days}/{total_days} active days")

    # Write JSON
    output_path = Path(__file__).parent / "dashboard_data.json"
    with open(output_path, "w") as f:
        json.dump(output, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n  ✅ Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"\n  Next steps:")
    print(f"    cd {Path(__file__).parent}")
    print(f"    python -m http.server 8080")
    print(f"    Then open: http://localhost:8080/dashboard.html")


if __name__ == "__main__":
    export()
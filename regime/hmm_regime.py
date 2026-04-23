"""
MOSAIC -- HMM Regime Classifier (Week 4)

Uses a Gaussian Hidden Markov Model to classify the market into 3 states:
  State 0: Risk-On       -- low VIX, broad participation, trends persist
  State 1: Transitional  -- moderate VIX, mixed signals, uncertain direction
  State 2: Risk-Off      -- high VIX, narrow breadth, crisis/panic

Inputs (observable features):
  1. VIX level (normalized)
  2. VIX term structure proxy (short MA vs long MA of VIX)
  3. Market breadth (% of stocks above their 200-day moving average)

Output: regime label (0, 1, 2) for each trading day
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGIME_SCALARS = {
    0: 1.0,
    1: 0.5,
    2: 0.0,
}


def build_regime_features(vix, prices):
    close = prices["Close"] if "Close" in prices.columns.get_level_values(0) else prices

    if isinstance(vix.columns, pd.MultiIndex):
        vix_values = vix.iloc[:, 0]
    else:
        vix_col = [c for c in vix.columns if "VIX" in str(c).upper()]
        vix_values = vix[vix_col[0]] if vix_col else vix.iloc[:, 0]

    vix_aligned = vix_values.reindex(close.index).ffill()

    feat_vix = vix_aligned.copy()

    vix_short_ma = vix_aligned.rolling(21).mean()
    vix_long_ma = vix_aligned.rolling(63).mean()
    feat_term_structure = vix_short_ma - vix_long_ma

    ma_200 = close.rolling(200).mean()
    above_200 = (close > ma_200).astype(float)
    feat_breadth = above_200.mean(axis=1)

    features = pd.DataFrame({
        "vix_level": feat_vix,
        "vix_term_structure": feat_term_structure,
        "market_breadth": feat_breadth,
    }, index=close.index)

    features = features.dropna()

    logger.info(f"Regime features shape: {features.shape}")
    logger.info(f"  Date range: {features.index.min().date()} -- {features.index.max().date()}")

    return features


def train_hmm(features, n_states=3, n_iter=100, random_state=42):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )

    model.fit(features_scaled)

    logger.info(f"  HMM converged: {model.monitor_.converged}")
    logger.info(f"  Log-likelihood: {model.score(features_scaled):.1f}")

    return model, scaler


def predict_regimes(model, scaler, features):
    features_scaled = scaler.transform(features.values)
    labels = model.predict(features_scaled)
    return pd.Series(labels, index=features.index, name="regime")


def label_states(regime_labels, features):
    state_vix = {}
    for state in regime_labels.unique():
        mask = regime_labels == state
        state_vix[state] = features.loc[mask, "vix_level"].mean()

    sorted_states = sorted(state_vix.keys(), key=lambda s: state_vix[s])

    remap = {
        sorted_states[0]: 0,
        sorted_states[1]: 1,
        sorted_states[2]: 2,
    }

    relabeled = regime_labels.map(remap)

    state_names = {0: "Risk-On", 1: "Transitional", 2: "Risk-Off"}
    for orig, new in remap.items():
        logger.info(f"  HMM state {orig} -> {state_names[new]} "
                     f"(avg VIX: {state_vix[orig]:.1f})")

    return relabeled, remap


def train_expanding_window(features, n_states=3, min_train_days=252, retrain_freq=21):
    logger.info(f"Training HMM with expanding window "
                f"(min_train={min_train_days}, retrain_freq={retrain_freq})...")

    all_labels = pd.Series(index=features.index, dtype=float)

    i = min_train_days
    while i < len(features):
        train_data = features.iloc[:i]

        try:
            model, scaler = train_hmm(train_data, n_states=n_states)
        except Exception as e:
            logger.warning(f"  HMM training failed at day {i}: {e}")
            i += retrain_freq
            continue

        predict_end = min(i + retrain_freq, len(features))
        predict_data = features.iloc[i:predict_end]

        if len(predict_data) > 0:
            raw_labels = predict_regimes(model, scaler, predict_data)

            train_labels = predict_regimes(model, scaler, train_data)
            _, remap = label_states(train_labels, train_data)

            mapped_labels = raw_labels.map(remap)
            all_labels.loc[predict_data.index] = mapped_labels

        i += retrain_freq

    all_labels = all_labels.ffill().bfill().astype(int)

    for state, name in {0: "Risk-On", 1: "Transitional", 2: "Risk-Off"}.items():
        pct = (all_labels == state).mean() * 100
        logger.info(f"  {name}: {pct:.1f}% of days")

    return all_labels


def validate_regime(regime_labels, features):
    results = {}

    results["total_days"] = len(regime_labels)
    results["date_range"] = (
        str(regime_labels.index.min().date()),
        str(regime_labels.index.max().date()),
    )

    state_names = {0: "Risk-On", 1: "Transitional", 2: "Risk-Off"}
    for state, name in state_names.items():
        count = (regime_labels == state).sum()
        pct = count / len(regime_labels) * 100
        avg_vix = features.loc[regime_labels == state, "vix_level"].mean()
        avg_breadth = features.loc[regime_labels == state, "market_breadth"].mean()
        results[name] = {
            "days": int(count),
            "pct": round(pct, 1),
            "avg_vix": round(avg_vix, 1) if not np.isnan(avg_vix) else None,
            "avg_breadth": round(avg_breadth, 2) if not np.isnan(avg_breadth) else None,
        }

    changes = (regime_labels != regime_labels.shift(1)).sum()
    results["regime_changes"] = int(changes)
    results["avg_regime_duration_days"] = round(len(regime_labels) / max(changes, 1), 1)

    return results


def run(vix=None, prices=None, use_expanding=True):
    if vix is None:
        vix_path = RAW_DIR / "vix.parquet"
        if vix_path.exists():
            vix = pd.read_parquet(vix_path)
        else:
            raise FileNotFoundError("No VIX data. Run price_loader.py first.")

    if prices is None:
        prices_path = RAW_DIR / "prices.parquet"
        if prices_path.exists():
            prices = pd.read_parquet(prices_path)
        else:
            raise FileNotFoundError("No price data. Run price_loader.py first.")

    logger.info("Building regime features...")
    features = build_regime_features(vix, prices)

    if use_expanding:
        regime_labels = train_expanding_window(features)
    else:
        logger.info("Training HMM on full dataset (exploration mode)...")
        model, scaler = train_hmm(features)
        raw_labels = predict_regimes(model, scaler, features)
        regime_labels, _ = label_states(raw_labels, features)

    regime_scalars = regime_labels.map(REGIME_SCALARS)
    regime_scalars.name = "regime_scalar"

    validation = validate_regime(regime_labels, features)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    regime_df = pd.DataFrame({
        "regime": regime_labels,
        "regime_scalar": regime_scalars,
    })
    regime_df.to_parquet(PROCESSED_DIR / "regime_labels.parquet")
    features.to_parquet(PROCESSED_DIR / "regime_features.parquet")
    logger.info(f"Saved regime data to {PROCESSED_DIR}")

    return regime_labels, features, regime_scalars, validation


if __name__ == "__main__":
    labels, features, scalars, validation = run(use_expanding=True)

    print("\n=== Regime Validation ===")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    print("\n=== Regime Distribution ===")
    state_names = {0: "Risk-On", 1: "Transitional", 2: "Risk-Off"}
    for state, name in state_names.items():
        if name in validation:
            info = validation[name]
            print(f"  {name}: {info['pct']}% of days, avg VIX={info['avg_vix']}, breadth={info['avg_breadth']}")

    print("\n=== Sample: Last 20 Days ===")
    print(pd.DataFrame({"regime": labels, "scalar": scalars}).tail(20))
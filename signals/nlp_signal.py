"""
MOSAIC — NLP Tone-Shift Signal (Week 2)

Takes MD&A text from 10-Q filings and measures how much the language
changed between consecutive quarters using FinBERT embeddings.

Signal logic:
  1. Chunk MD&A text into ~400-word segments (FinBERT max = 512 tokens)
  2. Embed each chunk with FinBERT → 768-dim vector
  3. Average chunk embeddings → one vector per filing
  4. Cosine similarity between current and previous quarter
  5. tone_shift = 1 - cosine_similarity (higher = more change = more uncertainty)
  6. Z-score normalize across all tickers
  7. Clip to [-1, +1]

Also computes sentiment (positive/negative/neutral) as a secondary sub-signal.

Output: one score per ticker per quarter, mapped to the filing date
(not quarter-end date) to avoid lookahead bias.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR, TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_embedding_model():
    logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("  ✓ Embedding model loaded")
    return model


def load_sentiment_model():
    logger.info("Loading FinBERT sentiment model (ProsusAI/finbert)...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    logger.info("  ✓ FinBERT sentiment model loaded")
    return tokenizer, model


def chunk_text(text, max_words=350, overlap_words=50):
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap_words

    return chunks


def embed_text(text, model):
    if not text or len(text.split()) < 10:
        return None

    chunks = chunk_text(text)
    embeddings = model.encode(chunks, show_progress_bar=False)
    doc_embedding = np.mean(embeddings, axis=0)

    return doc_embedding


def compute_cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return None
    similarity = 1 - cosine(emb1, emb2)
    return similarity


def compute_sentiment(text, tokenizer, model, max_chunks=20):
    if not text or len(text.split()) < 10:
        return None

    chunks = chunk_text(text)[:max_chunks]

    sentiments = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.squeeze().numpy()

        positive_prob = probs[0]
        negative_prob = probs[1]

        score = float(positive_prob - negative_prob)
        sentiments.append(score)

    return np.mean(sentiments)


def build_nlp_signals(filings_df, embedding_model, sentiment_tokenizer, sentiment_model):
    records = []

    tickers_in_data = filings_df["ticker"].unique()
    logger.info(f"Building NLP signals for {len(tickers_in_data)} tickers...")

    for ticker in tickers_in_data:
        ticker_filings = filings_df[filings_df["ticker"] == ticker].sort_values("filing_date")
        logger.info(f"  {ticker}: {len(ticker_filings)} filings")

        prev_embedding = None

        for idx, row in ticker_filings.iterrows():
            mda_text = row.get("mda_text", "")

            if not mda_text or len(str(mda_text).split()) < 20:
                logger.warning(f"    {ticker} {row.get('filing_date', '?')}: MD&A too short, skipping")
                prev_embedding = None
                continue

            mda_text = str(mda_text)

            current_embedding = embed_text(mda_text, embedding_model)

            if current_embedding is None:
                prev_embedding = None
                continue

            tone_shift = None
            if prev_embedding is not None:
                similarity = compute_cosine_similarity(current_embedding, prev_embedding)
                if similarity is not None:
                    tone_shift = 1.0 - similarity

            sentiment = compute_sentiment(mda_text, sentiment_tokenizer, sentiment_model)

            records.append({
                "ticker": ticker,
                "filing_date": row["filing_date"],
                "filing_type": row.get("filing_type", "10-Q"),
                "tone_shift": tone_shift,
                "sentiment": sentiment,
                "mda_word_count": len(mda_text.split()),
            })

            prev_embedding = current_embedding

    df = pd.DataFrame(records)

    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    return df


def normalize_signals(signals_df):
    df = signals_df.copy()

    if "tone_shift" in df.columns:
        valid = df["tone_shift"].dropna()
        if len(valid) > 1:
            mean_ts = valid.mean()
            std_ts = valid.std()
            if std_ts > 0:
                df["tone_shift_z"] = (df["tone_shift"] - mean_ts) / std_ts
            else:
                df["tone_shift_z"] = 0.0
        else:
            df["tone_shift_z"] = 0.0

    if "sentiment" in df.columns:
        valid = df["sentiment"].dropna()
        if len(valid) > 1:
            mean_sent = valid.mean()
            std_sent = valid.std()
            if std_sent > 0:
                df["sentiment_z"] = (df["sentiment"] - mean_sent) / std_sent
            else:
                df["sentiment_z"] = 0.0
        else:
            df["sentiment_z"] = 0.0

    if "tone_shift_z" in df.columns and "sentiment_z" in df.columns:
        df["nlp_score"] = (
            0.6 * df["tone_shift_z"].fillna(0)
            + 0.4 * (-df["sentiment_z"].fillna(0))
        )
        df["nlp_score"] = df["nlp_score"].clip(-1, 1)

    return df


def map_signals_to_daily(signals_df, price_dates):
    """
    Map quarterly NLP signals to daily frequency.
    Forward-fills from filing_date (not quarter-end) to avoid lookahead bias.
    Handles duplicate filing dates by keeping the latest entry.
    """
    daily_signals = pd.DataFrame(index=price_dates)

    tickers = signals_df["ticker"].unique()
    for ticker in tickers:
        ticker_data = signals_df[signals_df["ticker"] == ticker].copy()

        if "nlp_score" in ticker_data.columns:
            # Drop rows with no filing date
            ticker_data = ticker_data.dropna(subset=["filing_date"])
            # Remove duplicate filing dates — keep the last one
            ticker_data = ticker_data.drop_duplicates(subset=["filing_date"], keep="last")
            # Now safe to set index
            ticker_data = ticker_data.set_index("filing_date")

            daily = ticker_data["nlp_score"].reindex(price_dates).ffill()
            daily_signals[ticker] = daily

    return daily_signals


def validate_nlp_signals(signals_df, daily_signals=None):
    results = {}

    results["total_records"] = len(signals_df)
    results["tickers"] = signals_df["ticker"].nunique()
    results["records_per_ticker"] = signals_df.groupby("ticker").size().to_dict()

    if "tone_shift" in signals_df.columns:
        valid_ts = signals_df["tone_shift"].dropna()
        results["tone_shift_count"] = len(valid_ts)
        results["tone_shift_mean"] = round(valid_ts.mean(), 4) if len(valid_ts) > 0 else None
        results["tone_shift_std"] = round(valid_ts.std(), 4) if len(valid_ts) > 0 else None
        results["tone_shift_missing"] = int(signals_df["tone_shift"].isnull().sum())

    if "sentiment" in signals_df.columns:
        valid_sent = signals_df["sentiment"].dropna()
        results["sentiment_mean"] = round(valid_sent.mean(), 4) if len(valid_sent) > 0 else None
        results["sentiment_std"] = round(valid_sent.std(), 4) if len(valid_sent) > 0 else None

    if "nlp_score" in signals_df.columns:
        valid_score = signals_df["nlp_score"].dropna()
        results["nlp_score_mean"] = round(valid_score.mean(), 4) if len(valid_score) > 0 else None
        results["nlp_score_range"] = (
            round(valid_score.min(), 4),
            round(valid_score.max(), 4)
        ) if len(valid_score) > 0 else None

    if daily_signals is not None:
        results["daily_shape"] = daily_signals.shape
        results["daily_coverage_pct"] = round(
            (1 - daily_signals.isnull().sum().sum() / daily_signals.size) * 100, 1
        )

    return results


def run(filings_df=None, price_dates=None):
    if filings_df is None:
        filings_path = RAW_DIR / "filings_10q.parquet"
        if filings_path.exists():
            filings_df = pd.read_parquet(filings_path)
            logger.info(f"Loaded {len(filings_df)} filings from cache")
        else:
            raise FileNotFoundError(
                "No filings found. Run sec_loader.py first:\n"
                "  python main.py --stage data"
            )

    if price_dates is None:
        prices_path = RAW_DIR / "prices.parquet"
        if prices_path.exists():
            prices = pd.read_parquet(prices_path)
            price_dates = prices.index
        else:
            logger.warning("No price data found -- skipping daily mapping")

    valid_filings = filings_df[filings_df["mda_length"] > 100].copy()
    logger.info(f"Filings with usable MD&A: {len(valid_filings)} / {len(filings_df)}")

    if valid_filings.empty:
        logger.warning("No filings with MD&A text found. Returning empty signals.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "No valid filings"}

    embedding_model = load_embedding_model()
    sentiment_tokenizer, sentiment_model = load_sentiment_model()

    raw_signals = build_nlp_signals(
        valid_filings, embedding_model, sentiment_tokenizer, sentiment_model
    )

    if raw_signals.empty:
        logger.warning("No signals generated")
        return raw_signals, pd.DataFrame(), pd.DataFrame(), {"error": "No signals generated"}

    normalized_signals = normalize_signals(raw_signals)

    daily_signals = pd.DataFrame()
    if price_dates is not None:
        daily_signals = map_signals_to_daily(normalized_signals, price_dates)

    validation = validate_nlp_signals(normalized_signals, daily_signals)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalized_signals.to_parquet(PROCESSED_DIR / "nlp_signals.parquet")
    if not daily_signals.empty:
        daily_signals.to_parquet(PROCESSED_DIR / "nlp_signals_daily.parquet")
    logger.info(f"Saved NLP signals to {PROCESSED_DIR}")

    return raw_signals, normalized_signals, daily_signals, validation


if __name__ == "__main__":
    raw, normalized, daily, validation = run()

    print("\n=== NLP Signal Validation ===")
    for k, v in validation.items():
        if k != "records_per_ticker":
            print(f"  {k}: {v}")

    if not normalized.empty:
        print("\n=== Sample Signals (first 10 rows) ===")
        cols = ["ticker", "filing_date", "tone_shift", "sentiment", "nlp_score"]
        available_cols = [c for c in cols if c in normalized.columns]
        print(normalized[available_cols].head(10).to_string(index=False))
"""
MOSAIC — SEC EDGAR Loader
Downloads 10-Q and 8-K filings, extracts MD&A and Risk Factors text.

10-Q = quarterly report (filed 40-45 days after quarter end)
  → Contains MD&A: executives explain business in plain English
  → Tone shifts between quarters = our NLP signal

8-K = current report (filed within 4 days of material event)
  → CEO departure, merger, earnings surprise, etc.
  → More timely but less structured

SEC requires you to identify yourself via User-Agent header.
They WILL block anonymous requests. Update SEC_USER_AGENT in config.py.

The text extraction uses regex — SEC filings have wildly inconsistent
formatting across companies. Expect ~70-80% success rate on MD&A extraction.
For production, you'd use SEC-API.io or a proper XBRL parser.
"""

import re
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TICKERS, SEC_USER_AGENT, SEC_FILING_TYPES,
    SEC_MAX_FILINGS, RAW_DIR, START_DATE, END_DATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEC_RAW_DIR = RAW_DIR / "sec_filings"


def download_filings(tickers=None, filing_types=None, max_filings=SEC_MAX_FILINGS):
    """
    Download SEC filings using sec-edgar-downloader.
    
    Files are saved to: data/raw/sec_filings/sec-edgar-filings/{TICKER}/{TYPE}/
    Each filing gets its own folder containing the HTML document.
    
    This is SLOW (~5-10 min for 20 tickers). Only run once, then work 
    from the downloaded files.
    
    Requires: pip install sec-edgar-downloader
    """
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        raise ImportError("Install: pip install sec-edgar-downloader")

    tickers = tickers or TICKERS
    filing_types = filing_types or SEC_FILING_TYPES

    # sec-edgar-downloader wants company name and email separately
    parts = SEC_USER_AGENT.split()
    company = parts[0] if parts else "Research"
    email = parts[-1] if len(parts) > 1 else "researcher@example.com"

    dl = Downloader(company, email, str(SEC_RAW_DIR))

    for ticker in tickers:
        for ftype in filing_types:
            logger.info(f"Downloading {ftype} for {ticker} (max {max_filings})")
            try:
                dl.get(ftype, ticker, limit=max_filings, after=START_DATE, before=END_DATE)
                logger.info(f"  ✓ {ticker} {ftype}")
            except Exception as e:
                logger.warning(f"  ✗ {ticker} {ftype}: {e}")


def find_filing_files(ticker, filing_type="10-Q"):
    """
    Locate downloaded filing documents on disk.
    
    sec-edgar-downloader saves files as:
      sec_filings/sec-edgar-filings/{TICKER}/{TYPE}/{ACCESSION_NUM}/document.htm
    
    Returns list of Paths to the actual documents.
    """
    base = SEC_RAW_DIR / "sec-edgar-filings" / ticker / filing_type
    if not base.exists():
        return []

    files = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue

        # Try to find the primary document
        for f in folder.iterdir():
            if f.suffix in (".htm", ".html", ".txt") and "primary" in f.name.lower():
                files.append(f)
                break
        else:
            # Fallback: grab the largest HTML/TXT file
            candidates = [f for f in folder.iterdir() if f.suffix in (".htm", ".html", ".txt")]
            if candidates:
                files.append(max(candidates, key=lambda x: x.stat().st_size))

    return files


def strip_html(text):
    """Remove HTML tags and artifacts from raw filing text."""
    text = re.sub(r"<[^>]+>", " ", text)       # HTML tags
    text = re.sub(r"&[a-zA-Z]+;", " ", text)   # &amp; &nbsp; etc.
    text = re.sub(r"&#\d+;", " ", text)         # &#160; etc.
    text = re.sub(r"\s+", " ", text).strip()    # Collapse whitespace
    return text


def extract_mda_section(text):
    """
    Extract MD&A (Item 2) from a 10-Q filing.
    
    Looks for "Item 2 — Management's Discussion and Analysis" header
    and grabs everything until the next section (Item 3).
    
    Regex-based — works ~70-80% of the time. SEC filings have inconsistent
    formatting across companies and years. Some use all-caps headers, some
    use Title Case, some have extra dashes or periods. The patterns below
    cover the most common variants.
    """
    start_patterns = [
        r"Item\s*2\.?\s*[-–—.]?\s*Management['']?s?\s*Discussion\s*(?:and|&)\s*Analysis",
        r"MANAGEMENT['']?S?\s*DISCUSSION\s*(?:AND|&)\s*ANALYSIS",
    ]
    end_patterns = [
        r"Item\s*3\.?\s*[-–—.]?\s*Quantitative\s*(?:and|&)\s*Qualitative",
        r"QUANTITATIVE\s*(?:AND|&)\s*QUALITATIVE",
        r"Item\s*3\.",
    ]

    for start_pat in start_patterns:
        match = re.search(start_pat, text, re.IGNORECASE)
        if match:
            after_start = text[match.end():]
            for end_pat in end_patterns:
                end_match = re.search(end_pat, after_start, re.IGNORECASE)
                if end_match:
                    return after_start[:end_match.start()].strip()
            # No end marker found — take first ~10000 chars
            return after_start[:10000].strip()

    return ""  # Section not found


def extract_risk_factors(text):
    """
    Extract Risk Factors (Item 1A) from a filing.
    
    Risk factors are somewhat boilerplate, but CHANGES between quarters
    are meaningful. If a company adds new risk factors or changes wording,
    that's a signal worth detecting.
    """
    start_patterns = [
        r"Item\s*1A\.?\s*[-–—.]?\s*Risk\s*Factors",
        r"RISK\s*FACTORS",
    ]
    end_patterns = [
        r"Item\s*1B\.?\s*[-–—.]?\s*Unresolved\s*Staff\s*Comments",
        r"Item\s*2\.?\s*[-–—.]?\s*(?:Properties|Management)",
        r"UNRESOLVED\s*STAFF\s*COMMENTS",
    ]

    for start_pat in start_patterns:
        match = re.search(start_pat, text, re.IGNORECASE)
        if match:
            after_start = text[match.end():]
            for end_pat in end_patterns:
                end_match = re.search(end_pat, after_start, re.IGNORECASE)
                if end_match:
                    return after_start[:end_match.start()].strip()
            return after_start[:10000].strip()

    return ""


def parse_filing_date(filepath):
    """
    Try to extract filing date by scanning the first few KB of the document.
    
    The folder structure uses accession numbers (not dates), so we need
    to parse the date from inside the filing itself. Not always reliable.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")[:5000]

        # Try YYYY-MM-DD or YYYY/MM/DD
        match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", text)
        if match:
            return match.group(1).replace("/", "-")

        # Try MM/DD/YYYY
        match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", text)
        if match:
            try:
                dt = datetime.strptime(match.group(1), "%m/%d/%Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    except Exception:
        pass

    return None


def build_filings_dataframe(tickers=None, filing_type="10-Q"):
    """
    Parse all downloaded filings into a structured DataFrame.
    
    Output columns:
      ticker, filing_type, filing_date, filepath,
      mda_text, risk_text, full_text_length, mda_length, risk_length
    
    The NLP signal module (Week 2) will take the mda_text column,
    run FinBERT on consecutive quarters, and compute cosine similarity
    to measure how much the language changed.
    """
    tickers = tickers or TICKERS
    records = []

    for ticker in tickers:
        files = find_filing_files(ticker, filing_type)
        logger.info(f"{ticker}: {len(files)} {filing_type} filings found on disk")

        for fpath in files:
            full_text = strip_html(fpath.read_text(encoding="utf-8", errors="replace"))

            if filing_type == "10-Q":
                mda = extract_mda_section(full_text)
                risk = extract_risk_factors(full_text)
            else:
                # 8-K filings are short — use full text
                mda = full_text[:5000]
                risk = ""

            filing_date = parse_filing_date(fpath)

            records.append({
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "filepath": str(fpath),
                "mda_text": mda,
                "risk_text": risk,
                "full_text_length": len(full_text),
                "mda_length": len(mda),
                "risk_length": len(risk),
            })

    df = pd.DataFrame(records)

    if not df.empty and "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    return df


def validate_filings(df):
    """Quality checks on parsed filings."""
    if df.empty:
        return {"total_filings": 0, "note": "No filings found — run download first?"}

    return {
        "total_filings": len(df),
        "tickers_with_filings": df["ticker"].nunique(),
        "filings_per_ticker": df.groupby("ticker").size().to_dict(),
        "empty_mda_count": int((df["mda_length"] == 0).sum()),
        "avg_mda_length_chars": int(df["mda_length"].mean()),
        "missing_dates": int(df["filing_date"].isnull().sum()),
    }


def run(download=True, filing_type="10-Q"):
    """
    Main entry point.
    
    Args:
        download: If True, download from EDGAR first (slow, ~5-10 min)
        filing_type: "10-Q" or "8-K"
    
    Returns: (filings_df, validation_dict)
    """
    if download:
        download_filings(filing_types=[filing_type])

    df = build_filings_dataframe(filing_type=filing_type)
    validation = validate_filings(df)

    # Save
    output_path = RAW_DIR / f"filings_{filing_type.lower().replace('-', '')}.parquet"
    if not df.empty:
        df.to_parquet(output_path)
        logger.info(f"Saved to {output_path}")

    return df, validation


if __name__ == "__main__":
    for ft in SEC_FILING_TYPES:
        print(f"\n{'='*50}")
        print(f"  Processing {ft} filings")
        print(f"{'='*50}")

        df, val = run(download=True, filing_type=ft)

        print(f"\n=== {ft} Validation ===")
        for k, v in val.items():
            print(f"  {k}: {v}")

        if not df.empty:
            sample = df[df["mda_length"] > 0].head(1)
            if not sample.empty:
                print(f"\n=== Sample MD&A (first 200 chars) ===")
                print(f"  {sample.iloc[0]['ticker']}: {sample.iloc[0]['mda_text'][:200]}...")
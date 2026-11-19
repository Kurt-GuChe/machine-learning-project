
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT    = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
# time
START = "2000-01-01"
END   = "2026-08-31"

TICKERS = [
    # Technology
    "AAPL","MSFT","INTC","CSCO","ORCL","IBM","TXN","QCOM","ADBE","AMD",
    # Financials
    "JPM","BAC","WFC","C","GS","MS","AXP","USB","PNC","SCHW",
    # Healthcare
    "JNJ","PFE","MRK","ABT","TMO","UNH","BMY","AMGN","LLY","MDT",
    # Consumer
    "PG","KO","PEP","WMT","HD","MCD","COST","NKE","SBUX","TGT",
    # Energy / Industrials
    "XOM","CVX","COP","SLB","EOG","GE","HON","CAT","BA","MMM",
]
# trueday=day+1
END_EXCL = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

for tk in TICKERS:
    print(f"download {tk} ...")
    df = yf.download(tk, start=START, end=END_EXCL,
                     interval="1d", auto_adjust=True,
                     progress=False, threads=False)

    if df is None or df.empty:
        print(f"  {tk} nodata")
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(RAW_DIR / f"{tk}.csv")
    print(f"  → {tk}.parquet  ({len(df)} 行)")


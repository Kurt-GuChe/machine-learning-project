from pathlib import Path
import numpy as np
import pandas as pd
import backtest_yearlyrenew as report
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parent
OUT = BASE / "result_all_sectors_transfer_9to1"

report.PRED_DIR = OUT / "predictions"
report.OUT = OUT / "backtest_yearly"
report.START_YEAR = 2022
report.END_YEAR = 2026

TICKERS = [
    "AAPL", "MSFT", "INTC", "CSCO", "ORCL",
    "IBM", "TXN", "QCOM", "ADBE", "AMD",
    "JPM", "BAC", "WFC", "C", "GS",
    "MS", "AXP", "USB", "PNC", "SCHW",
    "JNJ", "PFE", "MRK", "ABT", "TMO",
    "UNH", "BMY", "AMGN", "LLY", "MDT",
    "PG", "KO", "PEP", "WMT", "HD",
    "MCD", "COST", "NKE", "SBUX", "TGT",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "GE", "HON", "CAT", "BA", "MMM",
]

def combine_predictions(ticker):
    predictions = []
    dates = []

    for year in range(2022, 2027):
        pred_path = report.PRED_DIR / f"{ticker}_{year}_pred.npy"
        date_path = report.PRED_DIR / f"{ticker}_{year}_dates.npy"

        if not pred_path.exists() or not date_path.exists():
            raise FileNotFoundError(
                f"Missing prediction: {ticker} {year}"
            )

        predictions.append(np.load(pred_path))
        dates.extend(
            np.load(date_path, allow_pickle=True).tolist()
        )

    combined_pred = np.concatenate(predictions)
    combined_dates = pd.DatetimeIndex(pd.to_datetime(dates))

    if len(combined_pred) != len(combined_dates):
        raise ValueError(f"Length mismatch: {ticker}")

    if combined_dates.has_duplicates or not combined_dates.is_monotonic_increasing:
        raise ValueError(f"Invalid dates: {ticker}")

    np.save(
        report.PRED_DIR / f"{ticker}_test_pred.npy",
        combined_pred,
    )

    np.save(
        report.PRED_DIR / f"{ticker}_test_dates.npy",
        combined_dates.strftime("%Y-%m-%d").to_numpy(),
    )



def show_figures():
    figure_dir = report.OUT / "figures"

    # Only search for portfolio equity curve
    keywords = ("portfolio", "equity")

    image_files = sorted(
        p for p in figure_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and all(word in p.stem.lower() for word in keywords)
    )

    if not image_files:
        print("Portfolio Equity Curve not found.")
        return

    # Display only one portfolio figure
    image_path = image_files[0]

    img = plt.imread(image_path)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":

     for folder in ("accounts", "trades", "signals", "figures"):
        (report.OUT / folder).mkdir(
            parents=True, exist_ok=True
        )

     for ticker in TICKERS:
        combine_predictions(ticker)

     report.main()

    # Display only the portfolio equity curve
     show_figures()
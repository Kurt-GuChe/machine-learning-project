
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

from dataset import load_one


# ============================================================
# 1. Configuration
# ============================================================

BASE = Path(__file__).resolve().parent

PARAM_DIR = BASE / "best_params"

EXPERIMENTS = {
    "A": BASE / "walkforward_relative_results",
    "B": BASE / "result_yearly_renew",
    "C": BASE / "result_all_sectors_transfer_9to1",
}

START_YEAR = 2022
END_YEAR = 2026

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

LABELS = [0, 1, 2]
CLASS_NAMES = ["Buy", "Hold", "Sell"]

OUT = BASE / "classification_results"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load Optuna parameters
# ============================================================

def load_parameters(ticker, year):

    path = PARAM_DIR / f"{ticker}_{year}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    window = int(config["barrier_window"])
    mult = float(config["atr_multiplier"])

    params = {
        k: v
        for k, v in config.items()
        if not k.startswith("_")
        and k not in ("barrier_window", "atr_multiplier")
    }

    return params, window, mult


# ============================================================
# 3. Reconstruct true labels
# ============================================================

def load_true_labels(ticker, year):

    params, window, mult = load_parameters(
        ticker, year
    )

    frame = load_one(
        ticker=ticker,
        params=params,
        window=window,
        mult=mult,
    )

    frame = frame.loc[
        (frame.index.year == year)
    ]

    return pd.Series(
        frame["label"].to_numpy(dtype=int),
        index=pd.DatetimeIndex(frame.index).normalize(),
        name="true_label",
    )


# ============================================================
# 4. Load saved predictions
# ============================================================

def load_predictions(folder, ticker, year):

    pred_dir = folder / "predictions"

    pred_path = (
        pred_dir / f"{ticker}_{year}_pred.npy"
    )

    dates_path = (
        pred_dir / f"{ticker}_{year}_dates.npy"
    )

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    if not dates_path.exists():
        raise FileNotFoundError(dates_path)

    pred = np.load(
        pred_path,
        allow_pickle=False,
    ).astype(int)

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            np.load(
                dates_path,
                allow_pickle=True,
            )
        )
    ).normalize()

    if len(pred) != len(dates):
        raise ValueError(
            f"Prediction length mismatch: {ticker} {year}"
        )

    if dates.has_duplicates:
        raise ValueError(
            f"Duplicate prediction dates: {ticker} {year}"
        )

    if not np.isin(pred, LABELS).all():
        raise ValueError(
            f"Invalid predictions: {ticker} {year}"
        )

    if not (dates.year == year).all():
        raise ValueError(
            f"Wrong prediction year: {ticker} {year}"
        )

    return pd.Series(
        pred,
        index=dates,
        name="prediction",
    )


# ============================================================
# 5. Align predictions with true labels
# ============================================================

def evaluate_one_year(folder, ticker, year):

    y_true = load_true_labels(ticker, year)

    y_pred = load_predictions(
        folder, ticker, year
    )

    # Every prediction must have a matching true-label date.
    missing = y_pred.index.difference(y_true.index)

    if len(missing) > 0:
        raise ValueError(
            f"Unmatched dates: {ticker} {year}: "
            f"{missing[:5].tolist()}"
        )

    aligned = pd.DataFrame({
        "true": y_true.reindex(y_pred.index),
        "pred": y_pred,
    })

    # The test set can contain label=-1.
    valid = aligned["true"].isin(LABELS)

    excluded = int((~valid).sum())

    aligned = aligned.loc[valid]

    if aligned.empty:
        raise ValueError(
            f"No valid labels: {ticker} {year}"
        )

    return (
        aligned["true"].to_numpy(dtype=int),
        aligned["pred"].to_numpy(dtype=int),
        excluded,
    )


# ============================================================
# 6. Calculate classification metrics
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
    }


def normalized_confusion_matrix(cm):

    row_sum = cm.sum(axis=1, keepdims=True)

    return np.divide(
        cm.astype(float),
        row_sum,
        out=np.zeros_like(cm, dtype=float),
        where=row_sum != 0,
    )


# ============================================================
# 7. Evaluate one experiment (50 stocks)
# ============================================================

def evaluate_experiment(name, folder):

    stock_rows = []
    yearly_rows = []

    total_cm = np.zeros((3, 3), dtype=np.int64)
    normalized_matrices = []

    for ticker in TICKERS:

        true_parts = []
        pred_parts = []

        excluded_total = 0

        for year in range(
            START_YEAR, END_YEAR + 1
        ):

            y_true, y_pred, excluded = (
                evaluate_one_year(
                    folder, ticker, year
                )
            )

            metrics = calculate_metrics(
                y_true, y_pred
            )

            yearly_rows.append({
                "experiment": name,
                "ticker": ticker,
                "year": year,
                "samples": len(y_true),
                "excluded": excluded,
                **metrics,
            })

            true_parts.append(y_true)
            pred_parts.append(y_pred)

            excluded_total += excluded

        # Combine all years for this stock.
        stock_true = np.concatenate(true_parts)
        stock_pred = np.concatenate(pred_parts)

        stock_metrics = calculate_metrics(
            stock_true, stock_pred
        )

        stock_rows.append({
            "experiment": name,
            "ticker": ticker,
            "samples": len(stock_true),
            "excluded": excluded_total,
            **stock_metrics,
        })

        cm = confusion_matrix(
            stock_true,
            stock_pred,
            labels=LABELS,
        )

        total_cm += cm

        normalized_matrices.append(
            normalized_confusion_matrix(cm)
        )

    stock_df = pd.DataFrame(stock_rows)
    yearly_df = pd.DataFrame(yearly_rows)

    # Equal-weight average across the 50 stocks.
    average_cm = np.mean(
        normalized_matrices,
        axis=0,
    )

    summary = {
        "experiment": name,
        "stocks": len(stock_df),
        "samples": int(stock_df["samples"].sum()),
        "excluded": int(stock_df["excluded"].sum()),
        "accuracy": stock_df["accuracy"].mean(),
        "macro_f1": stock_df["macro_f1"].mean(),
        "weighted_f1": stock_df["weighted_f1"].mean(),
    }

    return (
        stock_df,
        yearly_df,
        summary,
        total_cm,
        average_cm,
    )


# ============================================================
# 8. Save confusion matrix figures
# ============================================================

def plot_confusion_matrix(
    cm, experiment, normalized=False
):

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(cm, cmap="Blues", vmin=0)

    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    kind = (
        "Mean Normalized"
        if normalized else "Summed"
    )

    ax.set_title(
        f"Experiment {experiment} - {kind} Confusion Matrix"
    )

    for i in range(3):
        for j in range(3):

            value = (
                f"{cm[i, j]:.2%}"
                if normalized
                else f"{cm[i, j]:d}"
            )

            ax.text(
                j, i, value,
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()

    suffix = (
        "mean_normalized"
        if normalized else "summed"
    )

    fig.savefig(
        OUT / f"{experiment}_{suffix}_confusion.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 9. Run A, B and C
# ============================================================

def main():

    all_stocks = []
    all_years = []
    summaries = []

    for name, folder in EXPERIMENTS.items():

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Experiment {name} folder not found: {folder}"
            )

        (
            stock_df,
            yearly_df,
            summary,
            total_cm,
            average_cm,
        ) = evaluate_experiment(name, folder)

        all_stocks.append(stock_df)
        all_years.append(yearly_df)
        summaries.append(summary)

        pd.DataFrame(
            total_cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(
            OUT / f"{name}_summed_confusion.csv"
        )

        pd.DataFrame(
            average_cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(
            OUT / f"{name}_mean_normalized_confusion.csv"
        )

        plot_confusion_matrix(
            total_cm, name, normalized=False
        )

        plot_confusion_matrix(
            average_cm, name, normalized=True
        )

    stock_results = pd.concat(
        all_stocks, ignore_index=True
    )

    yearly_results = pd.concat(
        all_years, ignore_index=True
    )

    summary_df = pd.DataFrame(summaries)

    stock_results.to_csv(
        OUT / "stock_classification.csv",
        index=False,
    )

    yearly_results.to_csv(
        OUT / "yearly_classification.csv",
        index=False,
    )

    summary_df.to_csv(
        OUT / "ABC_classification_summary.csv",
        index=False,
    )

    print("\n50-stock average classification results:")
    print(
        summary_df[
            [
                "experiment",
                "stocks",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nResults saved to: {OUT}")


if __name__ == "__main__":
    main()

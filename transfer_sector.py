
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from dataset import FEATURE_COLS, SEQ_LEN, build_rolling_dataset
from backtest import evaluate_cash_strategy
from lstm_final_60days import LSTMAttention, apply_mode
from warkforward_reneweveryyear import (
    DEVICE, TRAIN_YEARS, BATCH_SIZE, LR, WEIGHT_DECAY,
    EPOCHS, PATIENCE, SEED, set_seed, clone_state_dict,
    predict_proba, evaluate_loss, signal_statistics,
    calculate_validation_score, valid_sharpe, train_year,
)

BASE = Path(__file__).resolve().parent
PARAM_DIR = BASE / "best_params"

OUT = BASE / "result_all_sectors_transfer_9to1"
MODEL_DIR = OUT / "models"
PRED_DIR = OUT / "predictions"
METRIC_DIR = OUT / "metrics"
LOG_DIR = OUT / "pretraining_logs"
METRICS_FILE = METRIC_DIR / "transfer_yearly.csv"

# Previously completed experiments.
LEGACY_DIRS = [
    BASE / "result_sector_transfer_9to1",
    BASE / "result_other_sectors_transfer_9to1",
]

SECTORS = {
    "Technology": [
        "AAPL", "MSFT", "INTC", "CSCO", "ORCL",
        "IBM", "TXN", "QCOM", "ADBE", "AMD",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "C", "GS",
        "MS", "AXP", "USB", "PNC", "SCHW",
    ],
    "Healthcare": [
        "JNJ", "PFE", "MRK", "ABT", "TMO",
        "UNH", "BMY", "AMGN", "LLY", "MDT",
    ],
    "Consumer": [
        "PG", "KO", "PEP", "WMT", "HD",
        "MCD", "COST", "NKE", "SBUX", "TGT",
    ],
    "Energy_Industrials": [
        "XOM", "CVX", "COP", "SLB", "EOG",
        "GE", "HON", "CAT", "BA", "MMM",
    ],
}

TICKER_SECTOR = {
    ticker: sector
    for sector, tickers in SECTORS.items()
    for ticker in tickers
}

ALL_TICKERS = [
    ticker
    for tickers in SECTORS.values()
    for ticker in tickers
]

SOURCE_EPOCHS = EPOCHS
SOURCE_PATIENCE = PATIENCE
SOURCE_LR = LR


def make_dirs():
    for folder in (MODEL_DIR, PRED_DIR, METRIC_DIR, LOG_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def read_metrics():
    if not METRICS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(METRICS_FILE)
    if df.empty:
        return df

    df["test_year"] = df["test_year"].astype(int)
    return df.drop_duplicates(
        ["ticker", "test_year"], keep="last"
    )


def save_metrics(df):
    if df.empty:
        return

    df = df.drop_duplicates(
        ["ticker", "test_year"], keep="last"
    )
    df.sort_values(
        ["test_year", "ticker"]
    ).to_csv(METRICS_FILE, index=False)


def import_previous_results():
    """Copy completed experiments into unified output.

    Original files are never deleted or overwritten.
    """
    df = read_metrics()
    existing = set()

    if not df.empty:
        existing = set(
            zip(df["ticker"], df["test_year"])
        )

    added = 0

    for legacy in LEGACY_DIRS:
        old_metrics = legacy / "metrics" / "transfer_yearly.csv"

        if not old_metrics.exists():
            continue

        old_df = pd.read_csv(old_metrics)

        for _, row in old_df.iterrows():
            ticker = str(row["ticker"])
            year = int(row["test_year"])
            key = (ticker, year)

            if ticker not in TICKER_SECTOR or key in existing:
                continue

            required = [
                (
                    legacy / "models" / f"{ticker}_{year}.pt",
                    MODEL_DIR / f"{ticker}_{year}.pt",
                ),
                (
                    legacy / "predictions" / f"{ticker}_{year}_pred.npy",
                    PRED_DIR / f"{ticker}_{year}_pred.npy",
                ),
                (
                    legacy / "predictions" / f"{ticker}_{year}_dates.npy",
                    PRED_DIR / f"{ticker}_{year}_dates.npy",
                ),
            ]

            if not all(src.exists() for src, _ in required):
                continue

            # Validate that saved prediction arrays can be read.
            try:
                pred = np.load(required[1][0], allow_pickle=False)
                dates = np.load(required[2][0], allow_pickle=True)
                if len(pred) == 0 or len(pred) != len(dates):
                    continue
            except (ValueError, OSError):
                continue

            for src, dst in required:
                if not dst.exists():
                    shutil.copy2(src, dst)

            new_row = row.to_dict()
            new_row["sector"] = TICKER_SECTOR[ticker]

            df = pd.concat(
                [df, pd.DataFrame([new_row])],
                ignore_index=True,
            )

            existing.add(key)
            added += 1

    save_metrics(df)
    print(f"Imported previous completed results: {added}")


def load_dataset(ticker, year):
    path = PARAM_DIR / f"{ticker}_{year}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Optuna parameters: {path}"
        )

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

    data = build_rolling_dataset(
        ticker=ticker,
        params=params,
        window=window,
        mult=mult,
        test_year=year,
        train_years=TRAIN_YEARS,
        seq_len=SEQ_LEN,
    )

    if data is None:
        raise ValueError(f"No dataset: {ticker} {year}")

    train, valid, test, scaler = data

    for name, part in (("train", train), ("valid", valid)):
        if not np.isin(part[1], [0, 1, 2]).all():
            raise ValueError(
                f"Invalid {name} labels: {ticker} {year}"
            )

    if train[0].shape[2] != len(FEATURE_COLS):
        raise ValueError(f"Feature mismatch: {ticker}")

    return data, params, window, mult


def source_checkpoint_path(ticker, year):
    return MODEL_DIR / f"source_excluding_{ticker}_{year}.pt"


def pretrain_sector(year, target_ticker):
    sector = TICKER_SECTOR[target_ticker]

    source_tickers = [
        t for t in SECTORS[sector]
        if t != target_ticker
    ]

    checkpoint_path = source_checkpoint_path(
        target_ticker, year
    )

    # Resume source pretraining if it was already completed.
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            checkpoint.get("source_tickers") == source_tickers
            and checkpoint.get("year") == year
        ):
            print(
                f"  Reuse source model: {target_ticker} {year}"
            )
            return (
                checkpoint["model_state_dict"],
                checkpoint["best_epoch"],
                checkpoint["best_valid_loss"],
            )

    set_seed(SEED + year)

    model = LSTMAttention(
        n_feat=len(FEATURE_COLS)
    ).to(DEVICE)

    datasets = []
    valid_parts = []
    counts = {}

    for ticker in source_tickers:
        data, _, _, _ = load_dataset(ticker, year)
        train, valid, _, _ = data

        X, y = train[:2]

        datasets.append(
            TensorDataset(
                torch.from_numpy(X),
                torch.from_numpy(y),
            )
        )

        valid_parts.append((valid[0], valid[1]))

        counts[ticker] = {
            "train": len(X),
            "valid": len(valid[0]),
        }

    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SOURCE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    logs = []

    for epoch in range(1, SOURCE_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            total_n += len(yb)

        train_loss = total_loss / total_n

        valid_total = sum(
            len(y) for _, y in valid_parts
        )

        valid_loss = sum(
            evaluate_loss(model, X, y) * len(y)
            for X, y in valid_parts
        ) / valid_total

        logs.append({
            "sector": sector,
            "ticker": target_ticker,
            "year": year,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        })

        if (
            np.isfinite(valid_loss)
            and valid_loss < best_loss - 1e-6
        ):
            best_loss = valid_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  Source epoch {epoch:3d} | "
                f"Train={train_loss:.4f} | "
                f"Valid={valid_loss:.4f}"
            )

        if wait >= SOURCE_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(
            f"Source training failed: {target_ticker} {year}"
        )

    pd.DataFrame(logs).to_csv(
        LOG_DIR / f"source_excluding_{target_ticker}_{year}.csv",
        index=False,
    )

    torch.save(
        {
            "sector": sector,
            "year": year,
            "target_ticker": target_ticker,
            "source_tickers": source_tickers,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "stock_counts": counts,
        },
        checkpoint_path,
    )

    return best_state, best_epoch, best_loss


def run_target(ticker, year, source_state, source_epoch, source_loss):
    start = time.perf_counter()
    set_seed(SEED + year)

    data, params, window, mult = load_dataset(ticker, year)
    train, valid, test, scaler = data

    X_train, _, _, _, train_dates = train
    X_valid, _, valid_open, _, valid_dates = valid
    X_test, _, test_open, _, test_dates = test

    model = LSTMAttention(
        n_feat=len(FEATURE_COLS)
    ).to(DEVICE)

    model.load_state_dict(source_state)

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("fc.")
        )

    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable fc parameters")

    print(
        f"{ticker} {year} | "
        f"train={len(X_train)} "
        f"valid={len(X_valid)} "
        f"test={len(X_test)}"
    )

    (
        before_loss,
        best_loss,
        epochs,
        selected_sharpe,
        selected_score,
        strategy,
    ) = train_year(model, train, valid)

    mode = strategy["mode"]
    thresholds = strategy["thresholds"]

    val_pred = apply_mode(
        predict_proba(model, X_valid),
        thresholds,
        mode,
    )

    test_pred = apply_mode(
        predict_proba(model, X_test),
        thresholds,
        mode,
    )

    val_eval = evaluate_cash_strategy(
        val_pred, valid_open, valid_dates
    )

    test_eval = evaluate_cash_strategy(
        test_pred, test_open, test_dates
    )

    val_stats = signal_statistics(val_pred)
    test_stats = signal_statistics(test_pred)

    val_score, _ = calculate_validation_score(
        val_eval["sharpe"],
        val_eval["entries"],
    )

    mu, sd = scaler

    torch.save(
        {
            "ticker": ticker,
            "year": year,
            "sector": TICKER_SECTOR[ticker],
            "training_type": "sector_transfer_frozen_head",
            "source_tickers": [
                t for t in SECTORS[TICKER_SECTOR[ticker]]
                if t != ticker
            ],
            "model_state_dict": clone_state_dict(model),
            "scaler_mean": mu.to_dict(),
            "scaler_std": sd.to_dict(),
            "params": params,
            "barrier_window": window,
            "atr_multiplier": mult,
            "thresholds": thresholds,
            "mode": mode,
            "train_years": TRAIN_YEARS,
            "seq_len": SEQ_LEN,
            "epochs": epochs,
            "validation_strategy": strategy,
            "source_best_epoch": source_epoch,
            "source_valid_loss": source_loss,
        },
        MODEL_DIR / f"{ticker}_{year}.pt",
    )

    dates = pd.DatetimeIndex(
        test_dates
    ).strftime("%Y-%m-%d").to_numpy()

    np.save(
        PRED_DIR / f"{ticker}_{year}_pred.npy",
        test_pred,
    )

    np.save(
        PRED_DIR / f"{ticker}_{year}_dates.npy",
        dates,
    )

    row = {
        "sector": TICKER_SECTOR[ticker],
        "ticker": ticker,
        "test_year": year,
        "train_start": str(train_dates[0].date()),
        "train_end": str(train_dates[-1].date()),
        "valid_start": str(valid_dates[0].date()),
        "valid_end": str(valid_dates[-1].date()),
        "test_start": str(test_dates[0].date()),
        "test_end": str(test_dates[-1].date()),
        "n_train": len(X_train),
        "n_valid": len(X_valid),
        "n_test": len(X_test),
        "epochs": epochs,
        "source_epoch": source_epoch,
        "source_valid_loss": source_loss,
        "val_loss_before": before_loss,
        "val_loss_best": best_loss,
        "val_sharpe": valid_sharpe(val_eval["sharpe"]),
        "val_score": val_score,
        "test_sharpe": valid_sharpe(test_eval["sharpe"]),
        "thresholds": str(thresholds),
        "mode": mode,
        "val_buy": val_stats["buy"],
        "val_hold": val_stats["hold"],
        "val_sell": val_stats["sell"],
        "val_entries": val_eval["entries"],
        "test_buy": test_stats["buy"],
        "test_hold": test_stats["hold"],
        "test_sell": test_stats["sell"],
        "test_entries": test_eval["entries"],
        "epoch_selection_sharpe": selected_sharpe,
        "epoch_selection_score": selected_score,
        "seconds": time.perf_counter() - start,
    }

    print(
        f"  Val Sharpe={row['val_sharpe']:.3f} | "
        f"Test Sharpe={row['test_sharpe']:.3f} | "
        f"Entries={row['test_entries']}"
    )

    return row


def is_completed(ticker, year, completed):
    if (ticker, year) not in completed:
        return False

    required = [
        MODEL_DIR / f"{ticker}_{year}.pt",
        PRED_DIR / f"{ticker}_{year}_pred.npy",
        PRED_DIR / f"{ticker}_{year}_dates.npy",
    ]

    if not all(p.exists() for p in required):
        return False

    try:
        preds = np.load(required[1])
        dates = np.load(required[2], allow_pickle=True)
        return len(preds) > 0 and len(preds) == len(dates)
    except (ValueError, OSError):
        return False


def combine_predictions(ticker, years):
    predictions = []
    date_parts = []

    for year in years:
        pred_path = PRED_DIR / f"{ticker}_{year}_pred.npy"
        date_path = PRED_DIR / f"{ticker}_{year}_dates.npy"

        if not pred_path.exists() or not date_path.exists():
            raise FileNotFoundError(
                f"Missing prediction: {ticker} {year}"
            )

        predictions.append(np.load(pred_path))

        date_parts.extend(
            np.load(
                date_path,
                allow_pickle=True,
            ).tolist()
        )

    dates = pd.DatetimeIndex(
        pd.to_datetime(date_parts)
    )

    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"Invalid dates: {ticker}")

    np.save(
        PRED_DIR / f"{ticker}_test_pred.npy",
        np.concatenate(predictions),
    )

    np.save(
        PRED_DIR / f"{ticker}_test_dates.npy",
        dates.strftime("%Y-%m-%d").to_numpy(),
    )


def run_backtest(start_year, end_year):
    import backtest_yearlyrenew as report

    report.PRED_DIR = PRED_DIR
    report.OUT = OUT / "backtest_yearly"
    report.START_YEAR = start_year
    report.END_YEAR = end_year

    for folder in (
        "accounts", "trades", "signals", "figures"
    ):
        (report.OUT / folder).mkdir(
            parents=True,
            exist_ok=True,
        )

    report.main()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, default=2022)
    parser.add_argument("--end", type=int, default=2026)

    parser.add_argument(
        "--sector",
        choices=list(SECTORS) + ["all"],
        default="all",
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
    )

    parser.add_argument(
        "--backtest-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be <= --end")

    make_dirs()
    import_previous_results()

    years = list(range(args.start, args.end + 1))

    selected = (
        ALL_TICKERS
        if args.sector == "all"
        else SECTORS[args.sector]
    )

    df = read_metrics()
    completed = set()

    if not df.empty:
        completed = set(
            zip(df["ticker"], df["test_year"])
        )

    if args.backtest_only:
        for ticker in selected:
            for year in years:
                if not is_completed(ticker, year, completed):
                    raise RuntimeError(
                        f"Incomplete: {ticker} {year}"
                    )

        for ticker in selected:
            combine_predictions(ticker, years)

        # Backtest script reads all combined prediction files.
        # Avoid mixing a partial sector run with other sectors.
        if args.sector != "all":
            raise ValueError(
                "Unified backtest requires --sector all"
            )

        run_backtest(args.start, args.end)
        return

    # Only check Optuna files for experiments still needing training.
    pending = [
        (ticker, year)
        for year in years
        for ticker in selected
        if not is_completed(ticker, year, completed)
    ]

    required_params = {
        (source, year)
        for ticker, year in pending
        for source in SECTORS[TICKER_SECTOR[ticker]]
    }

    missing = [
        f"{ticker}_{year}.json"
        for ticker, year in sorted(required_params)
        if not (
            PARAM_DIR / f"{ticker}_{year}.json"
        ).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Optuna files: " + ", ".join(missing)
        )

    total = len(selected) * len(years)
    done = total - len(pending)

    print(
        f"Device={DEVICE} | "
        f"Stocks={len(selected)} | "
        f"Years={args.start}-{args.end}"
    )
    print(
        f"Completed={done}/{total} | "
        f"Remaining={len(pending)}"
    )

    for year in years:
        for ticker in selected:
            if is_completed(ticker, year, completed):
                print(f"Skip: {ticker} {year}")
                continue

            print(
                f"\n===== {TICKER_SECTOR[ticker]} | "
                f"{ticker} | {year} =====",
                flush=True,
            )

            source_state, source_epoch, source_loss = (
                pretrain_sector(year, ticker)
            )

            row = run_target(
                ticker,
                year,
                source_state,
                source_epoch,
                source_loss,
            )

            df = df[
                ~(
                    (df["ticker"] == ticker)
                    & (df["test_year"] == year)
                )
            ] if not df.empty else df

            df = pd.concat(
                [df, pd.DataFrame([row])],
                ignore_index=True,
            )

            save_metrics(df)
            completed.add((ticker, year))

            del source_state

    # Generate combined predictions for every selected stock.
    for ticker in selected:
        combine_predictions(ticker, years)

    with (OUT / "run_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "experiment": "all_sectors_9_to_1",
                "sectors": SECTORS,
                "start_year": args.start,
                "end_year": args.end,
                "source_selection": "pooled_validation_loss",
                "target_selection": "penalized_validation_sharpe",
                "target_finetuning": "fc_only",
            },
            f,
            indent=2,
        )

    if args.backtest:
        if args.sector != "all":
            raise ValueError(
                "Use --sector all for unified backtest"
            )

        run_backtest(args.start, args.end)

    print(f"\nFinished: {OUT}")


if __name__ == "__main__":
    main()

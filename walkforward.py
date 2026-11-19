
"""
Annual Walk-Forward LSTM-Attention

2022: Train 2016-2020, Valid 2021, Test 2022
2023+: Inherit previous year's best weights.

Model selection: Validation cash-account Sharpe
Trading: Buy 100%, Hold, Sell 100%
No bull/bear market strategy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import FEATURE_COLS, SEQ_LEN, build_rolling_dataset
from backtest import SENTINEL, evaluate_cash_strategy
from lstm_final_60days import LSTMAttention, apply_mode


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PARAM_DIR = BASE_DIR / "best_params"

WF_DIR = BASE_DIR / "walkforward_relative_results"
MODEL_DIR = WF_DIR / "models"
PRED_DIR = WF_DIR / "predictions"
METRIC_DIR = WF_DIR / "metrics"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START_YEAR, END_YEAR = 2022, 2026
TRAIN_YEARS = 5

BATCH_SIZE = 128
PRED_BATCH_SIZE = 256

EPOCHS_FIRST = 80
EPOCHS_FINETUNE = 30

LR_FIRST = 5e-4
LR_FINETUNE = 1e-4

WEIGHT_DECAY = 1e-4
PATIENCE = 10
SEED = 42

THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]

MIN_VALID_ENTRIES = 3
MAX_DIRECTION_RATIO = 0.95
SHARPE_TIE_TOL = 0.03
MIN_DELTA = 1e-6
TARGET_ENTRIES = 5
PENALTY_LAMBDA = 0.1
PENALTY_K = 0.5

# ============================================================
# 2. Utilities
# ============================================================

def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    else:
        torch.set_num_threads(
            max(1, min(4, torch.get_num_threads()))
        )


def clone_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def predict_proba(model, X):
    model.eval()
    probs = []

    with torch.inference_mode():
        for i in range(0, len(X), PRED_BATCH_SIZE):
            xb = torch.from_numpy(
                X[i:i + PRED_BATCH_SIZE]
            ).to(DEVICE)

            probs.append(
                torch.softmax(model(xb), dim=1)
                .cpu().numpy()
            )

    return np.concatenate(probs)


def evaluate_loss(model, X, y):
    model.eval()
    total = 0.0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.inference_mode():
        for i in range(0, len(X), PRED_BATCH_SIZE):
            xb = torch.from_numpy(
                X[i:i + PRED_BATCH_SIZE]
            ).to(DEVICE)

            yb = torch.from_numpy(
                y[i:i + PRED_BATCH_SIZE]
            ).to(DEVICE)

            total += criterion(model(xb), yb).item()

    return total / len(y)


def signal_statistics(pred):
    buy, hold, sell = np.bincount(
        np.asarray(pred, dtype=int),
        minlength=3
    )[:3]

    directional = buy + sell

    return {
        "buy": int(buy),
        "hold": int(hold),
        "sell": int(sell),
        "direction_ratio": (
            max(buy, sell) / directional
            if directional else 1.0
        ),
    }
def calculate_validation_score(sharpe, entries):

    if not np.isfinite(sharpe) or sharpe <= SENTINEL / 2:
        return np.nan, np.nan

    penalty = PENALTY_LAMBDA * np.expm1(
        PENALTY_K * max(0, TARGET_ENTRIES - entries)
    )

    score = sharpe - penalty

    return float(score), float(penalty)


# ============================================================
# 3. Validation strategy selection
# ============================================================

def select_validation_strategy(probs, open_, dates):
    """
    Select the best strategy using penalized validation Sharpe.
    """

    candidates = [(None, "argmax")]

    candidates += [
        ((b, s, 0.0), "threshold")
        for b in THRESHOLD_GRID
        for s in THRESHOLD_GRID
    ]

    best = None

    for thresholds, mode in candidates:

        pred = apply_mode(probs, thresholds, mode)
        stats = signal_statistics(pred)

        # Reject excessively one-sided predictions
        if stats["direction_ratio"] > MAX_DIRECTION_RATIO:
            continue

        evaluation = evaluate_cash_strategy(
            pred, open_, dates
        )

        sharpe = float(evaluation["sharpe"])
        entries = int(evaluation["entries"])

        score, penalty = calculate_validation_score(
            sharpe, entries
        )

        if not np.isfinite(score):
            continue

        complexity = (
            0.0 if mode == "argmax"
            else abs(thresholds[0] - 0.5)
            + abs(thresholds[1] - 0.5)
        )

        candidate = {
            "sharpe": sharpe,
            "score": score,
            "penalty": penalty,
            "thresholds": thresholds,
            "mode": mode,
            "complexity": complexity,
            "entries": entries,
            "exposure": evaluation["exposure"],
            **stats,
        }

        if best is None:
            best = candidate
            continue

        if score > best["score"] + MIN_DELTA:
            best = candidate

    return best


# ============================================================
# 4. Train / Fine-tune one year
# ============================================================

def train_year(model, train, valid, first_year=False):

    X_train, y_train = train[:2]
    X_valid, y_valid, open_valid, _, dates_valid = valid

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    lr = LR_FIRST if first_year else LR_FINETUNE

    max_epochs = (
        EPOCHS_FIRST if first_year else EPOCHS_FINETUNE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    def evaluate_strategy():

        probs = predict_proba(model, X_valid)

        return select_validation_strategy(
            probs, open_valid, dates_valid
        )

    before_loss = evaluate_loss(
        model, X_valid, y_valid
    )

    best_state = None
    best_strategy = None

    best_score = -float("inf")
    best_loss = float("inf")

    fallback_state = None
    fallback_loss = float("inf")

    # Evaluate inherited weights before fine-tuning
    if not first_year:

        fallback_state = clone_state_dict(model)
        fallback_loss = before_loss

        strategy = evaluate_strategy()

        if strategy is not None:

            best_state = clone_state_dict(model)
            best_strategy = strategy.copy()

            best_score = strategy["score"]
            best_loss = before_loss

    patience_counter = 0
    epochs_run = 0

    for epoch in range(max_epochs):

        model.train()
        total_loss = 0.0

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            loss = criterion(model(xb), yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(X_train)

        valid_loss = evaluate_loss(
            model, X_valid, y_valid
        )

        strategy = evaluate_strategy()
        epochs_run = epoch + 1

        # Keep lowest-loss checkpoint as fallback
        if (
            np.isfinite(valid_loss)
            and valid_loss < fallback_loss - MIN_DELTA
        ):

            fallback_loss = valid_loss
            fallback_state = clone_state_dict(model)

        valid_sharpe = (
            strategy["sharpe"]
            if strategy is not None
            else np.nan
        )

        valid_score = (
            strategy["score"]
            if strategy is not None
            else np.nan
        )

        entries = (
            strategy["entries"]
            if strategy is not None
            else 0
        )

        # Select checkpoint using penalized Score
        if (
            strategy is not None
            and valid_score > best_score + MIN_DELTA
        ):

            best_score = valid_score
            best_loss = valid_loss

            best_state = clone_state_dict(model)
            best_strategy = strategy.copy()

            patience_counter = 0

        else:

            patience_counter += 1

        if epoch == 0 or (epoch + 1) % 5 == 0:

            print(
                f"  Epoch {epoch + 1:3d} | "
                f"Train={train_loss:.4f} | "
                f"Valid={valid_loss:.4f} | "
                f"Sharpe={valid_sharpe:.3f} | "
                f"Score={valid_score:.3f} | "
                f"Entries={entries}"
            )

        # Early stopping based on Score improvement
        if patience_counter >= PATIENCE:

            print(f"  Early stop: epoch {epoch + 1}")
            break

    # Fallback if no valid Score checkpoint
    if best_state is None:

        if fallback_state is None:
            raise RuntimeError("No valid checkpoint")

        best_state = fallback_state
        best_loss = fallback_loss
        best_score = np.nan

        best_strategy = {
            "thresholds": None,
            "mode": "argmax",
            "sharpe": np.nan,
            "score": np.nan,
            "penalty": np.nan,
            "entries": 0,
            "exposure": 0.0,
            "buy": 0,
            "hold": 0,
            "sell": 0,
            "direction_ratio": np.nan,
        }

        print("  No valid Score; using lowest loss")

    model.load_state_dict(best_state)

    best_sharpe = best_strategy["sharpe"]

    print(
        f"  Selected: {best_strategy['mode']} | "
        f"Thresholds={best_strategy['thresholds']} | "
        f"Sharpe={best_sharpe:.3f} | "
        f"Score={best_score:.3f} | "
        f"Entries={best_strategy['entries']}"
    )

    return (
        before_loss,
        best_loss,
        epochs_run,
        best_sharpe,
        best_score,
        best_strategy,
    )


# ============================================================
# 5. Walk-Forward one stock
# ============================================================

def run_stock(
    ticker,
    start_year=START_YEAR,
    end_year=END_YEAR,
):

    

    # Initialize once: later years inherit weights
    model = LSTMAttention(
        n_feat=len(FEATURE_COLS)
    ).to(DEVICE)

    all_predictions = []
    all_dates = []
    yearly_rows = []

    for year in range(start_year, end_year + 1):

        start_time = time.perf_counter()
        # Load year-specific Optuna parameters
        param_path = PARAM_DIR / f"{ticker}_{year}.json"

        with param_path.open("r", encoding="utf-8") as f:
            params = json.load(f)

        params = params.copy()

        window = int(params.pop("barrier_window"))
        multiplier = float(params.pop("atr_multiplier"))

        # Remove Optuna metadata
        params = {
            k: v for k, v in params.items()
            if not k.startswith("_")
        }
        dataset = build_rolling_dataset(
            ticker=ticker,
            params=params,
            window=window,
            mult=multiplier,
            test_year=year,
            train_years=TRAIN_YEARS,
            seq_len=SEQ_LEN,
        )

        if dataset is None:
            print(f"{ticker} {year}: insufficient data")
            break

        train, valid, test, scaler = dataset
        mu, sd = scaler

        X_train, y_train, _, _, dates_train = train
        X_valid, y_valid, open_valid, _, dates_valid = valid
        X_test, y_test, open_test, _, dates_test = test

        for name, labels in [
            ("Train", y_train),
            ("Valid", y_valid),
        ]:
            if not np.isin(labels, [0, 1, 2]).all():
                raise ValueError(
                    f"{ticker} {year}: invalid {name} labels"
                )

        first_year = year == start_year

        print(
            f"\n{ticker} | {year} | "
            f"{'Scratch' if first_year else 'Fine-tune'}"
        )
        print(
            f"  Train={len(X_train)} "
            f"Valid={len(X_valid)} "
            f"Test={len(X_test)}"
        )

        # Train or fine-tune
        (
            before_loss,
            best_loss,
            epochs,
            epoch_sharpe,
            epoch_score,
            strategy,
        ) = train_year(
            model, train, valid, first_year
        )

        thresholds = strategy["thresholds"]
        mode = strategy["mode"]

        # Evaluate selected strategy on validation
        valid_pred = apply_mode(
            predict_proba(model, X_valid),
            thresholds,
            mode,
        )

        valid_eval = evaluate_cash_strategy(
            valid_pred, open_valid, dates_valid
        )

        valid_stats = signal_statistics(valid_pred)

        # Out-of-sample prediction
        test_pred = apply_mode(
            predict_proba(model, X_test),
            thresholds,
            mode,
        )

        test_eval = evaluate_cash_strategy(
            test_pred, open_test, dates_test
        )

        test_stats = signal_statistics(test_pred)

        # Save model checkpoint
        checkpoint = {
            "ticker": ticker,
            "year": year,
            "model_state_dict": clone_state_dict(model),
            "scaler_mean": mu.to_dict(),
            "scaler_std": sd.to_dict(),
            "params": params,
            "barrier_window": window,
            "atr_multiplier": multiplier,
            "thresholds": thresholds,
            "mode": mode,
            "train_years": TRAIN_YEARS,
            "seq_len": SEQ_LEN,
            "epochs": epochs,
            "val_loss_best": best_loss,
            "epoch_selection_sharpe": epoch_sharpe,
            "epoch_selection_score": epoch_score,
            "selection_metric": (
                "penalized_cash_account_validation_sharpe"
            ),
            "validation_strategy": strategy,
            "trading_rule": "long_only_full_position",
            "market_regime": False,
        }

        torch.save(
            checkpoint,
            MODEL_DIR / f"{ticker}_{year}.pt",
        )

        # Save yearly test predictions
        date_strings = dates_test.strftime(
            "%Y-%m-%d"
        ).to_numpy()

        np.save(
            PRED_DIR / f"{ticker}_{year}_pred.npy",
            test_pred,
        )

        np.save(
            PRED_DIR / f"{ticker}_{year}_dates.npy",
            date_strings,
        )

        all_predictions.append(test_pred)
        all_dates.extend(date_strings.tolist())

        elapsed = time.perf_counter() - start_time

        def valid_sharpe(value):
            return (
                float(value)
                if np.isfinite(value)
                and value > SENTINEL / 2
                else np.nan
            )

        yearly_rows.append({
            "ticker": ticker,
            "test_year": year,
            "train_start": str(dates_train[0].date()),
            "train_end": str(dates_train[-1].date()),
            "valid_start": str(dates_valid[0].date()),
            "valid_end": str(dates_valid[-1].date()),
            "test_start": str(dates_test[0].date()),
            "test_end": str(dates_test[-1].date()),
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "n_test": len(X_test),
            "epochs": epochs,
            "val_loss_before": before_loss,
            "val_loss_best": best_loss,
            "val_sharpe": valid_sharpe(valid_eval["sharpe"]),

            "val_score": (
                calculate_validation_score(
                    valid_eval["sharpe"],
                    valid_eval["entries"],
                )[0]
            ),

            "test_sharpe": valid_sharpe(test_eval["sharpe"]),
            "thresholds": str(thresholds),
            "mode": mode,
            "val_buy": valid_stats["buy"],
            "val_hold": valid_stats["hold"],
            "val_sell": valid_stats["sell"],
            "val_entries": valid_eval["entries"],
            "test_buy": test_stats["buy"],
            "test_hold": test_stats["hold"],
            "test_sell": test_stats["sell"],
            "test_entries": test_eval["entries"],
            "epoch_selection_sharpe": epoch_sharpe,
            "epoch_selection_score": epoch_score,
            "seconds": elapsed,
        })

        pd.DataFrame(yearly_rows).to_csv(
            METRIC_DIR / f"{ticker}_yearly.csv",
            index=False,
        )

        val_score, _ = calculate_validation_score(
            valid_eval["sharpe"],
            valid_eval["entries"],
        )

        print(
            f"  Valid Sharpe={valid_sharpe(valid_eval['sharpe']):.3f} | "
            f"Valid Score={val_score:.3f} | "
            f"Test Sharpe={valid_sharpe(test_eval['sharpe']):.3f} | "
            f"Entries={test_eval['entries']} | "
            f"Time={elapsed:.1f}s"
        )

        del dataset, train, valid, test
        del X_train, X_valid, X_test

    # Combine all out-of-sample predictions
    if all_predictions:

        combined_pred = np.concatenate(all_predictions)
        combined_dates = pd.DatetimeIndex(
            pd.to_datetime(all_dates)
        )

        if (
            combined_dates.has_duplicates
            or not combined_dates.is_monotonic_increasing
        ):
            raise ValueError(
                f"{ticker}: invalid combined dates"
            )

        np.save(
            PRED_DIR / f"{ticker}_test_pred.npy",
            combined_pred,
        )

        np.save(
            PRED_DIR / f"{ticker}_test_dates.npy",
            combined_dates.strftime(
                "%Y-%m-%d"
            ).to_numpy(),
        )

    return yearly_rows


# ============================================================
# 6. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--start", type=int, default=START_YEAR)
    parser.add_argument("--end", type=int, default=END_YEAR)

    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start cannot exceed --end")

    set_seed()

    for directory in (MODEL_DIR, PRED_DIR, METRIC_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    tickers = (
        [args.ticker.upper()]
        if args.ticker
        else sorted({
            p.stem.rsplit("_", 1)[0]
            for p in PARAM_DIR.glob("*.json")
            if p.stem.rsplit("_", 1)[-1].isdigit()
        })
    )
    if not tickers:
        raise FileNotFoundError(
            "No Optuna parameter files found"
        )

    print(
        f"Device={DEVICE} | "
        f"Years={args.start}-{args.end} | "
        f"Stocks={len(tickers)}"
    )

    all_rows = []

    for ticker in tickers:
        rows = run_stock(
            ticker,
            start_year=args.start,
            end_year=args.end,
        )

        all_rows.extend(rows)

        if all_rows:
            pd.DataFrame(all_rows).to_csv(
                METRIC_DIR / "walkforward_yearly.csv",
                index=False,
            )

    print(f"\nFinished. Results: {WF_DIR}")


if __name__ == "__main__":
    main()

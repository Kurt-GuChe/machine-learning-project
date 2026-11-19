import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression

from dataset import (
    build_rolling_frame,
    FEATURE_COLS,
    RAW_DIR,
)

from backtest import (
    evaluate_cash_strategy,
    SENTINEL,
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "best_params"

START_YEAR = 2022
END_YEAR = 2026

TRAIN_YEARS = 5

# Search settings
MIN_TRIALS = 100
MAX_TRIALS = 500
BATCH_SIZE = 50

# Early stopping: best trial must have more than 5 entries
EARLY_STOP_ENTRIES = 5

# Exponential entry penalty
TARGET_ENTRIES = 5
PENALTY_LAMBDA = 0.1
PENALTY_K = 0.5

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Parameter search space

def suggest(trial):

    params = {
        "rsi_period": trial.suggest_int("rsi_period", 5, 21),
        "macd_fast": trial.suggest_int("macd_fast", 8, 16),
        "macd_slow": trial.suggest_int("macd_slow", 20, 30),
        "macd_signal": trial.suggest_int("macd_signal", 6, 12),
        "bb_period": trial.suggest_int("bb_period", 15, 30),
        "bb_std": trial.suggest_float("bb_std", 1.5, 3.0),
        "ema10_period": trial.suggest_int("ema10_period", 8, 16),
        "ema20_period": trial.suggest_int("ema20_period", 15, 30),
        "atr_period": trial.suggest_int("atr_period", 10, 20),
    }

    window = trial.suggest_int("barrier_window", 5, 10)
    mult = trial.suggest_float("atr_multiplier", 1.0, 2.5)

    if (
        params["macd_fast"] >= params["macd_slow"]
        or params["ema10_period"] >= params["ema20_period"]
    ):
        raise optuna.TrialPruned()

    return params, window, mult



# Objective

def objective(trial, ticker, test_year):

    params, window, mult = suggest(trial)

    result = build_rolling_frame(
        ticker=ticker,
        params=params,
        window=window,
        mult=mult,
        test_year=test_year,
        train_years=TRAIN_YEARS,
    )

    if result is None:
        trial.set_user_attr("reject_reason", "invalid_dataset")
        raise optuna.TrialPruned()

    train, valid, _, _ = result

    # Check dataset size
    if len(train) < 200 or len(valid) < 50:
        trial.set_user_attr("reject_reason", "insufficient_data")
        raise optuna.TrialPruned()

    # Check number of training classes
    if len(np.unique(train["label"])) < 3:
        trial.set_user_attr("reject_reason", "missing_class")
        raise optuna.TrialPruned()

    # Check training label distribution
    distribution = np.bincount(
        train["label"].to_numpy(dtype=int),
        minlength=3,
    ) / len(train)

    if distribution.max() > 0.75:
        trial.set_user_attr("reject_reason", "label_imbalance")
        raise optuna.TrialPruned()

    x_train = train[FEATURE_COLS].to_numpy(
        dtype=np.float64
    )

    x_valid = valid[FEATURE_COLS].to_numpy(
        dtype=np.float64
    )

    y_train = train["label"].to_numpy(dtype=int)

    # Record training label distribution
    trial.set_user_attr(
        "train_counts",
        np.bincount(y_train, minlength=3).tolist()
    )

    # Logistic Regression proxy model
    model = LogisticRegression(max_iter=500)

    model.fit(x_train, y_train)

    pred = model.predict(x_valid)

    # Record validation prediction distribution
    trial.set_user_attr(
        "pred_counts",
        np.bincount(pred, minlength=3).tolist()
    )

    # Cash-account backtest
    evaluation = evaluate_cash_strategy(
        pred,
        valid["_raw_open"].to_numpy(dtype=np.float64),
        valid.index,
    )

    sharpe = float(evaluation["sharpe"])
    entries = int(evaluation["entries"])

    trial.set_user_attr("entries", entries)

    # Check Sharpe validity before applying penalty
    if not np.isfinite(sharpe) or sharpe <= SENTINEL / 2:
        trial.set_user_attr("reject_reason", "invalid_sharpe")
        return SENTINEL

    # Exponential penalty for insufficient entries
    entry_shortfall = max(0, TARGET_ENTRIES - entries)

    penalty = PENALTY_LAMBDA * np.expm1(
        PENALTY_K * entry_shortfall
    )

    score = sharpe - penalty

    trial.set_user_attr("raw_sharpe", sharpe)
    trial.set_user_attr("penalty", float(penalty))
    trial.set_user_attr("score", float(score))

    return float(score)


# Search one stock and year

def search_one(ticker, test_year, n_trials=MAX_TRIALS):

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Upper bound can be specified from command line
    max_trials = min(n_trials, MAX_TRIALS)

    # Always run at least 100 trials
    max_trials = max(max_trials, MIN_TRIALS)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def run_trials(count):
        study.optimize(
            lambda trial: objective(trial, ticker, test_year),
            n_trials=count,
            show_progress_bar=False,
            catch=(ValueError, FloatingPointError),
        )

    def get_valid_trials():
        return [
            trial
            for trial in study.trials
            if (
                trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
                and np.isfinite(trial.value)
                and trial.value > SENTINEL / 2
            )
        ]
    # early stopping
  

    completed = []

    while len(study.trials) < max_trials:

        remaining = max_trials - len(study.trials)
        count = min(BATCH_SIZE, remaining)

        run_trials(count)

        completed = get_valid_trials()

        # At least 100 trials must be completed
        if len(study.trials) < MIN_TRIALS:
            continue

        if completed:

            best_trial = max(
                completed,
                key=lambda trial: trial.value,
            )

            best_entries = int(
                best_trial.user_attrs.get("entries", 0)
            )
            # Early stopping:
            # best trial has more than 5 real entries
            if best_entries > EARLY_STOP_ENTRIES:
                break
    # Handle no valid trial

    if not completed:

        from collections import Counter

        reasons = Counter(
            t.user_attrs.get("reject_reason", "other")
            for t in study.trials
        )

        print(
            f"{ticker} {test_year}: "
            f"no valid trial after {len(study.trials)} trials"
        )

        print("Failure reasons:", dict(reasons))
        return None

    # Select best trial by penalized score

    best_trial = max(
        completed,
        key=lambda trial: trial.value,
    )

    best = dict(best_trial.params)

    # Raw validation Sharpe
    best["_sharpe"] = round(
        float(best_trial.user_attrs["raw_sharpe"]),
        4,
    )

    # Penalized optimization score
    best["_score"] = round(
        float(best_trial.value),
        4,
    )

    # Real entry count
    best["_entries"] = int(
        best_trial.user_attrs["entries"]
    )

    # Entry penalty
    best["_penalty"] = round(
        float(best_trial.user_attrs["penalty"]),
        4,
    )

    # Actual number of Optuna trials
    best["_trials"] = len(study.trials)

    # Save parameters
    output_path = OUT_DIR / f"{ticker}_{test_year}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    return best


# Rolling search

def search_all(tickers, start_year, end_year, n_trials):

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = []
    start = time.perf_counter()

    total = len(tickers) * (end_year - start_year + 1)
    current = 0

    for ticker in tickers:

        for year in range(start_year, end_year + 1):

            current += 1
            trial_start = time.perf_counter()

            try:

                best = search_one(
                    ticker,
                    year,
                    n_trials=n_trials,
                )

                if best is None:

                    print(
                        f"[{current}/{total}] "
                        f"{ticker} {year}: no valid trial"
                    )

                    continue

                scores.append(best["_sharpe"])

                print(
                    f"[{current}/{total}] {ticker} {year} | "
                    f"Sharpe={best['_sharpe']:+.4f} | "
                    f"Score={best['_score']:+.4f} | "
                    f"Entries={best['_entries']} | "
                    f"Trials={best['_trials']} | "
                    f"Window={best['barrier_window']} | "
                    f"Mult={best['atr_multiplier']:.2f} | "
                    f"Time={time.perf_counter() - trial_start:.1f}s"
                )

            except Exception as exc:

                print(
                    f"[{current}/{total}] "
                    f"{ticker} {year}: {exc}"
                )

    if scores:

        print(
            f"\nValid searches: {len(scores)}/{total} | "
            f"Mean Sharpe: {np.mean(scores):+.4f}"
        )

    else:

        print("No valid parameters found")

    print(
        f"Total time: "
        f"{(time.perf_counter() - start) / 60:.1f} min"
    )


# Main

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=START_YEAR,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=END_YEAR,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=MAX_TRIALS,
    )

    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be <= --end")

    if args.trials < 1:
        parser.error("--trials must be >= 1")

    if args.ticker:

        tickers = [args.ticker.upper()]

    else:

        tickers = sorted(
            path.stem for path in RAW_DIR.glob("*.csv")
        )

    search_all(
        tickers,
        args.start,
        args.end,
        args.trials,
    )


if __name__ == "__main__":
    main()

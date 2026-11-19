
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from indicators import build_features, DEFAULT_PARAMS, FEATURE_COLS
from labels import make_labels

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

SEQ_LEN = 60
TRAIN_YEARS = 5
LABEL_SLIP = 0.0005

#Load market data
@lru_cache(maxsize=64)
def load_raw(ticker):

    path = RAW_DIR / f"{ticker}.csv"

    df = pd.read_csv(
        path,
        index_col="Date",
        parse_dates=True,
    ).sort_index()

    df = df.loc[
        ~df.index.duplicated(keep="last")
    ]

    return df

# Relative features

def make_relative_features(features):

    f = features.copy()

    # Preserve original prices for backtesting
    f["_raw_open"] = f["Open"].astype(np.float64)
    f["_raw_close"] = f["Close"].astype(np.float64)

    close = f["Close"].astype(np.float64)
    prev_close = close.shift(1)

    volume = f["Volume"].astype(np.float64)

    volume_ma20 = volume.rolling(
        window=20,
        min_periods=20,
    ).mean()

    relative = pd.DataFrame(index=f.index)

    # OHLC relative to previous close
    for column in ("Open", "High", "Low", "Close"):
        relative[column] = f[column] / prev_close - 1.0

    # Moving averages
    relative["SMA_50"] = close / f["SMA_50"] - 1.0
    relative["SMA_200"] = close / f["SMA_200"] - 1.0

    relative["EMA_10"] = close / f["EMA_10"] - 1.0
    relative["EMA_20"] = close / f["EMA_20"] - 1.0

    # Bollinger Bands
    relative["BB_upper"] = f["BB_upper"] / close - 1.0
    relative["BB_lower"] = f["BB_lower"] / close - 1.0

    # Volatility and momentum
    relative["ATR"] = f["ATR"] / close
    relative["RSI"] = f["RSI"]

    relative["MACD"] = f["MACD"] / close
    relative["MACD_signal"] = f["MACD_signal"] / close

    # Relative volume
    relative["Volume"] = volume / volume_ma20 - 1.0

    # Ensure consistent feature columns
    if set(relative.columns) != set(FEATURE_COLS):
        raise ValueError("FEATURE_COLS mismatch")

    relative = relative[FEATURE_COLS]

    relative = relative.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    f.loc[:, FEATURE_COLS] = relative

    return f

#Features and labels

def load_one(ticker, params=None, window=5, mult=2.0):
    """
    Build features and labels for one stock.

    Keep label=-1 because test dates must not be removed.
    """

    raw = load_raw(ticker).copy()

    features = build_features(
        raw,
        params if params is not None else DEFAULT_PARAMS,
    )

    # Generate labels using original prices
    features["label"] = make_labels(
        features["Open"],
        features["High"],
        features["Low"],
        features["Close"],
        features["ATR"],
        window,
        mult,
        LABEL_SLIP,
    )

    features = make_relative_features(features)

    return features.dropna(
        subset=FEATURE_COLS
    ).copy()

# Rolling time boundaries

def rolling_dates(test_year, train_years=TRAIN_YEARS):

    train_start = pd.Timestamp(
        test_year - train_years - 1, 1, 1
    )

    valid_start = pd.Timestamp(
        test_year - 1, 1, 1
    )

    test_start = pd.Timestamp(
        test_year, 1, 1
    )

    test_end = pd.Timestamp(
        test_year + 1, 1, 1
    )

    return train_start, valid_start, test_start, test_end
# Rolling DataFrame for Optuna

def build_rolling_frame(
    ticker,
    params,
    window,
    mult,
    test_year,
    train_years=TRAIN_YEARS,
):
    
    frame = load_one(ticker, params, window, mult)

    raw_dates = load_raw(ticker).index
    dates = frame.index

    train_start, valid_start, test_start, test_end = (
        rolling_dates(test_year, train_years)
    )

    # Map feature dates back to raw trading-day positions
    raw_pos = raw_dates.get_indexer(dates)

    # Last available raw trading day of each period
    last_train_raw = raw_dates.searchsorted(valid_start) - 1
    last_valid_raw = raw_dates.searchsorted(test_start) - 1

    valid_label = frame["label"].to_numpy() >= 0

    train_mask = (
        (dates >= train_start)
        & (dates < valid_start)
        & valid_label
        & (raw_pos + window <= last_train_raw)
    )

    valid_mask = (
        (dates >= valid_start)
        & (dates < test_start)
        & valid_label
        & (raw_pos + window <= last_valid_raw)
    )

    test_mask = (
        (dates >= test_start)
        & (dates < test_end)
    )

    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy()
    test = frame.loc[test_mask].copy()

    if (
        len(train) < 200
        or len(valid) < 30
        or len(test) < 20
    ):
        return None

    # Standardization based only on training data
    mu = train[FEATURE_COLS].mean()

    sd = (
        train[FEATURE_COLS]
        .std(ddof=0)
        .replace(0, 1)
        .fillna(1)
    )

    for part in (train, valid, test):
        part.loc[:, FEATURE_COLS] = (
            part[FEATURE_COLS] - mu
        ) / sd

    return train, valid, test, (mu, sd)


#Rolling LSTM sequences


def build_rolling_dataset(
    ticker,
    params,
    window,
    mult,
    test_year,
    train_years=TRAIN_YEARS,
    seq_len=SEQ_LEN,
):


    # Build unscaled full feature history
    frame = load_one(ticker, params, window, mult)

    # Obtain the exact same rolling split and scaler
    result = build_rolling_frame(
        ticker,
        params,
        window,
        mult,
        test_year,
        train_years,
    )

    if result is None:
        return None

    train_frame, valid_frame, test_frame, scaler = result
    mu, sd = scaler

    # Standardize entire history using training statistics
    z = (
        (frame[FEATURE_COLS] - mu) / sd
    ).to_numpy(dtype=np.float32)

    y = frame["label"].to_numpy(dtype=np.int64)

    opens = frame["_raw_open"].to_numpy(dtype=np.float64)
    closes = frame["_raw_close"].to_numpy(dtype=np.float64)

    dates = frame.index

    if len(frame) < seq_len:
        return None

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(
        z,
        seq_len,
        axis=0,
    )

    def make_part(part_frame):

        # Positions in the complete feature history
        idx = dates.get_indexer(part_frame.index)

        # Need enough historical observations
        idx = idx[idx >= seq_len - 1]

        if len(idx) == 0:
            return None

        X = np.ascontiguousarray(
            windows[idx - seq_len + 1].transpose(0, 2, 1)
        )

        return (
            X,
            y[idx],
            opens[idx],
            closes[idx],
            dates[idx],
        )

    train = make_part(train_frame)
    valid = make_part(valid_frame)
    test = make_part(test_frame)

    if any(part is None for part in (train, valid, test)):
        return None

    return train, valid, test, scaler

# test

if __name__ == "__main__":

    result = build_rolling_dataset(
        ticker="AAPL",
        params=DEFAULT_PARAMS,
        window=5,
        mult=2.0,
        test_year=2022,
    )

    if result is None:
        print("Insufficient data")
    else:
        train, valid, test, _ = result

        for name, part in (
            ("Train", train),
            ("Valid", valid),
            ("Test", test),
        ):
            X, y, _, _, dates = part

            print(
                f"{name}: "
                f"X={X.shape}, "
                f"{dates[0].date()} -> {dates[-1].date()}"
            )

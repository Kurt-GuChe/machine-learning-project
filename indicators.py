import numpy as np
import pandas as pd

# LSTM 15 inputs
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",                       # 基础行情 5
    "RSI", "MACD", "MACD_signal", "BB_upper", "BB_lower",
    "EMA_10", "EMA_20", "ATR",                                       # 可优化 8
    "SMA_50", "SMA_200",                                             # 固定 2
]

# Default parameters
DEFAULT_PARAMS = {
    "rsi_period":   14,                                              # [5, 21]
    "macd_fast":    12, "macd_slow": 26, "macd_signal": 9,           # [8,16]/[20,30]/[6,12]
    "bb_period":    20, "bb_std":    2.0,                            # [15,30]/[1.5,3.0]
    "ema10_period": 10, "ema20_period": 20,                          # [8,16]/[15,30]
    "atr_period":   14,                                              # [10,20]
}


def ema(s, p):
    # Exponential Moving Average. 
    return s.ewm(span=p, adjust=False).mean()


def rsi(close, p):
    # RSI
    d = close.diff()                                                 # 相邻日价差
    up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()       # 上涨均值
    dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()    # 下跌均值(取正)
    rs = up / dn.replace(0, np.nan)                                  # 避免除 0
    return 100 - 100 / (1 + rs)                                      # 映射到 [0, 100]


def macd(close, fast, slow, sig):
    # MACD
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, sig)


def bollinger(close, p, k):
    # Bollin *2
    ma = close.rolling(p).mean()
    sd = close.rolling(p).std()                                      # 样本标准差 ddof=1
    return ma + k * sd, ma - k * sd


def atr(high, low, close, p):
    # Average True Range
    pc = close.shift(1)                                              # 前一日收盘
    tr = pd.concat([high - low,                                     # 当日振幅
                    (high - pc).abs(),                              # 向上跳空
                    (low - pc).abs()], axis=1).max(axis=1)          # 向下跳空
    return tr.ewm(alpha=1 / p, adjust=False).mean()                  # Wilder 平滑


def build_features(df, p):
    # 5 base date
    out = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()

    # 8 Variable indicators
    out["RSI"] = rsi(df["Close"], p["rsi_period"])
    out["MACD"], out["MACD_signal"] = macd(
        df["Close"], p["macd_fast"], p["macd_slow"], p["macd_signal"])
    out["BB_upper"], out["BB_lower"] = bollinger(
        df["Close"], p["bb_period"], p["bb_std"])
    out["EMA_10"] = ema(df["Close"], p["ema10_period"])
    out["EMA_20"] = ema(df["Close"], p["ema20_period"])
    out["ATR"] = atr(df["High"], df["Low"], df["Close"], p["atr_period"])

    # 2 indicators
    out["SMA_50"] = df["Close"].rolling(50).mean()
    out["SMA_200"] = df["Close"].rolling(200).mean()

    return out[FEATURE_COLS]


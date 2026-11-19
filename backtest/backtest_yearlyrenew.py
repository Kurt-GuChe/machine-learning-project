
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import RAW_DIR
from backtest import simulate_cash_account, INITIAL_CAPITAL, SLIP, FEE


BASE = Path(__file__).resolve().parent
PRED_DIR = BASE / "result_yearly_renew" / "predictions"
OUT = BASE / "result_yearly_renew" / "backtest_yearly"

START_YEAR = 2022
END_YEAR = 2026

for folder in ["accounts", "trades", "signals", "figures"]:
    (OUT / folder).mkdir(parents=True, exist_ok=True)


def metrics(returns):
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return dict(ret=np.nan, sharpe=np.nan, max_dd=np.nan)

    curve = np.r_[1.0, np.cumprod(1 + r)]
    peaks = np.maximum.accumulate(curve)
    std = np.std(r, ddof=1) if len(r) > 1 else 0

    return dict(
        ret=float(curve[-1] - 1),
        sharpe=float(np.mean(r) / std * np.sqrt(252))
        if len(r) >= 20 and std > 1e-12 else np.nan,
        max_dd=float(np.min(curve / peaks - 1)),
    )


def run_stock(ticker):
    pred = np.load(PRED_DIR / f"{ticker}_test_pred.npy").astype(int).reshape(-1)
    dates = pd.DatetimeIndex(pd.to_datetime(
        np.load(PRED_DIR / f"{ticker}_test_dates.npy", allow_pickle=True).astype(str)
    ))

    if len(pred) != len(dates):
        raise ValueError("Prediction/date length mismatch")

    order = np.argsort(dates.values)
    pred, dates = pred[order], dates[order]

    mask = (dates.year >= START_YEAR) & (dates.year <= END_YEAR)
    pred, dates = pred[mask], dates[mask]

    if len(dates) < 2 or dates.has_duplicates:
        raise ValueError("Invalid test dates")

    prices = pd.read_csv(
        RAW_DIR / f"{ticker}.csv", index_col="Date", parse_dates=True
    ).sort_index().reindex(dates)["Open"]

    if prices.isna().any():
        raise ValueError("Missing open prices")

    open_prices = prices.to_numpy(dtype=float)

    daily, trades = simulate_cash_account(pred, open_prices, dates)

    bh_pred = np.ones(len(pred), dtype=int)
    bh_pred[0] = 0
    bh_daily, _ = simulate_cash_account(bh_pred, open_prices, dates)

    daily["bh_equity"] = bh_daily["equity"]
    daily["bh_return"] = bh_daily["return"]
    daily["year"] = daily.index.year

    daily.to_csv(OUT / "accounts" / f"{ticker}_daily.csv")
    trades.to_csv(OUT / "trades" / f"{ticker}_trades.csv", index=False)

    signals = pd.DataFrame({
        "date": dates,
        "prediction": pred,
        "signal": np.array(["Buy", "Hold", "Sell"])[pred],
        "open_price": open_prices,
    })
    signals.to_csv(OUT / "signals" / f"{ticker}_signals.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily["equity"] / INITIAL_CAPITAL, label="LSTM")
    ax.plot(daily.index, daily["bh_equity"] / INITIAL_CAPITAL,
            label="Buy & Hold", linestyle="--")
    ax.set_title(f"{ticker} Equity Curve")
    ax.set_ylabel("Equity / Initial Capital")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{ticker}_equity.png", dpi=120)
    plt.close(fig)

    trade_year = pd.to_datetime(trades["date"]).dt.year
    is_entry = trades["reason"].eq("ENTRY")

    yearly = []
    for year, group in daily.groupby("year"):
        m = metrics(group["return"])
        bh = metrics(group["bh_return"])

        yearly.append({
            "ticker": ticker,
            "year": int(year),
            "days": len(group),
            "return": m["ret"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
            "bh_return": bh["ret"],
            "bh_sharpe": bh["sharpe"],
            "bh_max_dd": bh["max_dd"],
            "entries": int((is_entry & trade_year.eq(year)).sum()),
            "orders": int(trade_year.eq(year).sum()),
            "exposure": float((group["shares"] > 1e-10).mean()),
            "end_equity": float(group["equity"].iloc[-1]),
        })

    m = metrics(daily["return"].to_numpy()[1:])
    bh = metrics(daily["bh_return"].to_numpy()[1:])

    total = {
        "ticker": ticker,
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "return": m["ret"],
        "sharpe": m["sharpe"],
        "max_dd": m["max_dd"],
        "bh_return": bh["ret"],
        "bh_sharpe": bh["sharpe"],
        "bh_max_dd": bh["max_dd"],
        "entries": int(is_entry.sum()),
        "orders": len(trades),
        "final_equity": float(daily["equity"].iloc[-1]),
    }

    return yearly, total, daily[["return", "bh_return"]]


def main():
    tickers = sorted(
        p.name.removesuffix("_test_pred.npy")
        for p in PRED_DIR.glob("*_test_pred.npy")
    )

    yearly_rows, total_rows, returns = [], [], {}

    for ticker in tickers:
        try:
            yearly, total, daily_ret = run_stock(ticker)
            yearly_rows.extend(yearly)
            total_rows.append(total)
            returns[ticker] = daily_ret
            print(f"{ticker}: completed")
        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"{ticker}: {e}")

    if not total_rows:
        print("No successful backtests")
        return

    stock_yearly = pd.DataFrame(yearly_rows).sort_values(["year", "ticker"])
    stock_total = pd.DataFrame(total_rows).sort_values("ticker")

    stock_yearly.to_csv(OUT / "stock_yearly.csv", index=False)
    stock_total.to_csv(OUT / "stock_total.csv", index=False)

    summary = stock_yearly.groupby("year").agg(
        stocks=("ticker", "count"),
        mean_return=("return", "mean"),
        mean_sharpe=("sharpe", "mean"),
        mean_max_dd=("max_dd", "mean"),
        mean_bh_return=("bh_return", "mean"),
        total_entries=("entries", "sum"),
        positive_stocks=("return", lambda x: int((x > 0).sum())),
    ).reset_index()

    summary["beat_bh_stocks"] = stock_yearly.groupby("year").apply(
        lambda x: int((x["return"] > x["bh_return"]).sum()),
        include_groups=False,
    ).to_numpy()

    summary.to_csv(OUT / "year_summary.csv", index=False)

    strategy = pd.DataFrame({
        ticker: df["return"] for ticker, df in returns.items()
    }).sort_index()

    benchmark = pd.DataFrame({
        ticker: df["bh_return"] for ticker, df in returns.items()
    }).sort_index()

    portfolio = pd.DataFrame(index=strategy.index)
    portfolio["return"] = strategy.mean(axis=1)
    portfolio["bh_return"] = benchmark.mean(axis=1)
    portfolio["n_stocks"] = strategy.notna().sum(axis=1)

    portfolio["equity"] = INITIAL_CAPITAL * (1 + portfolio["return"]).cumprod()
    portfolio["bh_equity"] = INITIAL_CAPITAL * (1 + portfolio["bh_return"]).cumprod()
    portfolio.index.name = "date"

    portfolio.to_csv(OUT / "portfolio_daily.csv")

    portfolio_yearly = []
    for year, group in portfolio.groupby(portfolio.index.year):
        m = metrics(group["return"])
        bh = metrics(group["bh_return"])
        portfolio_yearly.append({
            "year": int(year),
            "return": m["ret"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
            "bh_return": bh["ret"],
            "bh_sharpe": bh["sharpe"],
            "bh_max_dd": bh["max_dd"],
            "end_equity": float(group["equity"].iloc[-1]),
        })

    pd.DataFrame(portfolio_yearly).to_csv(
        OUT / "portfolio_yearly.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(portfolio.index, portfolio["equity"] / INITIAL_CAPITAL, label="LSTM")
    ax.plot(portfolio.index, portfolio["bh_equity"] / INITIAL_CAPITAL,
            label="Buy & Hold", linestyle="--")
    ax.set_title("Equal-Weight Portfolio")
    ax.set_ylabel("Equity / Initial Capital")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "portfolio_equity.png", dpi=120)
    plt.close(fig)

    config = {
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "initial_capital": INITIAL_CAPITAL,
        "slippage": SLIP,
        "fee_per_share": FEE,
        "market_regime": False,
        "stop_loss": False,
        "execution": "next_day_open",
        "stocks_completed": len(total_rows),
        "portfolio_method": "daily_equal_weight_average",
    }

    with open(OUT / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n", summary.to_string(index=False))
    print(f"\nCompleted: {len(total_rows)} stocks")
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()

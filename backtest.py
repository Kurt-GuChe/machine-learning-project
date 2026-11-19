
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import RAW_DIR


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WF_DIR = BASE_DIR / "walkforward_relative_results"
PRED_DIR = WF_DIR / "predictions"

OUT_DIR = WF_DIR / "backtest_no_regime"
FIG_DIR = OUT_DIR / "figures"
ACCOUNT_DIR = OUT_DIR / "accounts"
TRADE_DIR = OUT_DIR / "trades"

INITIAL_CAPITAL = 100_000.0
SLIP = 0.0005
FEE = 0.005

SENTINEL = -1e7
MIN_RETURN_POINTS = 20


# ============================================================
# 1. Cash account
# ============================================================

def simulate_cash_account(
    pred,
    open_,
    dates,
    initial_cash=INITIAL_CAPITAL,
):
    """
    Signal at close[t], execute at open[t+1].

    0 = Buy  : invest all available cash
    1 = Hold : keep current position
    2 = Sell : liquidate all shares

    Long-only, no leverage, no market regime.
    Fractional shares allowed.
    """

    pred = np.asarray(pred, dtype=int).reshape(-1)
    prices = np.asarray(open_, dtype=float).reshape(-1)
    dates = pd.DatetimeIndex(dates)

    n = len(pred)

    if not (n == len(prices) == len(dates)):
        raise ValueError("Input lengths differ")

    if n < 2:
        raise ValueError("At least two trading days required")

    if not np.isin(pred, [0, 1, 2]).all():
        raise ValueError("Invalid prediction labels")

    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Invalid open prices")

    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Dates must be strictly increasing")

    cash = float(initial_cash)
    shares = 0.0

    daily = []
    trades = []

    for i, price in enumerate(prices):

        # Execute yesterday's signal at today's open
        if i > 0:

            signal = int(pred[i - 1])
            signal_date = dates[i - 1]

            action = None
            quantity = 0.0

            if signal == 0 and shares <= 1e-10:
                action = "BUY"

                execution_price = price * (1 + SLIP)
                quantity = cash / (execution_price + FEE)

            elif signal == 2 and shares > 1e-10:
                action = "SELL"

                execution_price = price * (1 - SLIP)
                quantity = shares

            if action is not None and quantity > 1e-10:

                commission = quantity * FEE

                if action == "BUY":
                    cash -= quantity * execution_price + commission
                    shares += quantity
                    reason = "ENTRY"

                else:
                    cash += quantity * execution_price - commission
                    shares -= quantity
                    reason = "EXIT"

                if abs(cash) < 1e-8:
                    cash = 0.0

                if abs(shares) < 1e-10:
                    shares = 0.0

                trades.append({
                    "date": dates[i],
                    "signal_date": signal_date,
                    "action": action,
                    "reason": reason,
                    "quantity": quantity,
                    "market_price": price,
                    "execution_price": execution_price,
                    "commission": commission,
                    "slippage_cost": quantity * price * SLIP,
                    "cash_after": cash,
                    "shares_after": shares,
                    "equity_after": cash + shares * price,
                })

        equity = cash + shares * price

        daily.append({
            "date": dates[i],
            "cash": cash,
            "shares": shares,
            "open_price": price,
            "stock_value": shares * price,
            "equity": equity,
            "actual_weight": (
                shares * price / equity if equity > 0 else 0.0
            ),
        })

    daily = pd.DataFrame(daily).set_index("date")
    daily["return"] = daily["equity"].pct_change().fillna(0.0)

    trade_columns = [
        "date", "signal_date", "action", "reason",
        "quantity", "market_price", "execution_price",
        "commission", "slippage_cost", "cash_after",
        "shares_after", "equity_after",
    ]

    trades = pd.DataFrame(trades, columns=trade_columns)

    return daily, trades


# ============================================================
# 2. Performance metrics
# ============================================================

def calculate_metrics(daily):
    """
    Calculate performance from cash-account equity.
    """

    equity = daily["equity"].to_numpy(dtype=float)
    returns = daily["return"].to_numpy(dtype=float)[1:]

    total_ret = equity[-1] / equity[0] - 1

    years = len(returns) / 252

    cagr = (
        (equity[-1] / equity[0]) ** (1 / years) - 1
        if years > 0 and equity[-1] > 0
        else np.nan
    )

    ann_ret = float(np.mean(returns) * 252)

    volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(252))
        if len(returns) > 1
        else 0.0
    )

    sharpe = (
        float(np.mean(returns) / np.std(returns, ddof=1)
              * np.sqrt(252))
        if len(returns) >= MIN_RETURN_POINTS
        and np.std(returns, ddof=1) > 1e-12
        else np.nan
    )

    peaks = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peaks - 1))

    return {
        "ann_ret": ann_ret,
        "cagr": float(cagr),
        "volatility": volatility,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_ret": float(total_ret),
        "final_equity": float(equity[-1]),
    }


# ============================================================
# 3. Validation evaluation
# ============================================================

def evaluate_cash_strategy(pred, open_, dates):
    """
    Used by walkforward.py for checkpoint
    and threshold selection.
    """

    daily, trades = simulate_cash_account(
        pred, open_, dates
    )

    metrics = calculate_metrics(daily)

    entries = int(
        (trades["reason"] == "ENTRY").sum()
    )

    exposure = float(
        (daily["shares"].to_numpy()[1:] > 1e-10).mean()
    )

    sharpe = metrics["sharpe"]

    return {
        "sharpe": (
            sharpe if np.isfinite(sharpe) else SENTINEL
        ),
        "entries": entries,
        "exposure": exposure,
        "final_equity": metrics["final_equity"],
    }


# ============================================================
# 4. Single-stock backtest
# ============================================================

def backtest(ticker):

    data_path = RAW_DIR / f"{ticker}.csv"
    pred_path = PRED_DIR / f"{ticker}_test_pred.npy"
    date_path = PRED_DIR / f"{ticker}_test_dates.npy"

    df = pd.read_csv(
        data_path,
        index_col="Date",
        parse_dates=True,
    ).sort_index()

    pred = np.load(pred_path).astype(int).reshape(-1)

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            np.load(date_path, allow_pickle=True).astype(str)
        )
    )

    if len(pred) != len(dates):
        raise ValueError(f"{ticker}: prediction/date mismatch")

    order = np.argsort(dates.values)
    dates = dates[order]
    pred = pred[order]

    if dates.has_duplicates:
        raise ValueError(f"{ticker}: duplicate test dates")

    if len(pred) < MIN_RETURN_POINTS + 1:
        return None

    market = df.reindex(dates)

    if market["Open"].isna().any():
        raise ValueError(f"{ticker}: missing open prices")

    open_ = market["Open"].to_numpy(dtype=float)

    # LSTM strategy
    daily, trades = simulate_cash_account(
        pred, open_, dates
    )

    # Buy and Hold: first signal Buy, then Hold
    bh_pred = np.ones(len(pred), dtype=int)
    bh_pred[0] = 0

    bh_daily, _ = simulate_cash_account(
        bh_pred, open_, dates
    )

    metrics = calculate_metrics(daily)
    bh_metrics = calculate_metrics(bh_daily)

    daily.to_csv(
        ACCOUNT_DIR / f"{ticker}_daily.csv"
    )

    trades.to_csv(
        TRADE_DIR / f"{ticker}_trades.csv",
        index=False,
    )

    return {
        "ticker": ticker,
        "dates": dates,
        "curve": daily["equity"].to_numpy() / INITIAL_CAPITAL,
        "bh_curve": bh_daily["equity"].to_numpy() / INITIAL_CAPITAL,
        **metrics,
        "bh_total_ret": bh_metrics["total_ret"],
        "bh_cagr": bh_metrics["cagr"],
        "bh_sharpe": bh_metrics["sharpe"],
        "bh_max_dd": bh_metrics["max_dd"],
        "n_entries": int(
            (trades["reason"] == "ENTRY").sum()
        ),
        "n_orders": len(trades),
    }


# ============================================================
# 5. Single-stock plot
# ============================================================

def plot_single(result):

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        result["dates"],
        result["curve"],
        label="LSTM",
        linewidth=1.5,
    )

    ax.plot(
        result["dates"],
        result["bh_curve"],
        label="Buy and Hold",
        linestyle="--",
        linewidth=1.2,
    )

    ax.axhline(1.0, color="gray", linewidth=0.6)

    ax.set_title(f"{result['ticker']} Backtest")
    ax.set_ylabel("Equity / Initial Capital")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / f"{result['ticker']}_backtest.png",
        dpi=120,
    )
    plt.close(fig)


# ============================================================
# 6. Sharpe ranking
# ============================================================

def plot_sharpe(report):

    data = report.sort_values(
        "sharpe", ascending=False, na_position="last"
    )

    colors = [
        "green" if x > 0.7 else
        "orange" if x > 0 else
        "red" if np.isfinite(x) else "gray"
        for x in data["sharpe"]
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        data["ticker"],
        data["sharpe"].fillna(0),
        color=colors,
    )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Stock")

    plt.xticks(rotation=90)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "sharpe_by_stock.png",
        dpi=120,
    )
    plt.close(fig)


# ============================================================
# 7. Run all stocks
# ============================================================

def run_all():

    for path in (OUT_DIR, FIG_DIR, ACCOUNT_DIR, TRADE_DIR):
        path.mkdir(parents=True, exist_ok=True)

    tickers = sorted(
        p.name.removesuffix("_test_pred.npy")
        for p in PRED_DIR.glob("*_test_pred.npy")
    )

    if not tickers:
        print("No prediction files found")
        return

    rows = []

    for ticker in tickers:

        try:
            result = backtest(ticker)

            if result is None:
                continue

            plot_single(result)

            rows.append({
                k: v
                for k, v in result.items()
                if k not in ("dates", "curve", "bh_curve")
            })

            sharpe = result["sharpe"]

            print(
                f"{ticker:<6} "
                f"Return={result['total_ret']:+.2%} "
                f"BH={result['bh_total_ret']:+.2%} "
                f"Sharpe={sharpe:+.3f} "
                f"Entries={result['n_entries']}"
            )

        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"{ticker}: {exc}")

    if not rows:
        print("No successful backtests")
        return

    report = pd.DataFrame(rows)

    report.to_csv(
        OUT_DIR / "backtest_results.csv",
        index=False,
    )

    plot_sharpe(report)

    print(f"\nCompleted: {len(report)} stocks")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    run_all()

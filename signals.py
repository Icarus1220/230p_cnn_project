import pandas as pd
import ta


def MA(data: pd.DataFrame, length: int) -> pd.DataFrame:
    data[f"{length}_MA"] = ta.trend.sma_indicator(
        data["price"], window=length
    )
    data[f"{length}_EMA"] = ta.trend.ema_indicator(
        data["price"], window=length
    )


def simple_signal(data: pd.DataFrame) -> pd.DataFrame:

    for length in [5, 10, 20]:
        MA(data, length)
    data["MACD"] = ta.trend.macd(data["price"])
    data["MACD_diff"] = ta.trend.macd_diff(data["price"])
    data["MACD_signal"] = ta.trend.macd_signal(
        data["price"]
    )
    data["EMA"] = ta.trend.ema_indicator(
        data["price"], window=20
    )
    data["dema"] = 2 * data["EMA"] - ta.trend.ema_indicator(
        data["EMA"], window=20
    )

    # Hull Moving Average (HMA)
    data["HMA"] = ta.trend.wma_indicator(
        data["price"], window=9
    )

    # Weighted Moving Average (WMA)
    data["WMA"] = ta.trend.wma_indicator(
        data["price"], window=20
    )

    # Percentage Price Oscillator (PPO)
    data["PPO"] = ta.momentum.ppo(
        data["price"], window_slow=26, window_fast=12
    )

    # Detrended Price Oscillator (DPO)
    data["DPO"] = ta.trend.dpo(data["price"], window=20)

    data["RSI"] = ta.momentum.rsi(data["price"], window=14)
    data["ROC"] = ta.momentum.roc(data["price"], window=12)

    # Price Rate of Change (PROC)
    data["PROC"] = ta.momentum.roc(data["price"], window=12)

    return data

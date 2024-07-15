import pandas as pd
import ta


def MA(data: pd.DataFrame, length: int) -> pd.DataFrame:
    data[f"{length}_MA"] = ta.trend.sma_indicator(
        data["price"], window=length
    )
    data[f"{length}_EMA"] = ta.trend.ema_indicator(
        data["Close"], window=length
    )
    return data


def MACD(data: pd.DataFrame) -> pd.DataFrame:
    data["MACD"] = ta.trend.macd(data["price"])
    data["MACD_diff"] = ta.trend.macd_diff(data["price"])
    data["MACD_signal"] = ta.trend.macd_signal(
        data["price"]
    )
    return data


def RSI(data: pd.DataFrame, length: int) -> pd.DataFrame:
    data["RSI"] = ta.momentum.rsi(data["price"], window=14)
    return data


def BB(data: pd.DataFrame) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(
        data["price"], window=20, window_dev=2
    )
    data["BB_High"] = bb.bollinger_hband()
    data["BB_Low"] = bb.bollinger_lband()
    return data


def ROC(data: pd.DataFrame) -> pd.DataFrame:
    data["ROC"] = ta.momentum.roc(data["Close"], window=12)
    return data

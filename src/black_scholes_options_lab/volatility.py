"""Historical and rolling volatility calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """Return log returns from an adjusted close price series."""

    clean = prices.dropna().astype(float)
    return np.log(clean / clean.shift(1)).dropna()


def historical_volatility(
    prices: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized historical volatility from daily closing prices."""

    returns = log_returns(prices)
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(trading_days))


def rolling_volatility(
    prices: pd.Series,
    window: int = 30,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualized rolling volatility from daily closing prices."""

    returns = log_returns(prices)
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)

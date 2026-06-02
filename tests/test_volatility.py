import numpy as np
import pandas as pd
import pytest

from black_scholes_options_lab.volatility import (
    historical_volatility,
    rolling_volatility,
)


def test_historical_volatility_matches_manual_calculation():
    prices = pd.Series([100, 102, 101, 105, 107, 104], dtype=float)
    returns = np.log(prices / prices.shift(1)).dropna()
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert historical_volatility(prices) == pytest.approx(expected)


def test_historical_volatility_with_too_few_prices():
    assert historical_volatility(pd.Series([100.0])) == 0


def test_historical_volatility_drops_missing_prices():
    prices = pd.Series([100, None, 101, 103, None, 102], dtype=float)
    compact = pd.Series([100, 101, 103, 102], dtype=float)
    assert historical_volatility(prices) == pytest.approx(
        historical_volatility(compact)
    )


def test_rolling_volatility_length_and_last_value():
    prices = pd.Series([100, 101, 103, 102, 105, 106, 108], dtype=float)
    result = rolling_volatility(prices, window=3)
    assert len(result) == 6
    assert result.dropna().iloc[-1] > 0

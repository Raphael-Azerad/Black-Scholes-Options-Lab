import numpy as np
import pytest

from black_scholes_options_lab.strategies import payoff_for_strategy


def test_long_call_break_even_matches_strike_plus_premium():
    grid = np.linspace(50, 150, 101)
    payoff = payoff_for_strategy("Long Call", grid, 100, 100, 1, 0.05, 0.20)
    assert payoff.max_loss > 0
    assert payoff.break_evens[0] == pytest.approx(100 + payoff.max_loss, abs=0.01)


def test_short_put_has_defined_max_profit():
    grid = np.linspace(50, 150, 101)
    payoff = payoff_for_strategy("Short Put", grid, 100, 100, 1, 0.05, 0.20)
    assert payoff.max_profit > 0
    assert payoff.max_loss > 0


def test_straddle_has_two_break_evens():
    grid = np.linspace(50, 150, 101)
    payoff = payoff_for_strategy("Straddle", grid, 100, 100, 1, 0.05, 0.20)
    assert len(payoff.break_evens) == 2
    assert payoff.break_evens[0] < 100 < payoff.break_evens[1]

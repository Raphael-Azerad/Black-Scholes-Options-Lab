import math

import pytest

from black_scholes_options_lab.exceptions import InvalidOptionInputError
from black_scholes_options_lab.pricing import (
    black_scholes_price,
    classify_moneyness,
    d1_d2,
    intrinsic_value,
    option_summary,
)


def test_black_scholes_known_call_value():
    price = black_scholes_price(100, 100, 1, 0.05, 0.20, "call")
    assert price == pytest.approx(10.4506, abs=1e-4)


def test_black_scholes_known_put_value():
    price = black_scholes_price(100, 100, 1, 0.05, 0.20, "put")
    assert price == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_without_dividends():
    spot = 100
    strike = 105
    time_to_expiry = 0.75
    rate = 0.04
    volatility = 0.25
    call = black_scholes_price(spot, strike, time_to_expiry, rate, volatility, "call")
    put = black_scholes_price(spot, strike, time_to_expiry, rate, volatility, "put")
    assert call - put == pytest.approx(
        spot - strike * math.exp(-rate * time_to_expiry),
        abs=1e-8,
    )


def test_put_call_parity_with_dividends():
    spot = 100
    strike = 105
    time_to_expiry = 0.75
    rate = 0.04
    dividend_yield = 0.015
    volatility = 0.25
    call = black_scholes_price(
        spot, strike, time_to_expiry, rate, volatility, "call", dividend_yield
    )
    put = black_scholes_price(
        spot, strike, time_to_expiry, rate, volatility, "put", dividend_yield
    )
    assert call - put == pytest.approx(
        spot * math.exp(-dividend_yield * time_to_expiry)
        - strike * math.exp(-rate * time_to_expiry),
        abs=1e-8,
    )


def test_d1_d2_reference_values_with_dividend_yield():
    d1, d2 = d1_d2(100, 95, 0.5, 0.04, 0.30, dividend_yield=0.02)
    assert d1 == pytest.approx(0.3950, abs=1e-4)
    assert d2 == pytest.approx(0.1829, abs=1e-4)


def test_zero_time_returns_intrinsic_value():
    assert black_scholes_price(112, 100, 0, 0.05, 0.20, "call") == 12
    assert black_scholes_price(88, 100, 0, 0.05, 0.20, "put") == 12


def test_zero_volatility_returns_discounted_forward_intrinsic():
    call = black_scholes_price(100, 95, 0.5, 0.04, 0, "call", dividend_yield=0.01)
    forward = 100 * math.exp((0.04 - 0.01) * 0.5)
    expected = max(forward - 95, 0) * math.exp(-0.04 * 0.5)
    assert call == pytest.approx(expected)


def test_deep_in_and_out_of_the_money_prices_are_reasonable():
    deep_itm_call = black_scholes_price(150, 100, 0.5, 0.03, 0.20, "call")
    deep_otm_call = black_scholes_price(50, 100, 0.5, 0.03, 0.20, "call")
    assert deep_itm_call > 50
    assert deep_otm_call < 0.01


def test_intrinsic_value_for_call_and_put():
    assert intrinsic_value(110, 100, "call") == 10
    assert intrinsic_value(90, 100, "put") == 10


def test_option_summary_break_even_and_time_value():
    summary = option_summary(100, 100, 1, 0.05, 0.20, "call")
    assert summary.break_even == pytest.approx(110.4506, abs=1e-4)
    assert summary.time_value == pytest.approx(summary.selected_price)


def test_moneyness_classification_for_calls_and_puts():
    assert classify_moneyness(100, 100, "call") == "At the money"
    assert classify_moneyness(110, 100, "call") == "In the money"
    assert classify_moneyness(90, 100, "call") == "Out of the money"
    assert classify_moneyness(90, 100, "put") == "In the money"
    assert classify_moneyness(110, 100, "put") == "Out of the money"


@pytest.mark.parametrize(
    ("spot", "strike", "time_to_expiry", "volatility"),
    [
        (0, 100, 1, 0.2),
        (100, 0, 1, 0.2),
        (100, 100, -0.1, 0.2),
        (100, 100, 1, -0.2),
    ],
)
def test_invalid_pricing_inputs_raise(spot, strike, time_to_expiry, volatility):
    with pytest.raises(InvalidOptionInputError):
        black_scholes_price(spot, strike, time_to_expiry, 0.05, volatility, "call")


def test_invalid_option_type_raises():
    with pytest.raises(InvalidOptionInputError):
        black_scholes_price(100, 100, 1, 0.05, 0.20, "straddle")

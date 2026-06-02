import pytest

from black_scholes_options_lab.exceptions import InvalidOptionInputError
from black_scholes_options_lab.greeks import calculate_greeks


def test_known_call_greeks():
    greeks = calculate_greeks(100, 100, 1, 0.05, 0.20, "call")
    assert greeks.delta == pytest.approx(0.6368, abs=1e-4)
    assert greeks.gamma == pytest.approx(0.0188, abs=1e-4)
    assert greeks.theta == pytest.approx(-6.4140, abs=1e-4)
    assert greeks.vega == pytest.approx(37.5240, abs=1e-4)
    assert greeks.rho == pytest.approx(53.2325, abs=1e-4)


def test_known_put_greeks():
    greeks = calculate_greeks(100, 100, 1, 0.05, 0.20, "put")
    assert greeks.delta == pytest.approx(-0.3632, abs=1e-4)
    assert greeks.gamma == pytest.approx(0.0188, abs=1e-4)
    assert greeks.theta == pytest.approx(-1.6579, abs=1e-4)
    assert greeks.vega == pytest.approx(37.5240, abs=1e-4)
    assert greeks.rho == pytest.approx(-41.8905, abs=1e-4)


def test_call_and_put_greeks_have_expected_relationships():
    call = calculate_greeks(100, 100, 1, 0.05, 0.20, "call")
    put = calculate_greeks(100, 100, 1, 0.05, 0.20, "put")
    assert call.delta > 0
    assert put.delta < 0
    assert call.gamma == pytest.approx(put.gamma)
    assert call.vega == pytest.approx(put.vega)
    assert call.rho > 0
    assert put.rho < 0


def test_greek_display_units():
    greeks = calculate_greeks(100, 100, 1, 0.05, 0.20, "call")
    assert greeks.theta_per_day == pytest.approx(greeks.theta / 365)
    assert greeks.vega_per_percent == pytest.approx(greeks.vega / 100)
    assert greeks.rho_per_percent == pytest.approx(greeks.rho / 100)


def test_dividend_adjusted_delta_is_lower_for_calls():
    no_dividend = calculate_greeks(100, 100, 1, 0.05, 0.20, "call")
    dividend = calculate_greeks(100, 100, 1, 0.05, 0.20, "call", dividend_yield=0.03)
    assert dividend.delta < no_dividend.delta


def test_zero_time_greeks_are_zero():
    greeks = calculate_greeks(100, 100, 0, 0.05, 0.20, "call")
    assert greeks.delta == 0
    assert greeks.gamma == 0
    assert greeks.theta == 0
    assert greeks.vega == 0
    assert greeks.rho == 0


def test_invalid_greek_option_type_raises():
    with pytest.raises(InvalidOptionInputError):
        calculate_greeks(100, 100, 1, 0.05, 0.20, "butterfly")

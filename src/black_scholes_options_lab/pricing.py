"""Black-Scholes-Merton option pricing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Literal

from scipy.stats import norm

from black_scholes_options_lab.exceptions import InvalidOptionInputError

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionSummary:
    """Compact option pricing summary used by the dashboard and README."""

    call_price: float
    put_price: float
    selected_price: float
    intrinsic_value: float
    time_value: float
    moneyness: float
    moneyness_label: str
    break_even: float


def _validate_inputs(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
) -> None:
    if spot <= 0:
        raise InvalidOptionInputError("Spot price must be positive.")
    if strike <= 0:
        raise InvalidOptionInputError("Strike price must be positive.")
    if time_to_expiry < 0:
        raise InvalidOptionInputError("Time to expiry cannot be negative.")
    if volatility < 0:
        raise InvalidOptionInputError("Volatility cannot be negative.")


def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return Black-Scholes-Merton d1 and d2."""

    _validate_inputs(spot, strike, time_to_expiry, volatility)
    if time_to_expiry == 0 or volatility == 0:
        raise InvalidOptionInputError("d1 and d2 require positive time and volatility.")

    sigma_sqrt_t = volatility * sqrt(time_to_expiry)
    d1 = (
        log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call or put with the Black-Scholes-Merton model."""

    _validate_inputs(spot, strike, time_to_expiry, volatility)
    if option_type not in {"call", "put"}:
        raise InvalidOptionInputError("option_type must be 'call' or 'put'.")

    if time_to_expiry == 0:
        return (
            max(spot - strike, 0.0)
            if option_type == "call"
            else max(strike - spot, 0.0)
        )

    if volatility == 0:
        forward_adjusted_spot = spot * exp(
            (risk_free_rate - dividend_yield) * time_to_expiry
        )
        discounted_intrinsic = (
            max(forward_adjusted_spot - strike, 0.0)
            if option_type == "call"
            else max(strike - forward_adjusted_spot, 0.0)
        )
        return discounted_intrinsic * exp(-risk_free_rate * time_to_expiry)

    d1, d2 = d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
    )
    discounted_spot = spot * exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)


def intrinsic_value(spot: float, strike: float, option_type: OptionType) -> float:
    """Return the immediate exercise value."""

    if option_type == "call":
        return max(spot - strike, 0.0)
    if option_type == "put":
        return max(strike - spot, 0.0)
    raise InvalidOptionInputError("option_type must be 'call' or 'put'.")


def classify_moneyness(spot: float, strike: float, option_type: OptionType) -> str:
    """Classify an option as ITM, ATM, or OTM using a narrow ATM band."""

    ratio = spot / strike
    if 0.99 <= ratio <= 1.01:
        return "At the money"
    if option_type == "call":
        return "In the money" if spot > strike else "Out of the money"
    return "In the money" if spot < strike else "Out of the money"


def option_summary(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> OptionSummary:
    """Return prices and practical option diagnostics."""

    call_price = black_scholes_price(
        spot, strike, time_to_expiry, risk_free_rate, volatility, "call", dividend_yield
    )
    put_price = black_scholes_price(
        spot, strike, time_to_expiry, risk_free_rate, volatility, "put", dividend_yield
    )
    selected_price = call_price if option_type == "call" else put_price
    intrinsic = intrinsic_value(spot, strike, option_type)
    break_even = (
        strike + selected_price if option_type == "call" else strike - selected_price
    )

    return OptionSummary(
        call_price=call_price,
        put_price=put_price,
        selected_price=selected_price,
        intrinsic_value=intrinsic,
        time_value=max(selected_price - intrinsic, 0.0),
        moneyness=spot / strike,
        moneyness_label=classify_moneyness(spot, strike, option_type),
        break_even=break_even,
    )

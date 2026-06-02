"""Black-Scholes-Merton Greek calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

from scipy.stats import norm

from black_scholes_options_lab.exceptions import InvalidOptionInputError
from black_scholes_options_lab.pricing import OptionType, d1_d2


@dataclass(frozen=True)
class Greeks:
    """Core first- and second-order option sensitivities."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    @property
    def theta_per_day(self) -> float:
        return self.theta / 365

    @property
    def vega_per_percent(self) -> float:
        return self.vega / 100

    @property
    def rho_per_percent(self) -> float:
        return self.rho / 100


def calculate_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Calculate Black-Scholes-Merton Greeks.

    Theta is returned per year. Vega and rho are returned for a 1.00 absolute
    move in volatility/rates; convenience properties expose per-1% values.
    """

    if option_type not in {"call", "put"}:
        raise InvalidOptionInputError("option_type must be 'call' or 'put'.")
    if time_to_expiry <= 0 or volatility <= 0:
        return Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    d1, d2 = d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
    )
    discounted_spot_factor = exp(-dividend_yield * time_to_expiry)
    discounted_strike_factor = exp(-risk_free_rate * time_to_expiry)
    density = norm.pdf(d1)

    gamma = (
        discounted_spot_factor * density / (spot * volatility * sqrt(time_to_expiry))
    )
    vega = spot * discounted_spot_factor * density * sqrt(time_to_expiry)

    common_theta = -(
        spot
        * discounted_spot_factor
        * density
        * volatility
        / (2 * sqrt(time_to_expiry))
    )

    if option_type == "call":
        delta = discounted_spot_factor * norm.cdf(d1)
        theta = (
            common_theta
            - risk_free_rate * strike * discounted_strike_factor * norm.cdf(d2)
            + dividend_yield * spot * discounted_spot_factor * norm.cdf(d1)
        )
        rho = strike * time_to_expiry * discounted_strike_factor * norm.cdf(d2)
    else:
        delta = discounted_spot_factor * (norm.cdf(d1) - 1)
        theta = (
            common_theta
            + risk_free_rate * strike * discounted_strike_factor * norm.cdf(-d2)
            - dividend_yield * spot * discounted_spot_factor * norm.cdf(-d1)
        )
        rho = -strike * time_to_expiry * discounted_strike_factor * norm.cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

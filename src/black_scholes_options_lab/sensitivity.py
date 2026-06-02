"""Scenario and sensitivity helpers for option values and Greeks."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from black_scholes_options_lab.greeks import calculate_greeks
from black_scholes_options_lab.pricing import OptionType, black_scholes_price


def option_value_curve(
    variable: str,
    values: np.ndarray,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Return a one-dimensional option value sensitivity curve."""

    rows = []
    for value in values:
        kwargs = {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time_to_expiry,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
            "option_type": option_type,
            "dividend_yield": dividend_yield,
        }
        kwargs[variable] = max(float(value), 1e-9)
        rows.append(
            {
                variable: float(value),
                "option_value": black_scholes_price(**kwargs),
            }
        )
    return pd.DataFrame(rows)


def price_volatility_heatmap(
    spot_values: np.ndarray,
    volatility_values: np.ndarray,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Option value surface across stock prices and volatility assumptions."""

    return pd.DataFrame(
        [
            {
                "spot": float(spot),
                "volatility": float(volatility),
                "option_value": black_scholes_price(
                    spot=float(spot),
                    strike=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=float(volatility),
                    option_type=option_type,
                    dividend_yield=dividend_yield,
                ),
            }
            for volatility in volatility_values
            for spot in spot_values
        ]
    )


def price_time_heatmap(
    spot_values: np.ndarray,
    time_values: np.ndarray,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Option value surface across stock prices and time to expiration."""

    return pd.DataFrame(
        [
            {
                "spot": float(spot),
                "days_to_expiry": float(time_to_expiry * 365),
                "option_value": black_scholes_price(
                    spot=float(spot),
                    strike=strike,
                    time_to_expiry=max(float(time_to_expiry), 1e-9),
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    option_type=option_type,
                    dividend_yield=dividend_yield,
                ),
            }
            for time_to_expiry in time_values
            for spot in spot_values
        ]
    )


def greek_curve(
    spot_values: np.ndarray,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Greeks across a range of stock prices."""

    rows = []
    for spot in spot_values:
        greeks = calculate_greeks(
            spot=float(spot),
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        rows.append({"spot": float(spot), **asdict(greeks)})
    return pd.DataFrame(rows)


def scenario_table(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Return option price and Greeks under practical what-if scenarios."""

    scenarios = {
        "Base case": {},
        "Stock rises 10%": {"spot": spot * 1.10},
        "Stock falls 10%": {"spot": spot * 0.90},
        "Volatility +5 pts": {"volatility": volatility + 0.05},
        "Volatility -5 pts": {"volatility": max(volatility - 0.05, 0.01)},
        "30 days pass": {"time_to_expiry": max(time_to_expiry - 30 / 365, 1e-9)},
        "Rate +1 pt": {"risk_free_rate": risk_free_rate + 0.01},
        "Rate -1 pt": {"risk_free_rate": risk_free_rate - 0.01},
    }
    rows = []
    base_price = None
    for name, overrides in scenarios.items():
        kwargs = {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time_to_expiry,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
            "option_type": option_type,
            "dividend_yield": dividend_yield,
        }
        kwargs.update(overrides)
        price = black_scholes_price(**kwargs)
        if base_price is None:
            base_price = price
        greeks = calculate_greeks(**kwargs)
        rows.append(
            {
                "scenario": name,
                "option_price": price,
                "price_change": price - base_price,
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "theta_day": greeks.theta_per_day,
                "vega_1pct": greeks.vega_per_percent,
                "rho_1pct": greeks.rho_per_percent,
            }
        )
    return pd.DataFrame(rows)

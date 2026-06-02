"""Payoff calculations for common option strategies."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from black_scholes_options_lab.pricing import black_scholes_price


@dataclass(frozen=True)
class StrategyPayoff:
    """Payoff series and headline metrics for a strategy."""

    name: str
    prices: np.ndarray
    payoff: np.ndarray
    max_profit: float
    max_loss: float
    break_evens: tuple[float, ...]
    explanation: str


def _crossings(prices: np.ndarray, payoff: np.ndarray) -> tuple[float, ...]:
    """Approximate break-even points where payoff crosses zero."""

    breaks: list[float] = []
    for idx in range(1, len(prices)):
        left_payoff = payoff[idx - 1]
        right_payoff = payoff[idx]
        if left_payoff == 0:
            breaks.append(float(prices[idx - 1]))
        elif left_payoff * right_payoff < 0:
            left_price = prices[idx - 1]
            right_price = prices[idx]
            slope = (right_payoff - left_payoff) / (right_price - left_price)
            breaks.append(float(left_price - left_payoff / slope))
    return tuple(round(point, 2) for point in sorted(set(breaks)))


def estimate_premium(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    """Estimate a strategy leg premium using Black-Scholes."""

    return black_scholes_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=option_type,  # type: ignore[arg-type]
        dividend_yield=dividend_yield,
    )


def payoff_for_strategy(
    strategy: str,
    price_grid: np.ndarray,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> StrategyPayoff:
    """Return payoff data and metrics for a named strategy."""

    lower_strike = max(strike * 0.9, 0.01)
    upper_strike = strike * 1.1
    call_premium = estimate_premium(
        spot, strike, time_to_expiry, risk_free_rate, volatility, "call", dividend_yield
    )
    put_premium = estimate_premium(
        spot, strike, time_to_expiry, risk_free_rate, volatility, "put", dividend_yield
    )
    lower_put = estimate_premium(
        spot,
        lower_strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        "put",
        dividend_yield,
    )
    upper_call = estimate_premium(
        spot,
        upper_strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        "call",
        dividend_yield,
    )

    strategy_key = strategy.lower().strip()
    s = price_grid

    if strategy_key == "long call":
        payoff = np.maximum(s - strike, 0) - call_premium
        return StrategyPayoff(
            "Long Call",
            s,
            payoff,
            float("inf"),
            call_premium,
            (round(strike + call_premium, 2),),
            "Pays for upside exposure with a known premium at risk.",
        )

    if strategy_key == "long put":
        payoff = np.maximum(strike - s, 0) - put_premium
        return StrategyPayoff(
            "Long Put",
            s,
            payoff,
            max(strike - put_premium, 0),
            put_premium,
            (round(strike - put_premium, 2),),
            "Pays for downside exposure with limited loss equal to the premium.",
        )

    if strategy_key == "short call":
        payoff = call_premium - np.maximum(s - strike, 0)
        return StrategyPayoff(
            "Short Call",
            s,
            payoff,
            call_premium,
            float("inf"),
            (round(strike + call_premium, 2),),
            "Collects premium but carries theoretically unlimited upside risk.",
        )

    if strategy_key == "short put":
        payoff = put_premium - np.maximum(strike - s, 0)
        return StrategyPayoff(
            "Short Put",
            s,
            payoff,
            put_premium,
            max(strike - put_premium, 0),
            (round(strike - put_premium, 2),),
            "Collects premium while taking downside assignment risk.",
        )

    if strategy_key == "covered call":
        payoff = (s - spot) + call_premium - np.maximum(s - strike, 0)
        return StrategyPayoff(
            "Covered Call",
            s,
            payoff,
            max(strike - spot + call_premium, call_premium),
            max(spot - call_premium, 0),
            (round(spot - call_premium, 2),),
            "Combines stock ownership with a short call to trade upside for income.",
        )

    if strategy_key == "protective put":
        payoff = (s - spot) + np.maximum(strike - s, 0) - put_premium
        return StrategyPayoff(
            "Protective Put",
            s,
            payoff,
            float("inf"),
            max(spot - strike + put_premium, put_premium),
            (round(spot + put_premium, 2),),
            "Owns the stock and buys downside protection through a put.",
        )

    if strategy_key == "bull call spread":
        lower_call = call_premium
        higher_call = upper_call
        debit = lower_call - higher_call
        payoff = np.maximum(s - strike, 0) - np.maximum(s - upper_strike, 0) - debit
        width = upper_strike - strike
        return StrategyPayoff(
            "Bull Call Spread",
            s,
            payoff,
            max(width - debit, 0),
            max(debit, 0),
            (round(strike + debit, 2),),
            "Uses a long call and a higher-strike short call to define upside and loss.",
        )

    if strategy_key == "bear put spread":
        high_put = put_premium
        low_put = lower_put
        debit = high_put - low_put
        payoff = np.maximum(strike - s, 0) - np.maximum(lower_strike - s, 0) - debit
        width = strike - lower_strike
        return StrategyPayoff(
            "Bear Put Spread",
            s,
            payoff,
            max(width - debit, 0),
            max(debit, 0),
            (round(strike - debit, 2),),
            "Uses a long put and lower-strike short put for defined downside exposure.",
        )

    if strategy_key == "straddle":
        debit = call_premium + put_premium
        payoff = np.maximum(s - strike, 0) + np.maximum(strike - s, 0) - debit
        return StrategyPayoff(
            "Straddle",
            s,
            payoff,
            float("inf"),
            debit,
            (round(strike - debit, 2), round(strike + debit, 2)),
            "Buys a call and put at the same strike to express a large-move view.",
        )

    if strategy_key == "strangle":
        debit = lower_put + upper_call
        payoff = (
            np.maximum(lower_strike - s, 0) + np.maximum(s - upper_strike, 0) - debit
        )
        return StrategyPayoff(
            "Strangle",
            s,
            payoff,
            float("inf"),
            debit,
            (round(lower_strike - debit, 2), round(upper_strike + debit, 2)),
            "Buys an out-of-the-money put and call to reduce premium versus a straddle.",
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def format_metric(value: float) -> str:
    """Format finite strategy metrics while preserving unlimited risk labels."""

    if isfinite(value):
        return f"${value:,.2f}"
    return "Unlimited"

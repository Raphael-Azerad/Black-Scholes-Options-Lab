"""Small utility helpers for dates, formatting, and option inputs."""

from __future__ import annotations

from datetime import date


def years_to_expiry(days_to_expiry: int) -> float:
    """Convert calendar days to the year fraction used by Black-Scholes."""

    return max(days_to_expiry, 0) / 365


def days_between(start: date, end: date) -> int:
    """Return non-negative calendar days between two dates."""

    return max((end - start).days, 0)


def money(value: float) -> str:
    """Format a value as US dollars."""

    return f"${value:,.2f}"


def percent(value: float) -> str:
    """Format a decimal as a percent."""

    return f"{value:.2%}"

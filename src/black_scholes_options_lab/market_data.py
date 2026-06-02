"""Market data access through yfinance."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from black_scholes_options_lab.exceptions import MarketDataError


@dataclass(frozen=True)
class MarketSnapshot:
    """Historical prices and latest spot for one ticker."""

    ticker: str
    latest_price: float
    history: pd.DataFrame


def _normalize_yfinance_columns(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's single-ticker MultiIndex output when present."""

    if not isinstance(history.columns, pd.MultiIndex):
        return history

    if ticker in history.columns.get_level_values(-1):
        return history.xs(ticker, axis=1, level=-1)

    normalized = history.copy()
    normalized.columns = normalized.columns.get_level_values(0)
    return normalized


def fetch_market_snapshot(ticker: str, period: str = "2y") -> MarketSnapshot:
    """Fetch price history and latest close for a ticker."""

    normalized = ticker.strip().upper()
    if not normalized:
        raise MarketDataError("Enter a stock or ETF ticker to begin.")

    try:
        history = yf.download(
            normalized,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network boundary
        raise MarketDataError(f"Could not fetch market data for {normalized}.") from exc

    history = _normalize_yfinance_columns(history, normalized)

    if history.empty or "Close" not in history:
        raise MarketDataError(f"No usable price history was returned for {normalized}.")

    history = history.dropna(subset=["Close"])
    if history.empty:
        raise MarketDataError(f"No closing prices were returned for {normalized}.")

    return MarketSnapshot(
        ticker=normalized,
        latest_price=float(history["Close"].iloc[-1]),
        history=history,
    )


def fetch_treasury_proxy_rate(default_rate: float = 0.045) -> float:
    """Fetch a short Treasury proxy from Yahoo Finance.

    Yahoo quotes ^IRX as a percent yield. If the feed is unavailable, the
    dashboard can continue with a conservative editable fallback.
    """

    try:
        history = yf.download(
            "^IRX",
            period="5d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:  # pragma: no cover - network boundary
        return default_rate

    history = _normalize_yfinance_columns(history, "^IRX")

    if history.empty or "Close" not in history:
        return default_rate
    closes = history["Close"].dropna()
    if closes.empty:
        return default_rate
    latest = float(closes.iloc[-1])
    return latest / 100

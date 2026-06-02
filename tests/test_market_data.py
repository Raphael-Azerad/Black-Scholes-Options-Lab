import pandas as pd

from black_scholes_options_lab.market_data import (
    _normalize_yfinance_columns,
    fetch_treasury_proxy_rate,
)


def test_normalize_single_ticker_multiindex_columns():
    columns = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("Open", "AAPL")], names=["Price", "Ticker"]
    )
    history = pd.DataFrame([[101.0, 100.0]], columns=columns)

    normalized = _normalize_yfinance_columns(history, "AAPL")

    assert list(normalized.columns) == ["Close", "Open"]
    assert normalized.loc[0, "Close"] == 101.0


def test_treasury_proxy_rate_falls_back_on_empty_close_series(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame({"Close": [None, None]})

    monkeypatch.setattr(
        "black_scholes_options_lab.market_data.yf.download", fake_download
    )

    assert fetch_treasury_proxy_rate(default_rate=0.041) == 0.041

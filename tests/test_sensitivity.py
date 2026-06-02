import numpy as np

from black_scholes_options_lab.sensitivity import (
    greek_curve,
    option_value_curve,
    price_volatility_heatmap,
    scenario_table,
)


def test_option_value_curve_returns_one_row_per_value():
    values = np.array([90, 100, 110])
    df = option_value_curve("spot", values, 100, 100, 1, 0.05, 0.20, "call")
    assert len(df) == 3
    assert list(df.columns) == ["spot", "option_value"]


def test_price_volatility_heatmap_shape():
    df = price_volatility_heatmap(
        np.array([90, 100]),
        np.array([0.2, 0.3, 0.4]),
        100,
        1,
        0.05,
        "call",
    )
    assert len(df) == 6


def test_greek_curve_has_expected_columns():
    df = greek_curve(np.array([90, 100, 110]), 100, 1, 0.05, 0.20, "call")
    assert {"spot", "delta", "gamma", "theta", "vega", "rho"}.issubset(df.columns)


def test_scenario_table_includes_base_case_and_changes():
    df = scenario_table(100, 100, 1, 0.05, 0.20, "call")
    assert df.iloc[0]["scenario"] == "Base case"
    assert "Volatility +5 pts" in df["scenario"].to_list()
    assert df["option_price"].gt(0).all()

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from black_scholes_options_lab.exceptions import MarketDataError, OptionsLabError
from black_scholes_options_lab.greeks import calculate_greeks
from black_scholes_options_lab.market_data import (
    fetch_market_snapshot,
    fetch_treasury_proxy_rate,
)
from black_scholes_options_lab.pricing import option_summary
from black_scholes_options_lab.sensitivity import (
    greek_curve,
    option_value_curve,
    price_time_heatmap,
    price_volatility_heatmap,
    scenario_table,
)
from black_scholes_options_lab.strategies import (
    format_metric,
    payoff_for_strategy,
)
from black_scholes_options_lab.utils import (
    days_between,
    money,
    percent,
    years_to_expiry,
)
from black_scholes_options_lab.visualization import (
    bar_chart,
    heatmap_chart,
    line_chart,
    payoff_chart,
)
from black_scholes_options_lab.volatility import (
    historical_volatility,
    rolling_volatility,
)

st.set_page_config(
    page_title="Black-Scholes Options Lab",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #111111;
        --muted: #666666;
        --line: #e8e8e8;
        --accent: #0f766e;
        --soft: #f7f7f5;
    }
    .stApp {
        background: #ffffff;
        color: var(--ink);
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: #fff;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
        color: var(--ink);
    }
    .hero-note {
        border-left: 4px solid var(--accent);
        padding: 0.65rem 0 0.65rem 1rem;
        color: var(--muted);
        max-width: 980px;
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
    }
    div[data-testid="stHorizontalBlock"] > div {
        min-width: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_snapshot(ticker: str):
    return fetch_market_snapshot(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_treasury_rate() -> float:
    return fetch_treasury_proxy_rate()


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        column.metric(label, value, delta=delta)


def show_blank_state() -> None:
    st.title("Black-Scholes Options Lab")
    st.markdown(
        """
        <div class="hero-note">
        Enter a stock or ETF ticker in the sidebar to load market data and begin
        pricing European options with Black-Scholes-Merton.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Model scope")
        st.write(
            "The lab uses Black-Scholes-Merton as a transparent framework for "
            "understanding how spot price, strike, volatility, time, rates, and "
            "dividend yield affect European option value."
        )
    with right:
        st.subheader("Not investment advice")
        st.write(
            "Historical volatility is only a backward-looking assumption. The "
            "outputs are analytical estimates, not trading recommendations."
        )


st.sidebar.title("Inputs")
query_ticker = st.query_params.get("ticker", "")
if isinstance(query_ticker, list):
    query_ticker = query_ticker[0] if query_ticker else ""
ticker = (
    st.sidebar.text_input("Ticker", value=query_ticker, placeholder="AAPL, SPY, NVDA")
    .strip()
    .upper()
)

if not ticker:
    show_blank_state()
    st.stop()

with st.spinner(f"Fetching market data for {ticker}..."):
    try:
        snapshot = cached_market_snapshot(ticker)
    except MarketDataError as exc:
        st.title("Black-Scholes Options Lab")
        st.error(str(exc))
        st.stop()

close_prices = snapshot.history["Close"].dropna()
auto_rate = cached_treasury_rate()
hist_vol = historical_volatility(close_prices.tail(252))

st.sidebar.divider()
spot = st.sidebar.number_input(
    "Current stock price",
    min_value=0.01,
    value=float(round(snapshot.latest_price, 2)),
    step=1.0,
)
strike = st.sidebar.number_input(
    "Strike price",
    min_value=0.01,
    value=float(round(snapshot.latest_price, 0)),
    step=1.0,
)
option_type = st.sidebar.radio("Option type", ["call", "put"], horizontal=True)

expiry_mode = st.sidebar.radio(
    "Expiration input", ["Expiration date", "Days to expiration"]
)
if expiry_mode == "Expiration date":
    expiry_date = st.sidebar.date_input(
        "Expiration date",
        value=date.today() + timedelta(days=45),
        min_value=date.today(),
    )
    days_to_expiry = days_between(date.today(), expiry_date)
else:
    days_to_expiry = int(
        st.sidebar.number_input("Days to expiration", min_value=0, value=45, step=1)
    )
time_to_expiry = years_to_expiry(days_to_expiry)

st.sidebar.divider()
use_auto_rate = st.sidebar.toggle("Use Treasury proxy rate", value=True)
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate",
    min_value=-0.05,
    max_value=0.25,
    value=float(auto_rate if use_auto_rate else 0.045),
    step=0.0025,
    format="%.4f",
)
dividend_yield = st.sidebar.number_input(
    "Dividend yield",
    min_value=0.0,
    max_value=0.20,
    value=0.0,
    step=0.0025,
    format="%.4f",
    help="Continuous dividend yield used by the Black-Scholes-Merton adjustment.",
)
volatility_source = st.sidebar.radio(
    "Pricing volatility",
    ["Historical volatility", "Custom volatility"],
)
custom_volatility = st.sidebar.slider(
    "Custom volatility",
    min_value=0.01,
    max_value=1.50,
    value=float(max(min(hist_vol, 1.50), 0.01)),
    step=0.01,
    format="%.2f",
)
pricing_volatility = (
    hist_vol if volatility_source == "Historical volatility" else custom_volatility
)

try:
    summary = option_summary(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=pricing_volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )
    greeks = calculate_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=pricing_volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )
except OptionsLabError as exc:
    st.error(str(exc))
    st.stop()

st.title("Black-Scholes Options Lab")
st.caption(
    f"{ticker} · {money(spot)} spot · {days_to_expiry} days · "
    f"{percent(pricing_volatility)} volatility · {percent(risk_free_rate)} risk-free rate"
)

tab_labels = [
    "Pricing Engine",
    "Greeks Dashboard",
    "Volatility Lab",
    "Sensitivity Analysis",
    "Greeks Explorer",
    "Payoff Visualizer",
    "Scenario Testing",
    "Methodology",
]
requested_view = st.query_params.get("view", "")
if isinstance(requested_view, list):
    requested_view = requested_view[0] if requested_view else ""
view_map = {
    "pricing": "Pricing Engine",
    "greeks": "Greeks Dashboard",
    "volatility": "Volatility Lab",
    "sensitivity": "Sensitivity Analysis",
    "explorer": "Greeks Explorer",
    "payoff": "Payoff Visualizer",
    "scenario": "Scenario Testing",
    "methodology": "Methodology",
}
first_tab = view_map.get(str(requested_view).strip().lower())
if first_tab in tab_labels:
    tab_labels = [first_tab, *[label for label in tab_labels if label != first_tab]]
tab_by_label = dict(zip(tab_labels, st.tabs(tab_labels)))

with tab_by_label["Pricing Engine"]:
    metric_row(
        [
            ("Call price", money(summary.call_price), None),
            ("Put price", money(summary.put_price), None),
            ("Selected value", money(summary.selected_price), option_type.upper()),
            ("Break-even", money(summary.break_even), None),
        ]
    )
    st.write("")
    left, right = st.columns([1.05, 0.95])
    with left:
        metric_row(
            [
                ("Intrinsic value", money(summary.intrinsic_value), None),
                ("Time value", money(summary.time_value), None),
                ("Moneyness", f"{summary.moneyness:.3f}", summary.moneyness_label),
            ]
        )
    with right:
        st.subheader("Current assumptions")
        assumptions = pd.DataFrame(
            {
                "Input": [
                    "Ticker",
                    "Spot",
                    "Strike",
                    "Days to expiration",
                    "Risk-free rate",
                    "Dividend yield",
                    "Volatility source",
                    "Pricing volatility",
                ],
                "Value": [
                    ticker,
                    money(spot),
                    money(strike),
                    f"{days_to_expiry}",
                    percent(risk_free_rate),
                    percent(dividend_yield),
                    volatility_source,
                    percent(pricing_volatility),
                ],
            }
        )
        st.dataframe(assumptions, hide_index=True, use_container_width=True)

with tab_by_label["Greeks Dashboard"]:
    metric_row(
        [
            ("Delta", f"{greeks.delta:.4f}", "per $1 stock move"),
            ("Gamma", f"{greeks.gamma:.4f}", "delta curvature"),
            ("Theta", f"{greeks.theta_per_day:.4f}", "per day"),
            ("Vega", f"{greeks.vega_per_percent:.4f}", "per 1 vol point"),
            ("Rho", f"{greeks.rho_per_percent:.4f}", "per 1 rate point"),
        ]
    )
    st.write("")
    greek_notes = pd.DataFrame(
        [
            ("Delta", "Approximate option price change for a $1 move in the stock."),
            ("Gamma", "How quickly delta changes as the stock price changes."),
            (
                "Theta",
                "Estimated daily time decay, holding other assumptions constant.",
            ),
            (
                "Vega",
                "Estimated price change for a 1 percentage point volatility move.",
            ),
            ("Rho", "Estimated price change for a 1 percentage point rate move."),
        ],
        columns=["Greek", "Interpretation"],
    )
    st.dataframe(greek_notes, hide_index=True, use_container_width=True)

with tab_by_label["Volatility Lab"]:
    rolling_window = st.slider("Rolling volatility window", 10, 120, 30, 5)
    rolling = rolling_volatility(close_prices, rolling_window).dropna()
    vol_df = pd.DataFrame(
        {
            "date": rolling.index,
            "rolling_volatility": rolling.values,
        }
    )
    metric_row(
        [
            ("Historical volatility", percent(hist_vol), "last 252 closes"),
            (
                "Selected pricing volatility",
                percent(pricing_volatility),
                volatility_source,
            ),
            (
                "Difference",
                percent(pricing_volatility - hist_vol),
                "selected - historical",
            ),
        ]
    )
    st.plotly_chart(
        line_chart(
            vol_df,
            x="date",
            y="rolling_volatility",
            title=f"{ticker} rolling annualized volatility",
            labels={"rolling_volatility": "Annualized volatility", "date": "Date"},
        ),
        use_container_width=True,
    )
    st.markdown(
        "<p class='small-muted'>Historical volatility is backward-looking and does not "
        "represent implied volatility from listed option prices.</p>",
        unsafe_allow_html=True,
    )

with tab_by_label["Sensitivity Analysis"]:
    st.markdown(
        "<p class='small-muted'>Sensitivity charts hold all other assumptions constant. "
        "They show model mechanics, not forecasts.</p>",
        unsafe_allow_html=True,
    )
    spot_values = np.linspace(max(spot * 0.65, 0.01), spot * 1.35, 60)
    vol_values = np.linspace(
        max(pricing_volatility * 0.35, 0.01), pricing_volatility * 1.85, 50
    )
    time_values = np.linspace(
        max(1 / 365, time_to_expiry * 0.15), max(time_to_expiry * 1.8, 2 / 365), 50
    )
    strike_values = np.linspace(max(strike * 0.75, 0.01), strike * 1.25, 60)

    curve_cols = st.columns(2)
    curves = [
        (
            "spot",
            spot_values,
            f"{ticker} stock price vs option value",
            {"spot": "Stock price", "option_value": "Option value"},
        ),
        (
            "volatility",
            vol_values,
            "Volatility vs option value",
            {"volatility": "Volatility", "option_value": "Option value"},
        ),
        (
            "time_to_expiry",
            time_values,
            "Time to expiration vs option value",
            {"time_to_expiry": "Years to expiration", "option_value": "Option value"},
        ),
        (
            "strike",
            strike_values,
            "Strike price vs option value",
            {"strike": "Strike price", "option_value": "Option value"},
        ),
    ]
    for idx, (variable, values, title, labels) in enumerate(curves):
        df = option_value_curve(
            variable,
            values,
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            pricing_volatility,
            option_type,
            dividend_yield,
        )
        curve_cols[idx % 2].plotly_chart(
            line_chart(df, x=variable, y="option_value", title=title, labels=labels),
            use_container_width=True,
        )

    heat_cols = st.columns(2)
    heat_cols[0].plotly_chart(
        heatmap_chart(
            price_volatility_heatmap(
                spot_values,
                vol_values,
                strike,
                time_to_expiry,
                risk_free_rate,
                option_type,
                dividend_yield,
            ),
            x="spot",
            y="volatility",
            z="option_value",
            title="Stock price vs volatility",
            labels={"x": "Stock price", "y": "Volatility"},
        ),
        use_container_width=True,
    )
    heat_cols[1].plotly_chart(
        heatmap_chart(
            price_time_heatmap(
                spot_values,
                time_values,
                strike,
                risk_free_rate,
                pricing_volatility,
                option_type,
                dividend_yield,
            ),
            x="spot",
            y="days_to_expiry",
            z="option_value",
            title="Stock price vs time to expiration",
            labels={"x": "Stock price", "y": "Days to expiration"},
        ),
        use_container_width=True,
    )

with tab_by_label["Greeks Explorer"]:
    explorer_df = greek_curve(
        spot_values,
        strike,
        time_to_expiry,
        risk_free_rate,
        pricing_volatility,
        option_type,
        dividend_yield,
    )
    greek_cols = st.columns(2)
    greek_specs = [
        ("delta", "Delta curve"),
        ("gamma", "Gamma curve"),
        ("theta", "Theta decay curve"),
        ("vega", "Vega sensitivity chart"),
        ("rho", "Rho sensitivity chart"),
    ]
    for idx, (field, title) in enumerate(greek_specs):
        greek_cols[idx % 2].plotly_chart(
            line_chart(
                explorer_df,
                x="spot",
                y=field,
                title=title,
                labels={"spot": "Stock price", field: field.title()},
            ),
            use_container_width=True,
        )

with tab_by_label["Payoff Visualizer"]:
    strategy = st.selectbox(
        "Strategy",
        [
            "Long Call",
            "Long Put",
            "Short Call",
            "Short Put",
            "Covered Call",
            "Protective Put",
            "Bull Call Spread",
            "Bear Put Spread",
            "Straddle",
            "Strangle",
        ],
    )
    payoff_grid = np.linspace(max(spot * 0.45, 0.01), spot * 1.65, 180)
    payoff = payoff_for_strategy(
        strategy,
        payoff_grid,
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        pricing_volatility,
        dividend_yield,
    )
    payoff_df = pd.DataFrame({"stock_price": payoff.prices, "payoff": payoff.payoff})
    metric_row(
        [
            ("Max profit", format_metric(payoff.max_profit), None),
            ("Max loss", format_metric(payoff.max_loss), None),
            (
                "Break-even",
                ", ".join(money(point) for point in payoff.break_evens) or "N/A",
                None,
            ),
        ]
    )
    st.plotly_chart(payoff_chart(payoff_df, payoff.name), use_container_width=True)
    st.write(payoff.explanation)

with tab_by_label["Scenario Testing"]:
    scenarios = scenario_table(
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        pricing_volatility,
        option_type,
        dividend_yield,
    )
    display = scenarios.copy()
    for column in ["option_price", "price_change"]:
        display[column] = display[column].map(lambda value: f"${value:,.2f}")
    for column in ["delta", "gamma", "theta_day", "vega_1pct", "rho_1pct"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    st.dataframe(display, hide_index=True, use_container_width=True)
    scenario_cards = scenarios.set_index("scenario")
    metric_row(
        [
            (
                "Stock rises 10%",
                money(scenario_cards.loc["Stock rises 10%", "option_price"]),
                money(scenario_cards.loc["Stock rises 10%", "price_change"]),
            ),
            (
                "Volatility +5 pts",
                money(scenario_cards.loc["Volatility +5 pts", "option_price"]),
                money(scenario_cards.loc["Volatility +5 pts", "price_change"]),
            ),
            (
                "30 days pass",
                money(scenario_cards.loc["30 days pass", "option_price"]),
                money(scenario_cards.loc["30 days pass", "price_change"]),
            ),
        ]
    )
    st.plotly_chart(
        bar_chart(
            scenarios, x="scenario", y="price_change", title="Scenario price change"
        ),
        use_container_width=True,
    )
    st.markdown(
        "<p class='small-muted'>Each scenario changes one input from the base case and "
        "recalculates the selected option.</p>",
        unsafe_allow_html=True,
    )

with tab_by_label["Methodology"]:
    st.subheader("Black-Scholes-Merton framework")
    st.write(
        "The model prices European options by combining current stock price, strike, "
        "time to expiration, volatility, rates, and continuous dividend yield. It is "
        "useful because each input has a visible effect, but it is still a model with "
        "strong assumptions."
    )
    st.subheader("Core limitations")
    st.markdown(
        """
        - The model assumes European-style exercise.
        - It assumes lognormal underlying price behavior.
        - Volatility, rates, and dividend yield are held constant.
        - Transaction costs, liquidity, bid-ask spreads, and early exercise effects are ignored.
        - Historical volatility is not implied volatility from the option chain.
        - Model values can differ materially from traded option prices.
        - Outputs are educational estimates, not investment advice or trade recommendations.
        """
    )

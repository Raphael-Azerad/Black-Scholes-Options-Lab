# Black-Scholes Options Lab

Black-Scholes Options Lab is a Streamlit app for pricing European options, analyzing Greeks, testing volatility assumptions, and visualizing payoff profiles. It uses the Black-Scholes-Merton model as a clear framework for seeing how spot price, strike, volatility, time, dividend yield, and interest rates affect option value.

This is an educational finance project. It is built to make the model inspectable, not to recommend trades.

## Key Findings From Default Example

Default example: `AAPL`

The values below were generated from the project on June 2, 2026 using live market data from Yahoo Finance.

| Metric | Value |
| --- | --- |
| Current stock price | `$309.73` |
| Selected strike | `$310.00` |
| Days to expiration | `45` |
| Volatility assumption | `22.07%` historical volatility |
| Risk-free rate | `3.62%` Treasury proxy |
| Black-Scholes call price | `$10.12` |
| Black-Scholes put price | `$9.01` |
| Delta | `0.5339` |
| Gamma | `0.0166` |
| Theta per day | `-0.1214` |
| Vega per volatility point | `0.4323` |
| Rho per rate point | `0.1914` |

Using AAPL as the default example, the selected strike is close to at-the-money, so the call delta sits near 0.53. The model output is most sensitive to stock price and volatility around this strike.

- Raising volatility by 5 percentage points increases the call estimate from `$10.12` to `$12.28`, a gain of about `$2.16`.
- Letting 30 calendar days pass reduces the call estimate to `$5.62`, a drop of about `$4.50`, showing how much of the value is time value.
- In the sensitivity view, the option value changes most sharply near the selected strike, where small stock moves have the largest effect on moneyness.

## Features

- Black-Scholes-Merton call and put pricing for European-style options
- Editable assumptions for spot, strike, expiration, volatility, dividend yield, and risk-free rate
- Live ticker lookup through `yfinance`
- Treasury proxy rate lookup through `^IRX`
- Historical volatility and rolling volatility charts
- Delta, Gamma, Theta, Vega, and Rho dashboard
- Sensitivity curves for stock price, volatility, time to expiration, and strike
- Heatmaps for stock price vs volatility and stock price vs time
- Greek curves across stock price ranges
- Payoff diagrams for common single- and multi-leg strategies
- Scenario testing for price moves, volatility shocks, time decay, and rate shifts
- Pytest coverage for pricing, Greeks, volatility, payoff logic, and sensitivity helpers
- GitHub Actions workflow for tests, Ruff, and Black

## Methodology

The app prices European options with the Black-Scholes-Merton model. The dividend-adjusted call and put prices are:

```text
Call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
Put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

d1 = [ln(S/K) + (r - q + sigma^2 / 2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

Where:

- `S` is the current stock price
- `K` is the strike price
- `T` is time to expiration in years
- `r` is the continuously compounded risk-free rate
- `q` is the continuous dividend yield
- `sigma` is volatility
- `N()` is the standard normal cumulative distribution function

## Greeks

The app calculates:

- **Delta:** approximate option price change for a $1 move in the underlying stock.
- **Gamma:** how quickly delta changes as the stock price changes.
- **Theta:** estimated time decay, displayed per day in the dashboard.
- **Vega:** estimated option price change for a 1 percentage point move in volatility.
- **Rho:** estimated option price change for a 1 percentage point move in rates.

## Volatility Assumptions

Historical volatility is calculated from daily log returns and annualized with 252 trading days. The dashboard lets the user choose between this historical estimate and a custom volatility input.

Historical volatility is backward-looking. It is not the same as implied volatility from listed option prices, and the app does not claim that it predicts future option prices.

## Strategy Payoff Section

The payoff visualizer includes:

- Long call
- Long put
- Short call
- Short put
- Covered call
- Protective put
- Bull call spread
- Bear put spread
- Straddle
- Strangle

Premiums for strategy examples are estimated with the same Black-Scholes-Merton assumptions used elsewhere in the app. Multi-leg strategies use simple preset strikes around the selected strike to keep the section focused and readable.

## Project Structure

```text
Black-Scholes-Options-Lab/
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── assets/
├── data/
├── notebooks/
│   └── 01_black_scholes_methodology.ipynb
├── src/
│   └── black_scholes_options_lab/
│       ├── __init__.py
│       ├── pricing.py
│       ├── greeks.py
│       ├── volatility.py
│       ├── market_data.py
│       ├── strategies.py
│       ├── sensitivity.py
│       ├── visualization.py
│       ├── utils.py
│       └── exceptions.py
├── tests/
│   ├── test_pricing.py
│   ├── test_greeks.py
│   ├── test_volatility.py
│   ├── test_strategies.py
│   └── test_sensitivity.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Installation

```bash
git clone https://github.com/Raphael-Azerad/Black-Scholes-Options-Lab.git
cd Black-Scholes-Options-Lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
streamlit run app.py
```

Enter a ticker in the sidebar, review the market-data-loaded assumptions, and move through the dashboard tabs.

## Testing

```bash
pytest
ruff check .
black --check .
```

## Deployment

The app is ready for Streamlit Community Cloud:

1. Create a new Streamlit Community Cloud app from this repository.
2. Use `app.py` as the entrypoint.
3. Keep the default Python package install from `requirements.txt`.

No secrets are required for the default Yahoo Finance data path.

## Limitations

- Black-Scholes-Merton assumes European-style exercise.
- It assumes lognormal underlying price behavior.
- It assumes constant volatility, risk-free rates, and dividend yield.
- It ignores transaction costs, liquidity, bid-ask spreads, and early exercise effects.
- Historical volatility is not implied volatility from the option chain.
- Model values may differ materially from real market option prices.
- The project is educational and analytical, not investment advice.
- Outputs should not be interpreted as trading recommendations.

## References

- Fischer Black and Myron Scholes, “The Pricing of Options and Corporate Liabilities,” 1973.
- Robert C. Merton, “Theory of Rational Option Pricing,” 1973.
- John C. Hull, _Options, Futures, and Other Derivatives_.
- SciPy documentation for the standard normal distribution.
- Yahoo Finance market data through `yfinance`.

## License

MIT License. See [LICENSE](LICENSE).

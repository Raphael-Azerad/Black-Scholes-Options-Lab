"""Plotly chart builders with the project's visual identity."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BW_TEMPLATE = "plotly_white"
ACCENT = "#0f766e"
DARK = "#111111"
MUTED = "#737373"


def apply_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply a compact black-and-white chart style."""

    fig.update_layout(
        template=BW_TEMPLATE,
        title=title,
        font={"family": "Inter, Helvetica, Arial, sans-serif", "color": DARK},
        margin={"l": 40, "r": 20, "t": 54 if title else 24, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ededed", zerolinecolor="#d4d4d4")
    fig.update_yaxes(showgrid=True, gridcolor="#ededed", zerolinecolor="#d4d4d4")
    return fig


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    title: str,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Create a styled line chart."""

    fig = px.line(df, x=x, y=y, labels=labels)
    fig.update_traces(line={"width": 2.6})
    return apply_layout(fig, title)


def payoff_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a payoff diagram with a zero-profit reference line."""

    fig = px.line(df, x="stock_price", y="payoff", labels={"payoff": "Profit / loss"})
    fig.update_traces(line={"color": ACCENT, "width": 3})
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    return apply_layout(fig, title)


def heatmap_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Create a styled heatmap."""

    pivot = df.pivot(index=y, columns=x, values=z)
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels=labels,
        origin="lower",
    )
    fig.update_layout(coloraxis_colorbar={"title": "Option value"})
    return apply_layout(fig, title)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Create a compact bar chart."""

    fig = px.bar(df, x=x, y=y, color=y, color_continuous_scale="Teal")
    fig.update_layout(showlegend=False)
    return apply_layout(fig, title)

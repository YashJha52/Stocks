"""
financial_forecaster.py
Forecasts future financial statement line items using trend regression
and historical growth rates.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------------------------------------------------------
# Descriptive Statistics
# ---------------------------------------------------------------------------

class DescriptiveStats:
    """Computes key statistics on a financial data series."""

    def __init__(self, data: pd.Series | list | np.ndarray):
        self._s = np.asarray(data, dtype=float)
        self._s = self._s[~np.isnan(self._s)]
        if len(self._s) == 0:
            raise ValueError("Cannot compute stats on an empty series.")

    @property
    def mean(self) -> float:
        return float(np.mean(self._s))

    @property
    def median(self) -> float:
        return float(np.median(self._s))

    @property
    def sample_std(self) -> float:
        return float(np.std(self._s, ddof=1))

    @property
    def coefficient_of_variation(self) -> float:
        """How volatile this metric is relative to its average (%)."""
        return float(self.sample_std / self.mean * 100) if self.mean != 0 else np.nan

    def quartile(self, q: int) -> float:
        return float(np.percentile(self._s, q * 25))

    @property
    def skewness(self) -> float:
        """Positive = skewed high, Negative = skewed low."""
        mode_approx = 3 * self.median - 2 * self.mean
        return (self.mean - mode_approx) / self.sample_std if self.sample_std else 0.0

    def normal_range(self) -> tuple[float, float]:
        """Typical range covering ~95% of historical values."""
        return (self.mean - 2 * self.sample_std,
                self.mean + 2 * self.sample_std)

    def summary(self) -> dict[str, Any]:
        low, high = self.normal_range()
        return {
            "average":              round(self.mean, 2),
            "median":               round(self.median, 2),
            "std_dev":              round(self.sample_std, 2),
            "volatility_%":         round(self.coefficient_of_variation, 1),
            "Q1":                   round(self.quartile(1), 2),
            "Q3":                   round(self.quartile(3), 2),
            "skewness":             round(self.skewness, 3),
            "typical_range_low":    round(low, 2),
            "typical_range_high":   round(high, 2),
        }


# ---------------------------------------------------------------------------
# Trend Regression
# ---------------------------------------------------------------------------

class TrendModel:
    """
    Fits a simple linear trend to a time series and forecasts forward.
    Also stores model quality metrics (R², RMSE).
    """

    def __init__(self):
        self._model = LinearRegression()
        self.r_squared_: float | None = None
        self.rmse_: float | None = None
        self.slope_: float | None = None
        self.intercept_: float | None = None
        self._n: int = 0
        self._trend_direction: str = ""

    def fit(self, values: np.ndarray) -> "TrendModel":
        n = len(values)
        self._n = n
        t = np.arange(n).reshape(-1, 1)
        self._model.fit(t, values)
        self.slope_     = float(self._model.coef_[0])
        self.intercept_ = float(self._model.intercept_)
        y_pred = self._model.predict(t)
        self.r_squared_ = float(r2_score(values, y_pred))
        self.rmse_      = float(np.sqrt(mean_squared_error(values, y_pred)))
        self._trend_direction = "upward" if self.slope_ > 0 else "downward"
        return self

    def forecast(self, periods: int) -> list[float]:
        future = np.arange(self._n, self._n + periods).reshape(-1, 1)
        return self._model.predict(future).tolist()

    @property
    def trend_strength(self) -> str:
        if self.r_squared_ is None:
            return "unknown"
        if self.r_squared_ >= 0.85:
            return "strong"
        if self.r_squared_ >= 0.5:
            return "moderate"
        return "weak"

    @property
    def trend_summary(self) -> str:
        """Plain-English summary of the trend."""
        if self.r_squared_ is None:
            return "No trend data available."
        pct = abs(self.slope_) / abs(self.intercept_) * 100 if self.intercept_ else 0
        return (
            f"Shows a {self.trend_strength} {self._trend_direction} trend "
            f"(R² = {self.r_squared_:.0%}), growing roughly "
            f"{pct:.1f}% per period on average."
        )


# ---------------------------------------------------------------------------
# Core Forecaster
# ---------------------------------------------------------------------------

class FinancialForecaster:
    """
    Forecasts future quarterly/annual financial statement figures
    (Revenue, Net Income, Cash Flow, etc.) for a given company.

    Two forecasting approaches are blended:
      1. Linear trend — fits a straight trend line through history
      2. Growth rate   — compounds the historical average growth rate forward
    The final "Best Estimate" is the average of the two.
    """

    _INCOME_ITEMS = [
        "Total Revenue", "Gross Profit", "Operating Income",
        "EBITDA", "Net Income", "Basic EPS",
    ]
    _BALANCE_ITEMS = [
        "Total Assets", "Total Liabilities Net Minority Interest",
        "Stockholders Equity",
    ]
    _CASHFLOW_ITEMS = [
        "Operating Cash Flow", "Free Cash Flow", "Capital Expenditure",
    ]

    def __init__(self, ticker: str, financials: dict):
        self.ticker = ticker
        self._income   = self._clean(financials.get("income_statement"))
        self._balance  = self._clean(financials.get("balance_sheet"))
        self._cashflow = self._clean(financials.get("cash_flow"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_forecast_report(self, periods: int = 4) -> str:
        lines: list[str] = []
        lines.append(f"# Financial Forecast: {self.ticker}\n")
        lines.append(
            f"The tables below show the most likely values for the next "
            f"**{periods} reporting periods**, based on historical trends.\n"
        )
        lines.append("---\n")

        lines.append("## Income Statement\n")
        lines.append(self._render(self._forecast_statement(
            self._income, self._INCOME_ITEMS, periods), periods))

        lines.append("## Balance Sheet\n")
        lines.append(self._render(self._forecast_statement(
            self._balance, self._BALANCE_ITEMS, periods), periods))

        lines.append("## Cash Flow\n")
        lines.append(self._render(self._forecast_statement(
            self._cashflow, self._CASHFLOW_ITEMS, periods), periods))

        lines.append("""
---
> **Note:** These forecasts are based on historical patterns and straight-line
> trend analysis. They do not account for unexpected events, earnings surprises,
> or changes in business strategy. Use as a directional guide only.
""")
        return "\n".join(lines)

    def get_regression_forecast(self, line_item: str, periods: int = 4) -> dict | None:
        series = self._find_series(line_item)
        if series is None or len(series) < 3:
            return None
        return self._build_forecast(series.values.astype(float), periods)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        if isinstance(df.columns[0], pd.Timestamp):
            df = df.sort_index(axis=1)
        return df

    def _find_series(self, item: str) -> pd.Series | None:
        for df in [self._income, self._balance, self._cashflow]:
            if df is None:
                continue
            if item in df.index:
                s = df.loc[item].dropna().astype(float)
                if len(s) >= 3:
                    return s
            matches = [i for i in df.index if item.lower() in str(i).lower()]
            if matches:
                s = df.loc[matches[0]].dropna().astype(float)
                if len(s) >= 3:
                    return s
        return None

    def _forecast_statement(
        self, df: pd.DataFrame | None, items: list[str], periods: int
    ) -> dict[str, dict]:
        results: dict[str, dict] = {}
        if df is None:
            return results
        for item in items:
            series = self._find_series(item)
            if series is None:
                continue
            try:
                results[item] = self._build_forecast(series.values.astype(float), periods)
            except Exception:
                pass
        return results

    def _build_forecast(self, values: np.ndarray, periods: int) -> dict:
        n = len(values)

        # --- Trend forecast ---
        trend = TrendModel().fit(values)
        trend_forecast = trend.forecast(periods)

        # --- Growth-rate forecast ---
        growth_forecast: list[float] = []
        avg_growth: float | None = None
        if np.all(values > 0) and n >= 2:
            # Geometric mean growth: r = (last/first)^(1/(n-1)) - 1
            avg_growth = float((values[-1] / values[0]) ** (1 / (n - 1)) - 1)
            last = values[-1]
            for p in range(1, periods + 1):
                growth_forecast.append(last * (1 + avg_growth) ** p)
        else:
            growth_forecast = trend_forecast[:]  # fallback

        # --- Best estimate (average of both) ---
        best = [(t + g) / 2 for t, g in zip(trend_forecast, growth_forecast)]

        # --- Descriptive stats ---
        ds = DescriptiveStats(values)

        return {
            "historical":       values.tolist(),
            "trend_forecast":   [round(v, 2) for v in trend_forecast],
            "growth_forecast":  [round(v, 2) for v in growth_forecast],
            "best_estimate":    [round(v, 2) for v in best],
            "avg_growth_pct":   round(avg_growth * 100, 2) if avg_growth is not None else None,
            "trend_summary":    trend.trend_summary,
            "r_squared":        round(trend.r_squared_, 3) if trend.r_squared_ else None,
            "stats":            ds.summary(),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, forecasts: dict[str, dict], periods: int) -> str:
        if not forecasts:
            return "*Not enough historical data to forecast this statement.*\n\n"

        lines: list[str] = []
        for item, data in forecasts.items():
            lines.append(f"### {item}")

            # Plain-English trend line
            lines.append(f"*{data['trend_summary']}*")
            if data["avg_growth_pct"] is not None:
                direction = "growing" if data["avg_growth_pct"] >= 0 else "declining"
                lines.append(
                    f"Average historical growth: **{data['avg_growth_pct']:+.1f}%** per period "
                    f"({direction})."
                )

            s = data["stats"]
            lines.append(
                f"Historical average: **{_fmt(s['average'])}** "
                f"| Typical range: {_fmt(s['typical_range_low'])} – {_fmt(s['typical_range_high'])}"
            )

            # Forecast table
            lines.append("")
            header = "| Period | Trend Estimate | Growth Estimate | Best Estimate |"
            sep    = "|--------|---------------|-----------------|---------------|"
            lines.append(header)
            lines.append(sep)
            for i, (t, g, b) in enumerate(
                zip(data["trend_forecast"], data["growth_forecast"],
                    data["best_estimate"]), start=1
            ):
                lines.append(
                    f"| Period +{i} | {_fmt(t)} | {_fmt(g)} | **{_fmt(b)}** |"
                )
            lines.append("")

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if abs(v) >= 1e9:  return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:  return f"${v / 1e6:.2f}M"
        if abs(v) >= 1e3:  return f"${v / 1e3:.2f}K"
        return f"{v:,.2f}"
    except (TypeError, ValueError):
        return str(value)
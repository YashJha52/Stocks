"""
app.py  —  Streamlit frontend
Run:  streamlit run frontend/app.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import StatisticalFeatures
from src.models import StockPredictor
from src.report_generator import ReportGenerator
from src.financial_forecaster import FinancialForecaster

# ---------------------------------------------------------------------------
# Utility functions  (defined first to avoid reference-before-definition)
# ---------------------------------------------------------------------------

def _fmt_cap(value) -> str:
    if not value:
        return "N/A"
    try:
        v = float(value)
        if v >= 1e12: return f"${v/1e12:.2f}T"
        if v >= 1e9:  return f"${v/1e9:.2f}B"
        if v >= 1e6:  return f"${v/1e6:.2f}M"
        return f"${v:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_val(value) -> str:
    if value is None or value == "N/A":
        return "N/A"
    try:
        v = float(value)
        if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
        if abs(v) >= 1e3:  return f"${v/1e3:.2f}K"
        return f"{v:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _next_bday(base, offset: int):
    count, current = 0, base
    while count < offset:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return current


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stock Predictor & Financial Analyst",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .main-header { font-size:2.6rem; color:#1f77b4; text-align:center; margin-bottom:1.5rem; }
    .sub-header  { font-size:1.4rem; color:#2c3e50; margin-top:1.8rem; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str):
    loader = DataLoader()
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    company_info = loader.get_company_info(ticker)
    stock_data   = loader.load_stock_data(ticker, start, end)
    return company_info, stock_data


@st.cache_data(ttl=600, show_spinner=False)
def load_financials(ticker: str, quarterly: bool):
    loader = DataLoader()
    return loader.get_financial_statements(ticker, quarterly=quarterly)


@st.cache_resource(show_spinner=False)
def get_trained_predictor(ticker: str, data_hash: int):
    model_dir = Path("models") / "saved" / ticker
    if model_dir.exists():
        try:
            return StockPredictor.load(model_dir)
        except Exception:
            pass

    loader = DataLoader()
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    stock_data = loader.load_stock_data(ticker, start, end)

    fe = StatisticalFeatures()
    features = fe.compute_features(stock_data)
    features["target"] = stock_data["Close"].shift(-1)
    features = features.dropna()

    X = features.drop(columns=["target"])
    y = features["target"]

    predictor = StockPredictor()
    predictor.train_models(X, y)
    return predictor


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key in ("predictions", "price_report", "fin_forecast_report",
            "last_ticker", "financials"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("## Stock Analysis")
ticker        = st.sidebar.text_input("Ticker symbol", value="AAPL").strip().upper()
forecast_days = st.sidebar.slider("Price forecast days", 5, 60, 30)
fin_periods   = st.sidebar.slider("Financial forecast periods", 1, 8, 4)
quarterly     = st.sidebar.checkbox(
    "Use quarterly reports",
    value=True,
    help="Quarterly data gives more historical points for a better forecast."
)
analyze_btn = st.sidebar.button("Analyse", use_container_width=True)

st.markdown('<h1 class="main-header">📈 Stock Predictor & Financial Analyst</h1>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_price, tab_forecast, tab_stats = st.tabs([
    "📊 Price Analysis",
    "📋 Financial Forecast",
    "📐 Deep Dive",
])

# ===========================================================================
# TAB 1 — Price Analysis
# ===========================================================================

with tab_price:
    if analyze_btn and ticker:
        if st.session_state.last_ticker != ticker:
            for k in ("predictions", "price_report", "fin_forecast_report", "financials"):
                st.session_state[k] = None
            st.session_state.last_ticker = ticker

        with st.spinner(f"Loading data for {ticker}..."):
            try:
                company_info, stock_data = load_data(ticker)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Company",       company_info.get("longName", ticker))
        col2.metric("Sector",        company_info.get("sector", "N/A"))
        col3.metric("Current Price", f"${company_info.get('currentPrice', 0):.2f}")
        col4.metric("Market Cap",    _fmt_cap(company_info.get("marketCap")))

        st.markdown('<h2 class="sub-header">Price History (12 months)</h2>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"], high=stock_data["High"],
            low=stock_data["Low"],   close=stock_data["Close"],
            name=ticker,
        ))
        fig.update_layout(title=f"{ticker}", yaxis_title="Price (USD)",
                          xaxis_rangeslider_visible=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<h2 class="sub-header">Price Forecast</h2>',
                    unsafe_allow_html=True)
        with st.spinner("Generating price forecast..."):
            try:
                data_hash = hash(str(stock_data.index[-1]))
                predictor = get_trained_predictor(ticker, data_hash)

                fe = StatisticalFeatures()
                history = stock_data[["Open", "High", "Low", "Close", "Volume"]].copy()
                forecast_records = []
                last_date = history.index[-1]

                for step in range(forecast_days):
                    feats = fe.compute_features(history).dropna()
                    if feats.empty:
                        break
                    pred_df  = predictor.predict(feats.iloc[[-1]])
                    pred_row = pred_df.iloc[0]
                    next_date = _next_bday(last_date, step + 1)
                    record = pred_row.to_dict()
                    record["date"] = next_date
                    forecast_records.append(record)
                    ep = float(pred_row["ensemble"])
                    new_row = pd.DataFrame(
                        {"Open": ep, "High": ep*1.005, "Low": ep*0.995,
                         "Close": ep,
                         "Volume": history["Volume"].rolling(20).mean().iloc[-1]},
                        index=[next_date],
                    )
                    history = pd.concat([history, new_row])

                if forecast_records:
                    pred_df = pd.DataFrame(forecast_records).set_index("date")
                    st.session_state.predictions = pred_df
            except Exception as exc:
                st.error(f"Forecast error: {exc}")

        if st.session_state.predictions is not None:
            pred_df = st.session_state.predictions
            fig2 = go.Figure()
            hist_tail = stock_data["Close"].iloc[-60:]
            fig2.add_trace(go.Scatter(
                x=hist_tail.index, y=hist_tail.values,
                mode="lines", name="Historical",
                line=dict(color="#888", width=1.5)
            ))
            for col in [c for c in pred_df.columns if c != "ensemble"]:
                fig2.add_trace(go.Scatter(
                    x=pred_df.index, y=pred_df[col],
                    mode="lines", name=col.upper(),
                    line=dict(width=1), opacity=0.35,
                ))
            fig2.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df["ensemble"],
                mode="lines+markers", name="Best Estimate",
                line=dict(width=2.5, color="#1f77b4"),
            ))
            fig2.update_layout(
                title=f"{ticker} — {forecast_days}-day price forecast",
                xaxis_title="Date", yaxis_title="Price (USD)", height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)
            with st.expander("View forecast numbers"):
                st.dataframe(pred_df[["ensemble"]].rename(
                    columns={"ensemble": "Best Estimate ($)"}
                ).style.format("${:.2f}"), use_container_width=True)

        # Price + financial report
        if st.session_state.predictions is not None:
            try:
                fin_data = load_financials(ticker, quarterly=False)
                st.session_state.financials = fin_data
                company_info_r, _ = load_data(ticker)
                rg = ReportGenerator()
                report = rg.generate_financial_report(
                    ticker, company_info_r, fin_data,
                    st.session_state.predictions,
                )
                st.session_state.price_report = report
            except Exception:
                pass

        if st.session_state.price_report:
            st.markdown('<h2 class="sub-header">Financial Summary</h2>',
                        unsafe_allow_html=True)
            st.markdown(st.session_state.price_report)

    else:
        st.info("Enter a ticker symbol and click **Analyse** to get started.")

# ===========================================================================
# TAB 2 — Financial Statement Forecast
# ===========================================================================

with tab_forecast:
    current_ticker = ticker if analyze_btn else st.session_state.last_ticker

    if not current_ticker:
        st.info("Enter a ticker symbol and click **Analyse** to get started.")
    else:
        with st.spinner(f"Loading financial statements for {current_ticker}..."):
            try:
                fin_data = load_financials(current_ticker, quarterly=quarterly)
                st.session_state.financials = fin_data
            except Exception as exc:
                st.error(f"Could not load financials: {exc}")
                st.stop()

        period_label = "quarters" if quarterly else "years"
        st.markdown(
            f"Forecasting the next **{fin_periods} {period_label}** based on "
            f"historical trends and average growth rates."
        )

        with st.spinner("Calculating forecasts..."):
            try:
                fc = FinancialForecaster(current_ticker, fin_data)
                forecast_report = fc.generate_forecast_report(periods=fin_periods)
                st.session_state.fin_forecast_report = forecast_report
            except Exception as exc:
                st.error(f"Forecast error: {exc}")

        # Quick-view charts for 3 key metrics
        if st.session_state.fin_forecast_report:
            st.markdown("### Key Metrics at a Glance")
            key_items = ["Total Revenue", "Net Income", "Operating Cash Flow"]
            chart_cols = st.columns(len(key_items))

            for col_widget, item in zip(chart_cols, key_items):
                result = fc.get_regression_forecast(item, periods=fin_periods)
                if result is None:
                    col_widget.caption(f"{item}: insufficient data")
                    continue

                hist  = result["historical"]
                best  = result["best_estimate"]
                hist_x = [f"H{i+1}" for i in range(len(hist))]
                fore_x = [f"F{i+1}" for i in range(len(best))]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_x, y=[v/1e9 for v in hist],
                    mode="lines+markers", name="Historical",
                    line=dict(color="#555", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=fore_x, y=[v/1e9 for v in best],
                    mode="lines+markers", name="Forecast",
                    line=dict(color="#1f77b4", width=2.5, dash="dot"),
                ))
                fig.update_layout(
                    title=dict(text=item, font=dict(size=13)),
                    yaxis_title="$B",
                    height=260,
                    margin=dict(l=10, r=10, t=40, b=20),
                    showlegend=False,
                )
                col_widget.plotly_chart(fig, use_container_width=True)
                # Show direction badge
                if result["avg_growth_pct"] is not None:
                    g = result["avg_growth_pct"]
                    col_widget.caption(
                        f"{'📈' if g >= 0 else '📉'} Avg growth: {g:+.1f}% per period"
                    )

            # Full report
            st.markdown("---")
            st.markdown("### Full Forecast Report")
            st.markdown(st.session_state.fin_forecast_report)

# ===========================================================================
# TAB 3 — Deep Dive (Statistical Analysis)
# ===========================================================================

with tab_stats:
    current_ticker = ticker if analyze_btn else st.session_state.last_ticker

    if not current_ticker:
        st.info("Enter a ticker symbol and click **Analyse** to get started.")
    else:
        try:
            fin_data = load_financials(current_ticker, quarterly=quarterly)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        fc = FinancialForecaster(current_ticker, fin_data)
        all_items = (
            FinancialForecaster._INCOME_ITEMS
            + FinancialForecaster._BALANCE_ITEMS
            + FinancialForecaster._CASHFLOW_ITEMS
        )
        selected = st.selectbox("Choose a metric to analyse:", all_items)
        result = fc.get_regression_forecast(selected, periods=fin_periods)

        if result is None:
            st.warning(
                f"Not enough historical data for '{selected}'. "
                "Try switching to quarterly reports in the sidebar."
            )
        else:
            s     = result["stats"]
            hist  = result["historical"]
            best  = result["best_estimate"]
            trend = result["trend_forecast"]
            growth= result["growth_forecast"]

            st.markdown(f"## {selected}")
            st.markdown(f"*{result['trend_summary']}*")

            # ---- Key numbers ----
            st.markdown("### Historical Snapshot")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average",       _fmt_val(s["average"]))
            c2.metric("Median",        _fmt_val(s["median"]))
            c3.metric("Volatility",    f"{s['volatility_%']:.1f}%",
                      help="How much this metric varies year-to-year. Lower = more stable.")
            c4.metric("Avg Growth/Period",
                      f"{result['avg_growth_pct']:+.1f}%" if result["avg_growth_pct"] is not None else "N/A")

            # Typical range
            st.info(
                f"**Typical range:** {_fmt_val(s['typical_range_low'])} – "
                f"{_fmt_val(s['typical_range_high'])}  \n"
                f"About 95% of historical values fall within this band."
            )

            # Skewness plain-English
            sk = s["skewness"]
            if sk > 0.5:
                skew_msg = "Results tend to skew high — occasional large positive figures pull the average up."
            elif sk < -0.5:
                skew_msg = "Results tend to skew low — occasional weak periods pull the average down."
            else:
                skew_msg = "Results are fairly evenly distributed around the average."
            st.caption(f"Distribution: {skew_msg}")

            # ---- Chart ----
            st.markdown("### Historical & Forecast")
            hist_x = list(range(1, len(hist) + 1))
            fore_x = list(range(len(hist) + 1, len(hist) + len(best) + 1))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist, mode="lines+markers",
                name="Historical", line=dict(color="#444", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=fore_x, y=trend, mode="lines",
                name="Trend estimate",
                line=dict(color="#5b9bd5", dash="dot", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=fore_x, y=growth, mode="lines",
                name="Growth estimate",
                line=dict(color="#ed7d31", dash="dash", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=fore_x, y=best, mode="lines+markers",
                name="Best estimate",
                line=dict(color="#1f77b4", width=2.5),
            ))
            fig.add_vline(
                x=len(hist) + 0.5, line_dash="dash", line_color="gray",
                annotation_text="Forecast starts →",
            )
            fig.update_layout(
                yaxis_title="Value (USD)", xaxis_title="Period",
                height=400, legend=dict(orientation="h", yanchor="bottom", y=1.0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Forecast table ----
            st.markdown("### Forecast Numbers")
            fore_df = pd.DataFrame({
                "Period":          [f"Period +{i+1}" for i in range(len(best))],
                "Trend Estimate":  [_fmt_val(v) for v in trend],
                "Growth Estimate": [_fmt_val(v) for v in growth],
                "Best Estimate":   [_fmt_val(v) for v in best],
            })
            st.table(fore_df)

            # ---- Model accuracy ----
            if result["r_squared"] is not None:
                r2 = result["r_squared"]
                if r2 >= 0.85:
                    confidence = "high confidence — the trend fits the data well"
                elif r2 >= 0.5:
                    confidence = "moderate confidence — some variability not explained by the trend"
                else:
                    confidence = "low confidence — this metric is volatile and hard to predict"
                st.caption(f"Forecast confidence: **{confidence}** (R² = {r2:.0%})")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption("Data provided by Yahoo Finance. For informational purposes only.")
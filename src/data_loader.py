import yfinance as yf
import pandas as pd


class DataLoader:

    def load_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical stock data and normalise yfinance MultiIndex columns."""
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No price data returned for {ticker}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing expected columns for {ticker}: {missing}")

        if len(data) < 60:
            raise ValueError(
                f"Insufficient data for {ticker}: only {len(data)} rows (need >= 60)"
            )

        return data

    def get_company_info(self, ticker: str) -> dict:
        """Get company information, returning an empty dict on failure."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            return info
        except Exception:
            return {}

    def get_financial_statements(self, ticker: str, quarterly: bool = False) -> dict:
        """
        Get financial statements.

        Parameters
        ----------
        quarterly : if True fetch quarterly reports (more data points for regression);
                    if False fetch annual reports
        """
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        if quarterly:
            income_stmt = stock.quarterly_financials
            balance     = stock.quarterly_balance_sheet
            cashflow    = stock.quarterly_cashflow
        else:
            income_stmt = stock.financials
            balance     = stock.balance_sheet
            cashflow    = stock.cashflow

        def _sort_asc(df):
            """Sort columns ascending (oldest first) — needed for OLS t-index."""
            if df is not None and not df.empty:
                return df.sort_index(axis=1)
            return df

        financials = {
            "income_statement": _sort_asc(income_stmt),
            "balance_sheet":    _sort_asc(balance),
            "cash_flow":        _sort_asc(cashflow),
            "quarterly":        quarterly,
        }

        # Flat metrics dict consumed by ReportGenerator
        financials["metrics"] = {
            "peRatio":        info.get("trailingPE"),
            "priceToBook":    info.get("priceToBook"),
            "debtToEquity":   info.get("debtToEquity"),
            "returnOnEquity": _pct(info.get("returnOnEquity")),
            "profitMargins":  _pct(info.get("profitMargins")),
        }

        return financials


def _pct(value):
    if value is None:
        return "N/A"
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return "N/A"
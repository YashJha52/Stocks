"""
train_models.py  —  offline training script
Run from the project root:  python scripts/train_models.py
"""
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path so 'src' is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import StatisticalFeatures
from src.models import StockPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Download stock data with up to 3 retries."""
    for attempt in range(1, 4):
        try:
            print(f"  Downloading {ticker} (attempt {attempt})...")
            data = yf.download(ticker, start=start_date, end=end_date,
                               progress=False, auto_adjust=True)

            # Flatten MultiIndex columns returned by recent yfinance versions
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not data.empty and len(data) >= 60:
                return data

            print(f"  Insufficient rows ({len(data)}), retrying...")

        except Exception as exc:
            print(f"  Error: {exc}, retrying...")

        time.sleep(3)

    return None


def prepare_features(stock_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target series y.

    Target: next-day close (shift -1 so we predict tomorrow from today's features).
    Drop rows with NaNs only AFTER aligning X and y to avoid look-ahead leakage.
    """
    fe = StatisticalFeatures()
    features = fe.compute_features(stock_data)

    # Target is tomorrow's closing price
    features["target"] = stock_data["Close"].shift(-1)

    # Drop rows that have any NaN in features or target
    features = features.dropna()

    X = features.drop(columns=["target"])
    y = features["target"]
    return X, y


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_models(
    tickers: list[str] | None = None,
    lookback_years: int = 3,
    test_size: float = 0.2,
) -> None:
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print("Training ML models\n")

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=int(365 * lookback_years))

    for ticker in tickers:
        print("=" * 50)
        print(f"  {ticker}")
        print("=" * 50)

        try:
            # ---- 1. Download ----
            stock_data = download_stock_data(
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if stock_data is None:
                print(f"  Skipping {ticker}: could not download data\n")
                continue

            print(f"  Data rows: {len(stock_data)}")

            # ---- 2. Feature engineering ----
            X, y = prepare_features(stock_data)
            print(f"  Feature matrix: {X.shape}")

            # ---- 3. Temporal train / test split (no leakage) ----
            split_idx   = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            print(f"  Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

            # ---- 4. Train ----
            predictor = StockPredictor()
            predictor.train_models(X_train, y_train)

            # ---- 5. Evaluate on held-out test set ----
            y_pred  = predictor.predict(X_test)
            metrics = predictor.evaluate(y_test, y_pred)

            print("\n  Out-of-sample performance:")
            print(metrics.to_string())

            # ---- 6. Retrain on full dataset for deployment ----
            predictor_full = StockPredictor()
            predictor_full.train_models(X, y)

            # ---- 7. Save ----
            model_dir = Path("models") / "saved" / ticker
            predictor_full.save(model_dir)
            print(f"\n  Models saved to {model_dir}\n")

        except Exception as exc:
            print(f"  ERROR training {ticker}: {exc}\n")

        time.sleep(2)   # be polite to the Yahoo Finance API


if __name__ == "__main__":
    train_models()
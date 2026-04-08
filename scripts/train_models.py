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

# Add project root to path so 'src' is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import StatisticalFeatures
from src.models import StockPredictor


def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    for attempt in range(1, 4):
        try:
            print(f"  Downloading {ticker} (attempt {attempt})...")
            data = yf.download(ticker, start=start_date, end=end_date,
                               progress=False, auto_adjust=True)
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
    """Build feature matrix X and binary target series y."""
    fe = StatisticalFeatures()
    features = fe.compute_features(stock_data)

    # Target is 1 if tomorrow's Close is strictly greater than today's Close, else 0
    tomorrow_close = stock_data["Close"].shift(-1)
    target_series = (tomorrow_close > stock_data["Close"]).astype(float)
    
    # The very last day doesn't have a "tomorrow", so we must make it NaN before dropping
    target_series.iloc[-1] = np.nan 
    features["target"] = target_series

    # Drop rows that have any NaN in features or target
    features = features.dropna()

    X = features.drop(columns=["target"])
    y = features["target"].astype(int) # Convert target back to clean integers (0, 1)
    return X, y


def train_models(tickers: list[str] | None = None, lookback_years: int = 3, test_size: float = 0.2) -> None:
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print("Training Classification ML models\n")
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=int(365 * lookback_years))

    for ticker in tickers:
        print("=" * 50)
        print(f"  {ticker}")
        print("=" * 50)

        try:
            stock_data = download_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if stock_data is None:
                continue

            X, y = prepare_features(stock_data)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            predictor = StockPredictor()
            predictor.train_models(X_train, y_train)

            # Evaluate out-of-sample
            y_pred = predictor.predict(X_test)
            metrics_df, conf_matrices = predictor.evaluate(y_test, y_pred)

            print("\n  Out-of-sample performance:")
            print(metrics_df.to_string())

            # Save Metrics to TXT file
            model_dir = Path("models") / "saved" / ticker
            model_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_path = model_dir / "classification_results.txt"
            with open(metrics_path, "w") as f:
                f.write(f"--- Classification Evaluation Metrics for {ticker} ---\n\n")
                f.write(metrics_df.to_string())
                f.write("\n\n--- Confusion Matrices ---\n")
                
                for name in metrics_df.index:
                    cm = conf_matrices[name]
                    f.write(f"\n[{name.upper()}]\n")
                    f.write(f"True Negatives  (Predicted Down, Actual Down) : {cm[0][0]}\n")
                    f.write(f"False Positives (Predicted Up, Actual Down)   : {cm[0][1]}\n")
                    f.write(f"False Negatives (Predicted Down, Actual Up)   : {cm[1][0]}\n")
                    f.write(f"True Positives  (Predicted Up, Actual Up)     : {cm[1][1]}\n")
            
            print(f"  Metrics saved to {metrics_path}")

            # Retrain on full dataset for deployment
            predictor_full = StockPredictor()
            predictor_full.train_models(X, y)
            predictor_full.save(model_dir)

        except Exception as exc:
            print(f"  ERROR training {ticker}: {exc}\n")
        time.sleep(2)

if __name__ == "__main__":
    train_models()
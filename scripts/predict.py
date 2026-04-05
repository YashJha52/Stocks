"""
predict.py  —  CLI prediction script
Run from the project root:  python scripts/predict.py [TICKER] [--days N]
"""
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import StatisticalFeatures
from src.models import StockPredictor


def predict_stock(ticker: str, days: int = 30) -> pd.DataFrame | None:
    """
    Make an iterative multi-step forecast for *ticker* over *days* trading days.

    Strategy
    --------
    Rather than repeating a single prediction (the original flat-line bug), we
    use a walk-forward simulation:
      1. Load recent OHLCV history.
      2. For each future day, compute features on the latest window, predict
         the next close, then append the predicted close to the history so
         subsequent steps can build on it.

    This is still a simplified approach (we only have a Close estimate for
    synthetic future bars) but it produces a genuine forecast trajectory
    instead of a horizontal line.
    """
    print(f"\nPredicting {days} days ahead for {ticker} ...")

    # ---- 1. Load saved models ----
    model_dir = Path("models") / "saved" / ticker
    if not model_dir.exists():
        print(f"  No saved models found at {model_dir}. Run train_models.py first.")
        return None

    try:
        predictor = StockPredictor.load(model_dir)
    except Exception as exc:
        print(f"  Failed to load models: {exc}")
        return None

    # ---- 2. Load recent history (1 year is enough for feature windows) ----
    data_loader = DataLoader()
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

    try:
        stock_data = data_loader.load_stock_data(ticker, start_date, end_date)
    except ValueError as exc:
        print(f"  Data error: {exc}")
        return None

    # ---- 3. Walk-forward simulation ----
    fe = StatisticalFeatures()

    # We'll extend this buffer with synthetic rows as we go
    history = stock_data[["Open", "High", "Low", "Close", "Volume"]].copy()

    forecast_dates: list[datetime] = []
    forecast_values: dict[str, list[float]] = {
        name: [] for name in list(predictor.trained_models.keys()) + ["ensemble"]
    }

    last_date = history.index[-1]

    for step in range(days):
        # Compute features on current history window
        features = fe.compute_features(history).dropna()

        if features.empty:
            print(f"  Not enough data to forecast step {step + 1}")
            break

        last_row = features.iloc[[-1]]

        # Predict
        pred_df = predictor.predict(last_row)
        pred_row = pred_df.iloc[0]

        # Record predictions
        next_date = _next_business_day(last_date, step + 1)
        forecast_dates.append(next_date)
        for col in forecast_values:
            forecast_values[col].append(float(pred_row[col]))

        # Append synthetic row so next step can use it
        ensemble_price = float(pred_row["ensemble"])
        new_row = pd.DataFrame(
            {
                "Open":   [ensemble_price],
                "High":   [ensemble_price * 1.005],
                "Low":    [ensemble_price * 0.995],
                "Close":  [ensemble_price],
                "Volume": [history["Volume"].rolling(20).mean().iloc[-1]],
            },
            index=[next_date],
        )
        history = pd.concat([history, new_row])

    if not forecast_dates:
        return None

    pred_df = pd.DataFrame(forecast_values, index=forecast_dates)
    return pred_df


def _next_business_day(base_date: datetime, offset: int) -> datetime:
    """Return the *offset*-th business day after *base_date*."""
    count = 0
    current = base_date
    while count < offset:
        current += timedelta(days=1)
        if current.weekday() < 5:   # Mon-Fri
            count += 1
    return current


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stock price forecaster")
    parser.add_argument("ticker", nargs="?", default="AAPL",
                        help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of trading days to forecast (default: 30)")
    args = parser.parse_args()

    predictions = predict_stock(args.ticker.upper(), args.days)

    if predictions is not None:
        print(f"\nForecast for {args.ticker.upper()} ({args.days} trading days):")
        print(predictions[["ensemble"]].to_string())
    else:
        print("Prediction failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
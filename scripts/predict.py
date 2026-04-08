"""
predict.py  —  CLI prediction script
"""
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import StatisticalFeatures
from src.models import StockPredictor

def predict_stock(ticker: str, days: int = 30) -> pd.DataFrame | None:
    print(f"\nPredicting {days} days ahead signals for {ticker} ...")

    model_dir = Path("models") / "saved" / ticker
    if not model_dir.exists():
        print(f"  No saved models found at {model_dir}. Run train_models.py first.")
        return None

    try:
        predictor = StockPredictor.load(model_dir)
    except Exception as exc:
        print(f"  Failed to load models: {exc}")
        return None

    data_loader = DataLoader()
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

    try:
        stock_data = data_loader.load_stock_data(ticker, start_date, end_date)
    except ValueError as exc:
        print(f"  Data error: {exc}")
        return None

    fe = StatisticalFeatures()
    history = stock_data[["Open", "High", "Low", "Close", "Volume"]].copy()

    forecast_dates = []
    forecast_values = {name: [] for name in list(predictor.trained_models.keys()) + ["ensemble"]}

    last_date = history.index[-1]

    for step in range(days):
        features = fe.compute_features(history).dropna()
        if features.empty:
            break

        last_row = features.iloc[[-1]]
        pred_df = predictor.predict(last_row)
        pred_row = pred_df.iloc[0]

        next_date = _next_business_day(last_date, step + 1)
        forecast_dates.append(next_date)
        
        for col in forecast_values:
            # Append "UP" or "DOWN" instead of raw 1s and 0s for readability
            signal = int(pred_row[col])
            forecast_values[col].append("UP" if signal == 1 else "DOWN")

        # Walk-forward assumption: Simulate a +/- 0.5% move so MA features don't break
        ensemble_signal = int(pred_row["ensemble"])
        last_close = history["Close"].iloc[-1]
        simulated_next_close = last_close * 1.005 if ensemble_signal == 1 else last_close * 0.995

        new_row = pd.DataFrame({
            "Open":   [last_close],
            "High":   [simulated_next_close * 1.01],
            "Low":    [simulated_next_close * 0.99],
            "Close":  [simulated_next_close],
            "Volume": [history["Volume"].rolling(20).mean().iloc[-1]],
        }, index=[next_date])
        history = pd.concat([history, new_row])

    if not forecast_dates:
        return None

    return pd.DataFrame(forecast_values, index=forecast_dates)

def _next_business_day(base_date: datetime, offset: int) -> datetime:
    count = 0
    current = base_date
    while count < offset:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return current

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    predictions = predict_stock(args.ticker.upper(), args.days)

    if predictions is not None:
        print(f"\nSignal Forecast for {args.ticker.upper()} ({args.days} trading days):")
        print(predictions[["ensemble"]].to_string())
    else:
        print("Prediction failed.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_loader import DataLoader
from src.feature_engineering import StatisticalFeatures

def predict_stock(ticker, days=30):
    """Make predictions for a stock"""
    
    print(f"Predicting for {ticker}...")
    
    # Load models
    model_dir = f"models/saved/{ticker}"
    if not os.path.exists(model_dir):
        print(f"No models found for {ticker}")
        return None
    
    models = {}
    for name in ['linear', 'ridge', 'rf', 'gb', 'xgb']:
        model_path = f"{model_dir}/{name}.pkl"
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
    
    scaler = joblib.load(f"{model_dir}/feature_scaler.pkl")
    price_scaler = joblib.load(f"{model_dir}/price_scaler.pkl")
    
    # Load recent data
    data_loader = DataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    stock_data = data_loader.load_stock_data(ticker, start_date, end_date)
    
    # Compute features
    fe = StatisticalFeatures()
    features = fe.compute_features(stock_data)
    features = features.dropna()
    
    # Make prediction
    last_features = features.iloc[-1:].copy()
    X_scaled = scaler.transform(last_features)
    
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_scaled)
        predictions[name] = price_scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
    
    # Create future dates
    future_dates = [
        datetime.now() + timedelta(days=i) 
        for i in range(1, days+1)
    ]
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame(index=future_dates)
    for name, pred in predictions.items():
        pred_df[name] = [pred] * days
    
    pred_df['ensemble'] = pred_df.mean(axis=1)
    
    return pred_df

if __name__ == "__main__":
    ticker = "AAPL"
    predictions = predict_stock(ticker)
    if predictions is not None:
        print(f"\nPredictions for {ticker}:")
        print(predictions[['ensemble']].head())
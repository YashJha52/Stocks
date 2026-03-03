#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_loader import DataLoader
from src.feature_engineering import StatisticalFeatures
from models import StockPredictor

def train_models(tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']):
    """Train models for specified tickers"""
    
    print("Training ML models...")
    
    # Initialize components
    data_loader = DataLoader()
    feature_engineer = StatisticalFeatures()
    predictor = StockPredictor()
    
    for ticker in tickers:
        print(f"\nTraining for {ticker}...")
        
        try:
            # Load data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - pd.DateOffset(days=365*3)).strftime('%Y-%m-%d')
            
            stock_data = data_loader.load_stock_data(ticker, start_date, end_date)
            
            if stock_data.empty:
                print(f"No data for {ticker}")
                continue
            
            # Compute features
            features = feature_engineer.compute_features(stock_data)
            features['Close'] = stock_data['Close']
            features = features.dropna()
            
            # Train models
            X = features.drop('Close', axis=1)
            y = features['Close']
            predictor.train_models(X, y)
            
            # Save models
            model_dir = f"models/saved/{ticker}"
            os.makedirs(model_dir, exist_ok=True)
            
            for name, model in predictor.trained_models.items():
                joblib.dump(model, f"{model_dir}/{name}.pkl")
            
            joblib.dump(predictor.scaler, f"{model_dir}/feature_scaler.pkl")
            joblib.dump(predictor.price_scaler, f"{model_dir}/price_scaler.pkl")
            
            with open(f"{model_dir}/features.json", 'w') as f:
                json.dump(list(X.columns), f)
            
            print(f"✓ Models saved for {ticker}")
            
        except Exception as e:
            print(f"✗ Error training {ticker}: {e}")

if __name__ == "__main__":
    train_models()
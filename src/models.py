import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class StockPredictor:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42),
            'lstm': None
        }
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
    
    def train_models(self, X, y):
        """Train all models"""
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.price_scaler.fit_transform(y.values.reshape(-1, 1))
        
        for name, model in self.models.items():
            if name != 'lstm':
                model.fit(X_scaled, y_scaled.ravel())
                self.trained_models[name] = model
    
    def train_lstm(self, X, y, epochs=50, batch_size=32):
        """Train LSTM model"""
        # Prepare data for LSTM
        X_lstm, y_lstm = self.prepare_data(pd.concat([X, y], axis=1))
        
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        model.fit(X_lstm, y_lstm, epochs=epochs, batch_size=batch_size, verbose=0)
        self.models['lstm'] = model
    
    def prepare_data(self, data, lookback=60):
        """Prepare data for LSTM"""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, -1])  # Target is the last column (Close price)
        
        return np.array(X), np.array(y)
    
    def predict(self, X):
        """Make predictions with all models"""
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.trained_models.items():
            pred = model.predict(X_scaled)
            predictions[name] = self.price_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        
        return pd.DataFrame(predictions)
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance"""
        metrics = {}
        for name in y_pred.columns:
            mse = mean_squared_error(y_true, y_pred[name])
            mae = mean_absolute_error(y_true, y_pred[name])
            mape = np.mean(np.abs((y_true - y_pred[name]) / y_true)) * 100
            metrics[name] = {'MSE': mse, 'MAE': mae, 'MAPE': mape}
        
        return pd.DataFrame(metrics).T
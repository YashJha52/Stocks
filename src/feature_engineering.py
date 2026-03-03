import pandas as pd
import numpy as np
from scipy import stats

class StatisticalFeatures:
    def compute_features(self, data):
        """Compute statistical features from stock data"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moving averages
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['ema_12'] = data['Close'].ewm(span=12).mean()
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        
        # Statistical measures
        features['mean_5'] = data['Close'].rolling(5).mean()
        features['std_5'] = data['Close'].rolling(5).std()
        features['skew_5'] = data['Close'].rolling(5).skew()
        features['kurt_5'] = data['Close'].rolling(5).kurt()
        
        return features
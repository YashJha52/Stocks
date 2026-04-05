import pandas as pd
import numpy as np


class StatisticalFeatures:

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical and statistical features from OHLCV data.

        All features are expressed as returns, ratios, or normalised values so
        that the same model can generalise across tickers with very different
        absolute price levels (e.g. AAPL vs BRK.A).
        """
        features = pd.DataFrame(index=data.index)
        close = data["Close"]

        # ------------------------------------------------------------------
        # Returns
        # ------------------------------------------------------------------
        features["returns"]     = close.pct_change()
        features["log_returns"] = np.log(close / close.shift(1))

        # Multi-period returns
        features["returns_5d"]  = close.pct_change(5)
        features["returns_20d"] = close.pct_change(20)

        # ------------------------------------------------------------------
        # Moving-average ratios  (price / MA - 1  =>  dimensionless)
        # ------------------------------------------------------------------
        sma_5  = close.rolling(5).mean()
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        features["price_to_sma5"]  = close / sma_5  - 1
        features["price_to_sma10"] = close / sma_10 - 1
        features["price_to_sma20"] = close / sma_20 - 1
        features["sma5_to_sma20"]  = sma_5 / sma_20 - 1

        # MACD (already dimensionless as a ratio of price)
        features["macd"] = (ema_12 - ema_26) / close

        # ------------------------------------------------------------------
        # Volatility  (rolling std of returns — already dimensionless)
        # ------------------------------------------------------------------
        features["volatility_5d"]  = features["returns"].rolling(5).std()
        features["volatility_20d"] = features["returns"].rolling(20).std()

        # ------------------------------------------------------------------
        # RSI  (0-100 oscillator)
        # ------------------------------------------------------------------
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        features["rsi"] = 100 - (100 / (1 + rs))

        # ------------------------------------------------------------------
        # Bollinger Band width and %B  (both dimensionless)
        # ------------------------------------------------------------------
        std_20 = close.rolling(20).std()
        bb_upper = sma_20 + std_20 * 2
        bb_lower = sma_20 - std_20 * 2
        bb_width = (bb_upper - bb_lower) / sma_20          # width / mid
        features["bb_width"] = bb_width
        features["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower)

        # ------------------------------------------------------------------
        # Volume features
        # ------------------------------------------------------------------
        if "Volume" in data.columns and data["Volume"].sum() > 0:
            vol_ma20 = data["Volume"].rolling(20).mean()
            features["volume_ratio"] = data["Volume"] / vol_ma20.replace(0, np.nan)
        else:
            features["volume_ratio"] = np.nan

        # ------------------------------------------------------------------
        # Higher-moment statistics of returns (already dimensionless)
        # ------------------------------------------------------------------
        features["skew_5d"]  = features["returns"].rolling(5).skew()
        features["kurt_5d"]  = features["returns"].rolling(5).kurt()
        features["skew_20d"] = features["returns"].rolling(20).skew()

        return features
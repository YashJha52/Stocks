import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


class StockPredictor:
    """
    Trains and evaluates an ensemble of sklearn regressors to predict
    the next-day closing price.

    Changes from original:
    - Removed LSTM (None entry that crashed predict())
    - Added feature-column alignment guard so predict() never gets a
      shape mismatch after dropna produces different columns at runtime
    - price_scaler now fits only on y (not on feature+target concat)
    - evaluate() works on original price scale
    """

    _MODEL_NAMES = ["linear", "ridge", "rf", "gb", "xgb"]

    def __init__(self):
        self.models: dict = {
            "linear": LinearRegression(),
            "ridge":  Ridge(alpha=1.0),
            "rf":     RandomForestRegressor(n_estimators=100, random_state=42),
            "gb":     GradientBoostingRegressor(n_estimators=100, random_state=42),
            "xgb":    XGBRegressor(n_estimators=100, random_state=42,
                                   verbosity=0, eval_metric="rmse"),
        }
        self.trained_models: dict = {}
        self.feature_columns: list[str] = []   # saved for alignment in predict()
        self.scaler       = StandardScaler()
        self.price_scaler = MinMaxScaler()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train all models.

        X must NOT contain the target column — callers are responsible for
        passing a feature-only DataFrame (no 'Close' column).
        """
        if "Close" in X.columns:
            raise ValueError(
                "'Close' found in feature matrix X. "
                "Drop the target column before calling train_models()."
            )

        self.feature_columns = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.price_scaler.fit_transform(
            y.values.reshape(-1, 1)
        ).ravel()

        for name, model in self.models.items():
            model.fit(X_scaled, y_scaled)
            self.trained_models[name] = model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame of predictions (original price scale) with one
        column per model plus an 'ensemble' column (row-wise mean).
        """
        if not self.trained_models:
            raise RuntimeError("No trained models found. Call train_models() first.")

        # Align columns to what was seen during training
        X = X[self.feature_columns]

        X_scaled = self.scaler.transform(X)

        preds = {}
        for name, model in self.trained_models.items():
            scaled_pred = model.predict(X_scaled).reshape(-1, 1)
            preds[name] = self.price_scaler.inverse_transform(
                scaled_pred
            ).flatten()

        df = pd.DataFrame(preds, index=X.index)
        df["ensemble"] = df.mean(axis=1)
        return df

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MSE, MAE and MAPE for each model column in y_pred.
        Both y_true and y_pred must be in the original price scale.
        """
        metrics: dict[str, dict] = {}
        for name in y_pred.columns:
            mse  = mean_squared_error(y_true, y_pred[name])
            mae  = mean_absolute_error(y_true, y_pred[name])
            # Guard against zero prices
            safe_true = y_true.replace(0, np.nan)
            mape = float(
                np.nanmean(np.abs((safe_true - y_pred[name]) / safe_true)) * 100
            )
            metrics[name] = {"MSE": round(mse, 4), "MAE": round(mae, 4),
                              "MAPE%": round(mape, 2)}

        return pd.DataFrame(metrics).T

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path) -> None:
        """Save all artefacts to *model_dir*."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.trained_models.items():
            joblib.dump(model, model_dir / f"{name}.pkl")

        joblib.dump(self.scaler,          model_dir / "feature_scaler.pkl")
        joblib.dump(self.price_scaler,    model_dir / "price_scaler.pkl")
        joblib.dump(self.feature_columns, model_dir / "feature_columns.pkl")

    @classmethod
    def load(cls, model_dir: str | Path) -> "StockPredictor":
        """Restore a saved predictor from *model_dir*."""
        model_dir = Path(model_dir)
        predictor = cls()

        predictor.scaler          = joblib.load(model_dir / "feature_scaler.pkl")
        predictor.price_scaler    = joblib.load(model_dir / "price_scaler.pkl")
        predictor.feature_columns = joblib.load(model_dir / "feature_columns.pkl")

        for name in cls._MODEL_NAMES:
            path = model_dir / f"{name}.pkl"
            if path.exists():
                predictor.trained_models[name] = joblib.load(path)

        if not predictor.trained_models:
            raise FileNotFoundError(f"No model .pkl files found in {model_dir}")

        return predictor
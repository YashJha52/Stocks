import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class StockPredictor:
    """
    Trains and evaluates an ensemble of sklearn classifiers to predict
    if the next-day closing price will go UP (1) or DOWN (0).
    """

    _MODEL_NAMES = ["logistic", "ridge", "rf", "gb", "xgb"]

    def __init__(self):
        self.models: dict = {
            "logistic": LogisticRegression(max_iter=1000),
            "ridge":  RidgeClassifier(),
            "rf":     RandomForestClassifier(n_estimators=100, random_state=42),
            "gb":     GradientBoostingClassifier(n_estimators=100, random_state=42),
            "xgb":    XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss"),
        }
        self.trained_models: dict = {}
        self.feature_columns: list[str] = []
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all models on binary targets (0 or 1)."""
        if "Close" in X.columns:
            raise ValueError(
                "'Close' found in feature matrix X. "
                "Drop the target column before calling train_models()."
            )

        self.feature_columns = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        for name, model in self.models.items():
            model.fit(X_scaled, y.values)
            self.trained_models[name] = model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of binary predictions (0 or 1)."""
        if not self.trained_models:
            raise RuntimeError("No trained models found. Call train_models() first.")

        # Align columns to what was seen during training
        X = X[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        preds = {}
        for name, model in self.trained_models.items():
            preds[name] = model.predict(X_scaled)

        df = pd.DataFrame(preds, index=X.index)
        
        # Majority voting for the ensemble (Mode)
        df["ensemble"] = df.mode(axis=1)[0].astype(int)
        return df

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, y_true: pd.Series, y_pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Compute Accuracy, F1 Score, and Confusion Matrix."""
        metrics: dict[str, dict] = {}
        conf_matrices: dict[str, np.ndarray] = {}

        for name in y_pred.columns:
            acc = accuracy_score(y_true, y_pred[name])
            f1  = f1_score(y_true, y_pred[name], average='weighted')
            cm  = confusion_matrix(y_true, y_pred[name])

            metrics[name] = {"Accuracy": round(acc, 4), "F1 Score": round(f1, 4)}
            conf_matrices[name] = cm

        return pd.DataFrame(metrics).T, conf_matrices

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.trained_models.items():
            joblib.dump(model, model_dir / f"{name}.pkl")

        joblib.dump(self.scaler,          model_dir / "feature_scaler.pkl")
        joblib.dump(self.feature_columns, model_dir / "feature_columns.pkl")

    @classmethod
    def load(cls, model_dir: str | Path) -> "StockPredictor":
        model_dir = Path(model_dir)
        predictor = cls()

        predictor.scaler          = joblib.load(model_dir / "feature_scaler.pkl")
        predictor.feature_columns = joblib.load(model_dir / "feature_columns.pkl")

        for name in cls._MODEL_NAMES:
            path = model_dir / f"{name}.pkl"
            if path.exists():
                predictor.trained_models[name] = joblib.load(path)

        if not predictor.trained_models:
            raise FileNotFoundError(f"No model .pkl files found in {model_dir}")

        return predictor
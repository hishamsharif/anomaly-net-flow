"""
Isolation Forest anomaly detector.

Lightweight sklearn alternative to MLPAutoencoder. No GPU/PyTorch needed.
Trains in seconds. Fine-tune = full retrain (trees have no weight carry-over).
Per-feature explainability: SHAP TreeExplainer (same API as XGBoost).
Score convention: higher = more anomalous (negated from IF's decision_function).
"""
from __future__ import annotations
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest as _IF

from core.base import PreprocessingSpec


class IsolationForestDetector:
    """sklearn Isolation Forest matching the BaseAnomalyDetector protocol."""

    preprocessing = PreprocessingSpec(
        needs_scaling=False,
        preferred_scalers=("none",),
        supervised=False,
    )

    def __init__(self, n_estimators: int = 200, max_samples: str | int = "auto",
                 contamination: float = 0.01, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self._model: _IF | None = None
        self.threshold: float = 0.0

    def fit(self, X_benign: np.ndarray) -> None:
        self._model = _IF(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_benign)
        self.threshold = float(np.percentile(self.score_batch(X_benign), 95))

    def fine_tune(self, X_incremental: np.ndarray,
                  X_replay: np.ndarray | None = None) -> None:
        X = np.vstack([X_incremental, X_replay]) if X_replay is not None else X_incremental
        self.fit(X)

    def score(self, x: np.ndarray) -> float:
        return float(self.score_batch(x.reshape(1, -1))[0])

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (negated from IF's negative-for-anomaly convention)."""
        return -self._model.decision_function(X)

    def is_anomaly(self, x: np.ndarray) -> tuple[bool, float]:
        s = self.score(x)
        flagged = int(self._model.predict(x.reshape(1, -1))[0]) == -1
        return flagged, s

    def feature_importances(self, X_anomaly: np.ndarray) -> np.ndarray | None:
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            sv = explainer.shap_values(X_anomaly)
            return np.abs(sv).mean(axis=0) if sv.ndim > 1 else np.abs(sv)
        except ImportError:
            return None

    def save(self, path: str) -> None:
        joblib.dump({
            "model":        self._model,
            "threshold":    self.threshold,
            "n_estimators": self.n_estimators,
            "max_samples":  self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }, path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        d = joblib.load(path)
        obj = cls(n_estimators=d["n_estimators"], max_samples=d["max_samples"],
                  contamination=d["contamination"], random_state=d["random_state"])
        obj._model = d["model"]
        obj.threshold = d["threshold"]
        return obj

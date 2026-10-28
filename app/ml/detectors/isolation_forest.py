"""
Isolation Forest anomaly detector.

Lightweight sklearn-based alternative to VanillaAE. No GPU/PyTorch needed.
Trains in seconds even on large datasets. Fine-tune = full retrain (same
story as XGBoost — trees don't carry weights across runs).

Anomaly score uses the IF decision function negated so that higher = more
anomalous (consistent with VanillaAE's MSE convention).
Per-feature explainability: use SHAP TreeExplainer (same API as XGBoost).
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest as _IF


class IsolationForestDetector:
    """
    sklearn Isolation Forest wrapped to match the BaseAnomalyDetector protocol.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: str | int = "auto",
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self._model: _IF | None = None
        self.threshold: float = 0.0    # kept for protocol compatibility; IF uses contamination

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X_benign: np.ndarray) -> None:
        """
        Train on benign-only flows.
        contamination sets the decision boundary; 0.01 assumes ~1% of
        training data may still be noisy/anomalous.
        """
        self._model = _IF(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_benign)
        # Store the decision threshold as the 95th percentile of anomaly scores
        # on benign training data so the protocol attribute is meaningful.
        scores = self.score_batch(X_benign)
        self.threshold = float(np.percentile(scores, 95))

    def fine_tune(self, X_incremental: np.ndarray, X_replay: np.ndarray | None = None) -> None:
        """Full retrain on incremental + replay benign flows (IF has no weight carry-over)."""
        X = np.vstack([X_incremental, X_replay]) if X_replay is not None else X_incremental
        self.fit(X)

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(self, x: np.ndarray) -> float:
        """
        Anomaly score for a single flow (1-D or 2-D array of shape (1, n_features)).
        Higher = more anomalous. Negated from IF's decision_function
        (which is negative for anomalies) so convention matches VanillaAE.
        """
        return float(self.score_batch(x.reshape(1, -1))[0])

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores for a batch. Higher = more anomalous."""
        return -self._model.decision_function(X)   # negate: IF gives negative for anomalies

    def is_anomaly(self, x: np.ndarray) -> tuple[bool, float]:
        s = self.score(x)
        flagged = int(self._model.predict(x.reshape(1, -1))[0]) == -1
        return flagged, s

    def feature_importances(self, X_anomaly: np.ndarray) -> np.ndarray | None:
        """
        SHAP-based per-feature contribution to the anomaly score.
        Requires shap to be installed (pip install shap).
        Returns array of shape (n_features,) or None if shap is unavailable.
        """
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            sv = explainer.shap_values(X_anomaly)
            return np.abs(sv).mean(axis=0) if sv.ndim > 1 else np.abs(sv)
        except ImportError:
            return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump({
            "model": self._model,
            "threshold": self.threshold,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }, path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        data = joblib.load(path)
        obj = cls(
            n_estimators=data["n_estimators"],
            max_samples=data["max_samples"],
            contamination=data["contamination"],
            random_state=data["random_state"],
        )
        obj._model = data["model"]
        obj.threshold = data["threshold"]
        return obj

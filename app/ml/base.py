from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class BaseClassifier(Protocol):
    """Supervised classifier slot in the 3-stage inference pipeline."""

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None: ...

    def predict(self, X: np.ndarray) -> list[tuple[str, float]]:
        """Returns a list of (predicted_label, confidence) — one per row in X."""
        ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "BaseClassifier": ...


@runtime_checkable
class BaseAnomalyDetector(Protocol):
    """Unsupervised anomaly detector slot in the 3-stage inference pipeline."""

    def fit(self, X_benign: np.ndarray) -> None: ...

    def score(self, X: np.ndarray) -> float:
        """Returns reconstruction error; higher = more anomalous."""
        ...

    def is_anomaly(self, X: np.ndarray) -> tuple[bool, float]:
        """Returns (anomaly_flag, score) using the stored threshold."""
        ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "BaseAnomalyDetector": ...

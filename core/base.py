from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable
import numpy as np


@dataclass
class PreprocessingSpec:
    """
    Declared as a class variable on every classifier and detector.
    The training pipeline reads this to build the appropriate preprocessing
    graph without if/else branching on model type names.

    Rules used by the pipeline:
      needs_scaling=False  → skip the scaler stage for this model's data path
      preferred_scalers    → tried in order during hyperparameter search
      supervised=True      → apply SMOTE; train on all classes
      supervised=False     → filter to BENIGN rows only; no SMOTE
    """
    needs_scaling: bool = True
    # Ordered preference for the hyperparameter search loop.
    # ("none",) means scaling is always skipped for this model.
    preferred_scalers: tuple[str, ...] = ("robust",)
    # True  = supervised classifier (XGBoost, RF, …)
    # False = anomaly detector (MLP-AE, Isolation Forest, …)
    supervised: bool = True


@runtime_checkable
class BaseClassifier(Protocol):
    """Supervised multi-class classifier slot in the 3-stage pipeline."""

    preprocessing: ClassVar[PreprocessingSpec]

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> None: ...

    def predict(self, X: np.ndarray) -> list[tuple[str, float]]:
        """Returns [(predicted_label, confidence), …] — one tuple per row."""
        ...

    def feature_importances(self) -> np.ndarray | None: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "BaseClassifier": ...


@runtime_checkable
class BaseAnomalyDetector(Protocol):
    """Unsupervised anomaly detector slot in the 3-stage pipeline."""

    preprocessing: ClassVar[PreprocessingSpec]

    def fit(self, X_benign: np.ndarray) -> None: ...

    def score(self, x: np.ndarray) -> float:
        """Scalar anomaly score — higher = more anomalous."""
        ...

    def is_anomaly(self, x: np.ndarray) -> tuple[bool, float]:
        """Returns (anomaly_flag, score) using the stored threshold."""
        ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "BaseAnomalyDetector": ...

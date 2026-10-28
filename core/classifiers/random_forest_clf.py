from __future__ import annotations
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier as _RF
from sklearn.preprocessing import LabelEncoder

from core.base import PreprocessingSpec


class RandomForestClassifier:
    """
    Supervised classifier — full cumulative retrain each fine-tune cycle.
    Bootstrap aggregation reduces overfitting on small incremental datasets.
    class_weight="balanced" handles CIC-IDS2017 class imbalance without SMOTE overhead.
    Scale-invariant: scaler skipped.
    """

    preprocessing = PreprocessingSpec(
        needs_scaling=False,
        preferred_scalers=("none",),
        supervised=True,
    )

    def __init__(self, n_estimators: int = 300, max_depth: int | None = None,
                 min_samples_leaf: int = 2, n_jobs: int = -1,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: _RF | None = None
        self._encoder = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> None:
        y_enc = self._encoder.fit_transform(y)
        self.classes_ = list(self._encoder.classes_)
        self._model = _RF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self._model.fit(X, y_enc, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> list[tuple[str, float]]:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        proba = self._model.predict_proba(X)
        return [
            (self.classes_[int(np.argmax(p))], float(np.max(p)))
            for p in proba
        ]

    def feature_importances(self) -> np.ndarray | None:
        return self._model.feature_importances_ if self._model else None

    def save(self, path: str) -> None:
        joblib.dump({"model": self._model, "encoder": self._encoder,
                     "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._model = data["model"]
        obj._encoder = data["encoder"]
        obj.classes_ = data.get("classes", list(obj._encoder.classes_))
        return obj

from __future__ import annotations
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier as _RF
from sklearn.preprocessing import LabelEncoder


class RandomForestClassifier:
    """
    Supervised classifier — full cumulative retrain each fine-tune cycle.
    RF has no weight carry-over; fine-tune = retrain from scratch on all
    labelled_flows up to the current savepoint (same story as XGBoost).

    Advantages over XGBoost: lower hyperparameter sensitivity, built-in
    bootstrap aggregation reduces overfitting on small incremental datasets.
    Trade-off: slower inference and larger memory footprint than XGBoost.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: _RF | None = None
        self._encoder = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        y_enc = self._encoder.fit_transform(y)
        self.classes_ = list(self._encoder.classes_)
        self._model = _RF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            class_weight="balanced",   # handles class imbalance like XGBoost's scale_pos_weight
        )
        self._model.fit(X, y_enc, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> list[tuple[str, float]]:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        proba = self._model.predict_proba(X)
        results = []
        for p in proba:
            idx = int(np.argmax(p))
            results.append((self.classes_[idx], float(p[idx])))
        return results

    def feature_importances(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.feature_importances_

    def save(self, path: str) -> None:
        joblib.dump({"model": self._model, "encoder": self._encoder}, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._model = data["model"]
        obj._encoder = data["encoder"]
        obj.classes_ = list(obj._encoder.classes_)
        return obj

from __future__ import annotations
import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from core.base import PreprocessingSpec


class XGBoostClassifier:
    """
    Supervised classifier — full cumulative retrain each fine-tune cycle.
    Trees are additive; no weight carry-over between versions.
    Scale-invariant: VarianceThreshold + correlation pruning applied, scaler skipped.
    """

    preprocessing = PreprocessingSpec(
        needs_scaling=False,
        preferred_scalers=("none",),
        supervised=True,
    )

    def __init__(
        self,
        n_estimators: int = 489,
        max_depth: int = 5,
        learning_rate: float = 0.136,
        colsample_bytree: float = 0.964,
        subsample: float = 0.683,
        min_child_weight: int = 6,
        gamma: float = 0.129,
    ):
        # Parameters selected by RandomizedSearchCV (n_iter=20, cv=3, stratified 100K subsample)
        # on CIC-IDS2017 Mon–Thu training split (10 classes; Heartbleed / SQL-Injection excluded
        # as too rare for CV). Best CV F1(macro) = 0.8423.
        #
        # Key choices:
        #   n_estimators=489  — more trees than the prior 300 default; the search converged toward
        #                        higher values, indicating the 12-class problem benefits from more
        #                        boosting rounds before overfitting on this feature set.
        #   max_depth=5       — slightly shallower than the prior 6; reduces overfitting on the
        #                        heavily imbalanced raw class distribution.
        #   subsample=0.683   — stochastic row subsampling per tree adds regularisation and speeds
        #                        up training on the 2M-row dataset.
        #   colsample_bytree=0.964 — near-full feature sampling; most features are informative
        #                            after VT + correlation pruning so aggressive column dropout hurts.
        #   min_child_weight=6 — higher leaf minimum weight prevents splits on very rare
        #                         attack classes that have few natural representatives.
        #   gamma=0.129       — small but non-zero minimum loss reduction; prunes useless splits
        #                        without cutting genuinely informative ones.
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            min_child_weight=min_child_weight,
            gamma=gamma,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cpu",
        )
        self._model: xgb.XGBClassifier | None = None
        self._encoder = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> None:
        y_enc = self._encoder.fit_transform(y)
        self.classes_ = list(self._encoder.classes_)
        self._model = xgb.XGBClassifier(**self.params, num_class=len(self.classes_))
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
    def load(cls, path: str) -> "XGBoostClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._model = data["model"]
        obj._encoder = data["encoder"]
        obj.classes_ = data.get("classes", list(obj._encoder.classes_))
        return obj

"""
In-memory model store — holds the currently loaded classifier + detector.

Thread-safe: the training thread calls load_from_dir() after training;
FastAPI request handlers call get() for inference.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedModel:
    version_id: int
    artefact_dir: str
    classifier_type: str
    detector_type: str
    classifier: Any           # BaseClassifier instance
    detector: Any             # BaseAnomalyDetector instance
    # Preprocessing pipeline — applied in order at inference time
    variance_threshold: Any   # Step 1: fitted VarianceThreshold
    corr_mask: Any            # Step 2: np.ndarray[bool] | None
    feature_selector: Any     # Step 3: fitted SelectKBest | None
    clf_scaler: Any           # Step 4a: scaler for classifier path | None
    det_scaler: Any           # Step 4b: scaler for detector path | None
    feature_names: list[str]          # all raw input feature names (pre-VT)
    selected_feature_names: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def preprocess_clf(self, X_raw) -> Any:
        """Apply the full feature engineering pipeline for the classifier path."""
        return self._preprocess(X_raw, self.clf_scaler)

    def preprocess_det(self, X_raw) -> Any:
        """Apply the full feature engineering pipeline for the detector path."""
        return self._preprocess(X_raw, self.det_scaler)

    def _preprocess(self, X_raw, scaler) -> Any:
        import numpy as np
        X = self.variance_threshold.transform(X_raw)
        if self.corr_mask is not None:
            X = X[:, self.corr_mask]
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        if scaler is not None:
            X = scaler.transform(X)
        return X


class ModelStore:
    """Singleton. Training thread writes; request handlers read."""

    def __init__(self):
        self._lock = threading.RLock()
        self._model: LoadedModel | None = None

    def load_from_dir(
        self,
        version_id: int,
        artefact_dir: str,
        classifier_type: str,
        detector_type: str,
        metrics: dict,
    ) -> None:
        import joblib, json, os
        from core.registry import get_classifier, get_detector

        clf = get_classifier(classifier_type).load(
            os.path.join(artefact_dir, "classifier.joblib"))
        det = get_detector(detector_type).load(
            os.path.join(artefact_dir, "detector.joblib"))
        vt = joblib.load(os.path.join(artefact_dir, "variance_threshold.joblib"))

        def _load_opt(fname):
            p = os.path.join(artefact_dir, fname)
            return joblib.load(p) if os.path.exists(p) else None

        corr_mask        = _load_opt("corr_mask.joblib")
        feature_selector = _load_opt("feature_selector.joblib")
        clf_scaler       = _load_opt("clf_scaler.joblib")
        det_scaler       = _load_opt("det_scaler.joblib")
        # Backwards-compat: older artefacts have a single scaler.joblib
        if clf_scaler is None and det_scaler is None:
            legacy = _load_opt("scaler.joblib")
            clf_scaler = legacy
            det_scaler = legacy

        with open(os.path.join(artefact_dir, "metadata.json")) as f:
            meta = json.load(f)

        with self._lock:
            self._model = LoadedModel(
                version_id=version_id,
                artefact_dir=artefact_dir,
                classifier_type=classifier_type,
                detector_type=detector_type,
                classifier=clf,
                detector=det,
                variance_threshold=vt,
                corr_mask=corr_mask,
                feature_selector=feature_selector,
                clf_scaler=clf_scaler,
                det_scaler=det_scaler,
                feature_names=meta.get("feature_names", []),
                selected_feature_names=meta.get("selected_feature_names", []),
                metrics=metrics,
            )

    def clear(self) -> None:
        with self._lock:
            self._model = None

    def get(self) -> LoadedModel | None:
        with self._lock:
            return self._model

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None


model_store = ModelStore()

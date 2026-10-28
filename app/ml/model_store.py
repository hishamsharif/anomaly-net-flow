"""
In-memory model store — holds the currently loaded classifier + detector.

Thread-safe: the training thread calls load() after a successful training run;
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
    classifier: Any      # BaseClassifier instance
    detector: Any        # BaseAnomalyDetector instance
    scaler: Any          # fitted RobustScaler
    variance_threshold: Any  # fitted VarianceThreshold
    feature_names: list[str]
    metrics: dict = field(default_factory=dict)


class ModelStore:
    """Singleton. Training thread writes; request handlers read."""

    def __init__(self):
        self._lock = threading.RLock()
        self._model: LoadedModel | None = None

    # ── Write (called from training thread) ────────────────────────────────

    def load_from_dir(
        self,
        version_id: int,
        artefact_dir: str,
        classifier_type: str,
        detector_type: str,
        metrics: dict,
    ) -> None:
        import joblib, json, os
        from app.ml.registry import get_classifier, get_detector

        clf_path = os.path.join(artefact_dir, "classifier.joblib")
        det_path = os.path.join(artefact_dir, "detector.joblib")
        scaler_path = os.path.join(artefact_dir, "scaler.joblib")
        vt_path = os.path.join(artefact_dir, "variance_threshold.joblib")
        meta_path = os.path.join(artefact_dir, "metadata.json")

        clf_cls = get_classifier(classifier_type)
        det_cls = get_detector(detector_type)

        clf = clf_cls.load(clf_path)
        det = det_cls.load(det_path)
        scaler = joblib.load(scaler_path)
        vt = joblib.load(vt_path)

        with open(meta_path) as f:
            meta = json.load(f)

        with self._lock:
            self._model = LoadedModel(
                version_id=version_id,
                artefact_dir=artefact_dir,
                classifier_type=classifier_type,
                detector_type=detector_type,
                classifier=clf,
                detector=det,
                scaler=scaler,
                variance_threshold=vt,
                feature_names=meta.get("feature_names", []),
                metrics=metrics,
            )

    def clear(self) -> None:
        with self._lock:
            self._model = None

    # ── Read (called from request handlers) ────────────────────────────────

    def get(self) -> LoadedModel | None:
        with self._lock:
            return self._model

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None


model_store = ModelStore()

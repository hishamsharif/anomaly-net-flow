"""
Component registry — add one line here to expose a new model in the training UI.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base import BaseClassifier, BaseAnomalyDetector

from core.classifiers.xgboost_clf import XGBoostClassifier
from core.classifiers.random_forest_clf import RandomForestClassifier
from core.detectors.mlp_ae import MLPAutoencoder
from core.detectors.isolation_forest import IsolationForestDetector

CLASSIFIERS: dict[str, type] = {
    "xgboost":       XGBoostClassifier,
    "random_forest": RandomForestClassifier,
}

ANOMALY_DETECTORS: dict[str, type] = {
    "mlp_ae":           MLPAutoencoder,
    "isolation_forest": IsolationForestDetector,
}


def get_classifier(name: str) -> type:
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{name}'. Available: {list(CLASSIFIERS)}")
    return CLASSIFIERS[name]


def get_detector(name: str) -> type:
    if name not in ANOMALY_DETECTORS:
        raise ValueError(f"Unknown detector '{name}'. Available: {list(ANOMALY_DETECTORS)}")
    return ANOMALY_DETECTORS[name]

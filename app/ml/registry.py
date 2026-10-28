"""
Component registry — add a new model class here to expose it in the training UI.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ml.base import BaseClassifier, BaseAnomalyDetector

from app.ml.classifiers.xgboost_clf import XGBoostClassifier
from app.ml.detectors.vanilla_ae import VanillaAE
from app.ml.detectors.isolation_forest import IsolationForestDetector

# Supervised classifiers: name → class
# Upgrade path: add FT-Transformer here when ready.
CLASSIFIERS: dict[str, type] = {
    "xgboost": XGBoostClassifier,
    # "ft_transformer": FTTransformerClassifier,
}

# Unsupervised anomaly detectors: name → class
# Upgrade path: add TransformerAE or VAE here when ready.
ANOMALY_DETECTORS: dict[str, type] = {
    "vanilla_ae": VanillaAE,
    "isolation_forest": IsolationForestDetector,   # lightweight; no GPU needed
    # "transformer_ae": TransformerAE,
    # "vae": VAE,
}


def get_classifier(name: str) -> type:
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{name}'. Available: {list(CLASSIFIERS)}")
    return CLASSIFIERS[name]


def get_detector(name: str) -> type:
    if name not in ANOMALY_DETECTORS:
        raise ValueError(f"Unknown detector '{name}'. Available: {list(ANOMALY_DETECTORS)}")
    return ANOMALY_DETECTORS[name]

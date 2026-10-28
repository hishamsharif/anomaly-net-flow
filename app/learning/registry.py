"""
Component registry — add a new model class here to expose it in the training UI.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.learning.base import BaseClassifier, BaseAnomalyDetector

from app.learning.classifiers.xgboost_clf import XGBoostClassifier
from app.learning.classifiers.random_forest_clf import RandomForestClassifier
from app.learning.detectors.mlp_ae import MLPAutoencoder
from app.learning.detectors.isolation_forest import IsolationForestDetector

# Supervised classifiers: name → class
# Upgrade path: add FT-Transformer here when ready.
CLASSIFIERS: dict[str, type] = {
    "xgboost": XGBoostClassifier,
    "random_forest": RandomForestClassifier,
    # "ft_transformer": FTTransformerClassifier,
}

# Unsupervised anomaly detectors: name → class
# Upgrade path: add TransformerAE or VAE here when ready.
ANOMALY_DETECTORS: dict[str, type] = {
    "mlp_ae": MLPAutoencoder,
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

"""
Initial training pipeline — runs once when no active model exists.

Reads bundled CIC-IDS2017 CSV, trains a selected classifier + anomaly
detector, saves versioned artefacts, returns metrics.

Feature engineering (stages 1-3 shared; stage 4 per-model-path):
  1. VarianceThreshold   — drop near-constant features
  2. Correlation pruning — drop one from each highly-correlated pair
  3. SelectKBest (MI)    — optional: keep top-K by mutual information
  4a. clf_scaler         — determined by classifier PreprocessingSpec
  4b. det_scaler         — determined by detector PreprocessingSpec
"""
from __future__ import annotations
import json
import os
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

from training.data_loader import load_initial
from training.feature_eng.pipeline import FeatureEngineeringPipeline

_META_COLS = {"Label", "Flow ID", "Timestamp", "Source IP", "Destination IP"}


def run(
    artefact_dir: str,
    bundled_dir: str,
    classifier_type: str = "xgboost",
    detector_type: str = "mlp_ae",
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """
    Train on CIC-IDS2017, save artefacts to artefact_dir, return metrics dict.
    artefact_dir must already exist (caller creates it with the version ID).
    """
    def progress(msg: str):
        if on_progress:
            on_progress(msg)

    from api.config import settings as cfg
    from core.registry import get_classifier, get_detector

    # 1. Load dataset
    progress("Loading CIC-IDS2017 CSV files ...")
    df = load_initial(bundled_dir)
    label_col = next((c for c in df.columns if c.strip() == "Label"), None)
    if label_col is None:
        raise ValueError("CIC-IDS2017 CSV must contain a 'Label' column.")

    # 2. Sort by Timestamp for day-based ordering
    ts_col = next((c for c in df.columns if c.strip() == "Timestamp"), None)
    if ts_col:
        df = df.sort_values(ts_col)

    labels = df[label_col].str.strip()
    feature_cols = [c for c in df.columns if c not in _META_COLS and c != label_col]
    X_raw = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )

    # 3. Time-ordered 80/10/10 split
    progress(f"Splitting {len(X_raw):,} rows (80% train / 10% val / 10% test) ...")
    n = len(X_raw)
    train_end, val_end = int(n * 0.8), int(n * 0.9)

    X_train_raw = X_raw.iloc[:train_end].values
    y_train     = labels.iloc[:train_end].values
    X_val_raw   = X_raw.iloc[train_end:val_end].values
    y_val       = labels.iloc[train_end:val_end].values
    X_test_raw  = X_raw.iloc[val_end:].values
    y_test      = labels.iloc[val_end:].values

    # 4. Feature engineering stages 1-3 (shared across all paths)
    fe = FeatureEngineeringPipeline(
        variance_threshold=cfg.variance_threshold,
        correlation_threshold=cfg.correlation_threshold,
        feature_selection_k=cfg.feature_selection_k,
    )
    fe_result = fe.fit_transform(
        X_train_raw, X_val_raw, X_test_raw,
        y_train, list(X_raw.columns),
        on_progress=progress,
    )

    # 5. Stage 4: path-specific scalers driven by PreprocessingSpec
    clf_cls = get_classifier(classifier_type)
    det_cls = get_detector(detector_type)

    clf_scaler_type = (
        "none" if not clf_cls.preprocessing.needs_scaling
        else cfg.scaler_type
    )
    det_scaler_type = (
        "none" if not det_cls.preprocessing.needs_scaling
        else cfg.scaler_type
    )
    progress(f"Scaler -- classifier: {clf_scaler_type}  detector: {det_scaler_type}")

    os.makedirs(artefact_dir, exist_ok=True)

    X_clf_tr, X_clf_v, X_clf_te = FeatureEngineeringPipeline.apply_scaler(
        fe_result, clf_scaler_type, artefact_dir, "clf_scaler.joblib")
    X_det_tr, X_det_v, X_det_te = FeatureEngineeringPipeline.apply_scaler(
        fe_result, det_scaler_type, artefact_dir, "det_scaler.joblib")

    # 6. SMOTE on classifier train split
    progress("Applying SMOTE to training split ...")
    smote = SMOTE(random_state=42)
    X_clf_bal, y_clf_bal = smote.fit_resample(X_clf_tr, y_train)

    # 7. Train classifier
    progress(f"Training {classifier_type} classifier ...")
    clf = clf_cls()
    clf.fit(X_clf_bal, y_clf_bal)

    # 8. Evaluate classifier on test set
    progress("Evaluating classifier on test split ...")
    raw_pred = clf.predict(X_clf_te)
    if raw_pred and isinstance(raw_pred[0], tuple):
        y_pred   = [p[0] for p in raw_pred]
        y_scores = np.array([p[1] if p[0] != "BENIGN" else 1.0 - p[1] for p in raw_pred])
    else:
        y_pred   = list(raw_pred)
        y_scores = None
    metrics = _compute_metrics(y_test, y_pred, y_scores)
    progress(f"Test  F1={metrics['f1']:.4f}  FAR={metrics['far']:.4f}  AMR={metrics['amr']:.4f}")

    # 9. Train anomaly detector on benign flows only
    progress(f"Training {detector_type} anomaly detector on benign flows ...")
    X_benign_train = X_det_tr[y_train == "BENIGN"]
    if detector_type == "mlp_ae":
        det = det_cls(
            anomaly_threshold_percentile=getattr(cfg, "ae_anomaly_threshold_percentile", 95.0),
            freeze_epochs=getattr(cfg, "ae_freeze_epochs", 5),
        )
    else:
        det = det_cls()
    det.fit(X_benign_train)

    # 10. Log top-20 feature importances
    importances = clf.feature_importances() if hasattr(clf, "feature_importances") else None
    sel_names = fe_result.feature_names_selected
    if importances is not None and len(importances) == len(sel_names):
        top_idx = np.argsort(importances)[::-1][:20]
        progress("Top-20 features by importance:")
        for rank, idx in enumerate(top_idx, 1):
            progress(f"  {rank:2d}. {sel_names[idx]:<45s}  {importances[idx]:.4f}")

    # 11. Save artefacts
    progress("Saving model artefacts ...")
    clf.save(os.path.join(artefact_dir, "classifier.joblib"))
    det.save(os.path.join(artefact_dir, "detector.joblib"))
    FeatureEngineeringPipeline.save_artefacts(fe_result, artefact_dir)

    metadata = {
        "feature_names":          list(X_raw.columns),
        "selected_feature_names": sel_names,
        "n_features_initial":     fe_result.n_initial,
        "n_features_final":       fe_result.n_final,
        "classes":                list(clf.classes_) if hasattr(clf, "classes_") else [],
        "ae_threshold":           float(det.threshold) if hasattr(det, "threshold") else None,
        "train_rows":             int(len(X_clf_bal)),
        "val_rows":               int(len(X_clf_v)),
        "test_rows":              int(len(X_clf_te)),
        "clf_scaler_type":        clf_scaler_type,
        "det_scaler_type":        det_scaler_type,
        "variance_threshold":     cfg.variance_threshold,
        "correlation_threshold":  cfg.correlation_threshold,
        "feature_selection_k":    cfg.feature_selection_k,
    }
    with open(os.path.join(artefact_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return {**metrics, **metadata, "artefact_dir": artefact_dir}


def _compute_metrics(y_true, y_pred, y_scores: np.ndarray | None = None) -> dict:
    """
    Compute F1 (macro), FAR, AMR, Recall, and PR-AUC for binary BENIGN vs ATTACK framing.
    PR-AUC is preferred for imbalanced IDS data -- less inflated by the benign majority.
    """
    from sklearn.metrics import average_precision_score

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_macro = report.get("macro avg", {}).get("f1-score", 0.0)

    y_bin_true = np.array([0 if v == "BENIGN" else 1 for v in y_true])
    y_bin_pred = np.array([0 if v == "BENIGN" else 1 for v in y_pred])

    cm = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    far    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    amr    = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    scores_for_pr = y_scores if y_scores is not None else y_bin_pred.astype(float)
    try:
        pr_auc = float(average_precision_score(y_bin_true, scores_for_pr))
    except Exception:
        pr_auc = 0.0

    return {
        "f1":     float(f1_macro),
        "far":    float(far),
        "amr":    float(amr),
        "recall": float(recall),
        "pr_auc": float(pr_auc),
    }

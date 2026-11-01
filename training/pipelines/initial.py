"""
Initial training pipeline — runs once when no active model exists.

Reads bundled CIC-IDS2017 CSV, trains a selected classifier + anomaly
detector, saves versioned artefacts, returns metrics.

Feature engineering (stages 1-3 shared; stage 4 per-model-path):
  1. VarianceThreshold    — drop near-constant features
  2. Correlation pruning  — drop one from each highly-correlated pair
  3. Feature selection    — optional: mi (SelectKBest) | rfe (RF-RFE) | none
  4a. clf_scaler          — determined by classifier PreprocessingSpec
  4b. detector_scaler     — determined by detector PreprocessingSpec

Naming conventions (standard scikit-learn / PyTorch ML practice):
  X_<path>_<split>  — feature matrix for a model path and data split
                       path: clf (classifier), detector (anomaly detector)
                       split: train, val, test
  y_<split>         — label vector (string class names from CIC-IDS2017)
  clf               — classifier instance  (scikit-learn convention)
  anomaly_detector  — anomaly detector instance (MLP-AE or Isolation Forest)
  detector_cls      — class object for the selected detector type
"""
from __future__ import annotations
import json
import os
import time
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

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

    started_at = time.time()

    from api.config import settings as cfg
    from core.registry import get_classifier, get_detector

    # 1-3. Load dataset with strict day-based split:
    #   Mon-Thu  → train  (all non-Friday files)
    #   Fri-AM   → val    (Friday WorkingHours Morning)
    #   Fri-PM   → test   (Friday WorkingHours Afternoon: DDos + PortScan)
    progress(
        "Loading CIC-IDS2017 CSV files "
        "(day split: Mon-Thu → train | Fri-AM → val | Fri-PM → test) ..."
    )
    from training.data_loader import load_initial_by_day
    df_train, df_val, df_test = load_initial_by_day(bundled_dir)

    label_col = next((c for c in df_train.columns if c.strip() == "Label"), None)
    if label_col is None:
        raise ValueError("CIC-IDS2017 CSV must contain a 'Label' column.")

    feature_cols = [c for c in df_train.columns if c not in _META_COLS and c != label_col]

    def _to_matrix(df: pd.DataFrame) -> np.ndarray:
        return (
            df[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
            .values
        )

    X_train_raw = _to_matrix(df_train)
    X_val_raw   = _to_matrix(df_val)
    X_test_raw  = _to_matrix(df_test)
    y_train = df_train[label_col].str.strip().values
    y_val   = df_val[label_col].str.strip().values
    y_test  = df_test[label_col].str.strip().values

    progress(
        f"Split — train: {len(X_train_raw):,} rows | "
        f"val: {len(X_val_raw):,} rows | test: {len(X_test_raw):,} rows"
    )

    # 4. Feature engineering stages 1-3 (shared across all paths)
    fe = FeatureEngineeringPipeline(
        variance_threshold=cfg.variance_threshold,
        correlation_threshold=cfg.correlation_threshold,
        feature_selection_k=cfg.feature_selection_k,
        feature_selection_method=getattr(cfg, "feature_selection_method", "mi"),
    )
    fe_result = fe.fit_transform(
        X_train_raw, X_val_raw, X_test_raw,
        y_train, feature_cols,
        on_progress=progress,
    )

    # 5. Stage 4: path-specific scalers driven by PreprocessingSpec
    clf_cls      = get_classifier(classifier_type)
    detector_cls = get_detector(detector_type)

    clf_scaler_type = (
        "none" if not clf_cls.preprocessing.needs_scaling
        else (clf_cls.preprocessing.preferred_scalers[0]
              if clf_cls.preprocessing.preferred_scalers[0] != "none"
              else cfg.scaler_type)
    )
    # Detector scaler: use the model's first preferred scaler (e.g. QuantileTransformer
    # for MLP-AE) rather than the global cfg.scaler_type. RobustScaler breaks on
    # CIC-IDS2017 sparse features (IQR≈0 → scale_=1 → unscaled outliers reach the AE).
    det_scaler_type = (
        "none" if not detector_cls.preprocessing.needs_scaling
        else (detector_cls.preprocessing.preferred_scalers[0]
              if detector_cls.preprocessing.preferred_scalers[0] != "none"
              else cfg.scaler_type)
    )
    progress(f"Scaler -- classifier: {clf_scaler_type}  detector: {det_scaler_type}")

    os.makedirs(artefact_dir, exist_ok=True)

    X_clf_train, X_clf_val, X_clf_test = FeatureEngineeringPipeline.apply_scaler(
        fe_result, clf_scaler_type, artefact_dir, "clf_scaler.joblib")
    X_detector_train, X_detector_val, X_detector_test = FeatureEngineeringPipeline.apply_scaler(
        fe_result, det_scaler_type, artefact_dir, "det_scaler.joblib")

    # 6. SMOTE on classifier train split (capped per minority class to avoid 10M+ rows)
    smote_cap = getattr(cfg, "smote_max_per_class", 50_000)
    counts = Counter(y_train)
    majority_count = max(counts.values())
    target = min(majority_count, smote_cap)
    sampling_strategy = {
        cls: target for cls, cnt in counts.items() if cnt < target
    }
    n_minority = sum(1 for cnt in counts.values() if cnt < target)
    progress(
        f"Applying SMOTE to training split "
        f"(cap={target:,}/class, {n_minority} minority classes) ..."
    )
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    X_clf_train_smote, y_clf_train_smote = smote.fit_resample(X_clf_train, y_train)

    # 7. Train classifier
    progress(f"Training {classifier_type} classifier ...")
    clf = clf_cls()
    clf.fit(X_clf_train_smote, y_clf_train_smote)

    # 8. Evaluate classifier on test set
    progress("Evaluating classifier on test split ...")
    raw_pred = clf.predict(X_clf_test)
    if raw_pred and isinstance(raw_pred[0], tuple):
        y_pred   = [p[0] for p in raw_pred]
        y_scores = np.array([p[1] if p[0] != "BENIGN" else 1.0 - p[1] for p in raw_pred])
    else:
        y_pred   = list(raw_pred)
        y_scores = None
    metrics = _compute_metrics(y_test, y_pred, y_scores)
    # Known-attack F1: only test rows whose true class is a NON-BENIGN class seen
    # in training.  BENIGN is excluded because when the test set contains few
    # attack rows, including BENIGN would make sklearn dilute the macro F1 across
    # every class that appears in y_pred (most with F1=0), masking classifier quality.
    trained_classes = set(clf.classes_) if hasattr(clf, "classes_") else set()
    trained_attack_classes = trained_classes - {"BENIGN"}
    known_attack_mask = np.array([label in trained_attack_classes for label in y_test])
    if known_attack_mask.sum() > 0:
        km = _compute_metrics(
            np.array(y_test)[known_attack_mask],
            np.array(y_pred)[known_attack_mask],
            y_scores[known_attack_mask] if y_scores is not None else None,
        )
        metrics["known_attack_f1"] = km["f1"]
    else:
        metrics["known_attack_f1"] = 0.0
        progress(
            "NOTE: no trained attack classes present in test set — "
            "known_attack_f1=0 (expected when all Friday attacks are novel)."
        )
    progress(
        f"Test  F1(macro)={metrics['f1']:.4f}  "
        f"F1(known-attacks)={metrics['known_attack_f1']:.4f}  "
        f"FAR={metrics['far']:.4f}  AMR={metrics['amr']:.4f}"
    )

    # 9. Train anomaly detector on benign flows only.
    # Threshold is set on validation benign rows (not training rows) so it reflects
    # out-of-sample reconstruction error, giving a calibrated FAR floor.
    progress(f"Training {detector_type} anomaly detector on benign flows ...")
    X_detector_train_benign = X_detector_train[y_train == "BENIGN"]
    X_detector_val_benign   = X_detector_val[y_val == "BENIGN"]
    detector_cls = get_detector(detector_type)
    if detector_type in ("mlp_ae", "transfer_ae"):
        anomaly_detector = detector_cls(
            anomaly_threshold_percentile=getattr(cfg, "ae_anomaly_threshold_percentile", 95.0),
            freeze_epochs=getattr(cfg, "ae_freeze_epochs", 5),
        )
        anomaly_detector.fit(
            X_detector_train_benign, X_benign_val=X_detector_val_benign, on_progress=progress
        )
    else:
        anomaly_detector = detector_cls()
        anomaly_detector.fit(X_detector_train_benign)

    # 10. Novel-attack detection rate: fraction of test-set novel attacks caught by AE.
    # Novel = non-BENIGN AND not in trained attack classes (i.e. the unseen Friday attacks).
    # This is the primary metric when the classifier has zero coverage of the test classes.
    novel_attack_mask = np.array([
        label != "BENIGN" and label not in trained_attack_classes
        for label in y_test
    ])
    if novel_attack_mask.sum() > 0:
        ae_scores_test    = anomaly_detector.score_batch(X_detector_test)
        ae_threshold      = anomaly_detector.threshold or 0.0
        novel_detected = int((ae_scores_test[novel_attack_mask] > ae_threshold).sum())
        novel_total    = int(novel_attack_mask.sum())
        metrics["novel_attack_detection_rate"] = round(novel_detected / novel_total, 4)
        metrics["novel_attack_test_rows"]      = novel_total
        metrics["novel_attack_detected"]       = novel_detected
        progress(
            f"Novel-attack detection (AE only): "
            f"{novel_detected:,}/{novel_total:,} = "
            f"{metrics['novel_attack_detection_rate']:.1%}"
        )
    else:
        metrics["novel_attack_detection_rate"] = None
        metrics["novel_attack_test_rows"]      = 0
        metrics["novel_attack_detected"]       = 0

    # 11a. Log top-20 feature importances
    importances = clf.feature_importances() if hasattr(clf, "feature_importances") else None
    sel_names = fe_result.feature_names_selected
    if importances is not None and len(importances) == len(sel_names):
        top_idx = np.argsort(importances)[::-1][:20]
        progress("Top-20 features by importance:")
        for rank, idx in enumerate(top_idx, 1):
            progress(f"  {rank:2d}. {sel_names[idx]:<45s}  {importances[idx]:.4f}")

    # 11b. Save regression snapshot for AE candidate gate (fine-tune pipeline).
    # 2,000 Fri-AM benign flows + up to 2,000 novel-attack flows from the test set.
    _save_regression_snapshot(
        artefact_dir, X_detector_val, y_val, X_detector_test, y_test,
        n_benign=2000, n_novel=2000
    )

    # 12. Save artefacts
    progress("Saving model artefacts ...")
    clf.save(os.path.join(artefact_dir, "classifier.joblib"))
    anomaly_detector.save(os.path.join(artefact_dir, "detector.joblib"))
    FeatureEngineeringPipeline.save_artefacts(fe_result, artefact_dir)

    completed_at = time.time()

    # 13. Emit feature_schema.json — versioned feature contract for inference
    _save_feature_schema(artefact_dir, feature_cols, sel_names)

    # Feature importances for top-15 display in the UI
    importances_full = clf.feature_importances() if hasattr(clf, "feature_importances") else None
    top_features: list[dict] = []
    if importances_full is not None and len(importances_full) == len(sel_names):
        top_idx = np.argsort(importances_full)[::-1][:15]
        top_features = [
            {"name": sel_names[i], "importance": float(importances_full[i])}
            for i in top_idx
        ]

    metadata = {
        "classifier_type":        classifier_type,
        "detector_type":          detector_type,
        "feature_names":          list(feature_cols),
        "selected_feature_names": sel_names,
        "n_features_initial":     fe_result.n_initial,
        "n_features_after_vt":    fe_result.n_after_vt,
        "n_features_after_corr":  fe_result.n_after_corr,
        "n_features_final":       fe_result.n_final,
        "top_features":           top_features,
        "classes":                list(clf.classes_) if hasattr(clf, "classes_") else [],
        "class_distribution_train": {k: int(v) for k, v in counts.items()},
        "ae_threshold":           float(anomaly_detector.threshold) if hasattr(anomaly_detector, "threshold") else None,
        "train_rows_raw":         int(len(X_clf_train)),
        "train_rows":             int(len(X_clf_train_smote)),
        "val_rows":               int(len(X_clf_val)),
        "test_rows":              int(len(X_clf_test)),
        "smote_before_total":     int(len(X_clf_train)),
        "smote_after_total":      int(len(X_clf_train_smote)),
        "smote_n_minority":       int(n_minority),
        "smote_target_per_class": int(target),
        "clf_scaler_type":        clf_scaler_type,
        "det_scaler_type":        det_scaler_type,
        "variance_threshold":        cfg.variance_threshold,
        "correlation_threshold":     cfg.correlation_threshold,
        "feature_selection_k":       cfg.feature_selection_k,
        "feature_selection_method":  fe_result.selection_method,
        "smote_max_per_class":    smote_cap,
        "started_at":             started_at,
        "completed_at":           completed_at,
        "training_duration_s":    round(completed_at - started_at, 1),
        "known_attack_f1":               metrics.get("known_attack_f1", 0.0),
        "novel_attack_detection_rate":   metrics.get("novel_attack_detection_rate"),
        "novel_attack_test_rows":        metrics.get("novel_attack_test_rows", 0),
        "novel_attack_detected":         metrics.get("novel_attack_detected", 0),
    }
    with open(os.path.join(artefact_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return {**metrics, **metadata, "artefact_dir": artefact_dir,
            "known_attack_f1": metrics.get("known_attack_f1", 0.0)}


def _save_feature_schema(artefact_dir: str, raw_feature_cols: list, selected_names: list) -> None:
    """Write feature_schema.json — versioned feature contract for training/inference."""
    import datetime
    try:
        from inference.adapters.cicfm_adapter import _CICFM_VERSION as extractor_ver
        extractor_name = "cicflowmeter-python"
    except ImportError:
        extractor_name = "nfstream"
        extractor_ver = "unknown"

    # Features known to be absent from NFStream (present in cicflowmeter)
    _NFSTREAM_ABSENT = [
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Std", "Fwd Header Length", "Bwd Header Length",
    ]
    absent = [f for f in _NFSTREAM_ABSENT if f in selected_names] if extractor_name == "nfstream" else []

    schema = {
        "schema_version":      "cicids2017-flow-schema-v1",
        "extractor_name":      extractor_name,
        "extractor_version":   extractor_ver,
        "raw_feature_count":   len(raw_feature_cols),
        "selected_feature_count": len(selected_names),
        "selected_features":   selected_names,
        "features_absent_from_extractor": absent,
        "compatibility_notes": (
            "Selected features are the output of VarianceThreshold + correlation pruning "
            "applied to CIC-IDS2017 CSV columns. Inference must supply the same ordered "
            "feature list; missing columns are filled with 0. "
            + (
                f"WARNING: {len(absent)} selected features not producible by {extractor_name}: "
                + str(absent)
                if absent else
                "All selected features are producible by the registered extractor."
            )
        ),
        "missing_value_policy": "fill_zero",
        "scaler_note":          "clf_scaler.joblib (classifier path), det_scaler.joblib (detector path)",
        "created_at":           datetime.datetime.utcnow().isoformat() + "Z",
    }
    path = os.path.join(artefact_dir, "feature_schema.json")
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def _save_regression_snapshot(
    artefact_dir: str,
    X_det_val: np.ndarray, y_val: np.ndarray,
    X_det_test: np.ndarray, y_test: np.ndarray,
    n_benign: int = 2000, n_novel: int = 2000,
) -> None:
    """
    Save a fixed regression snapshot for AE candidate quality gate in fine_tune.py.
    Uses Fri-AM benign (val) + a sample of novel-attack rows from test.
    Stored as regression_X.npy + regression_y_is_attack.npy in artefact_dir.
    """
    rng = np.random.default_rng(42)

    benign_idx = np.where(y_val == "BENIGN")[0]
    chosen_b = rng.choice(benign_idx, size=min(n_benign, len(benign_idx)), replace=False)
    X_b = X_det_val[chosen_b]

    novel_idx = np.where(y_test != "BENIGN")[0]
    chosen_n = rng.choice(novel_idx, size=min(n_novel, len(novel_idx)), replace=False)
    X_n = X_det_test[chosen_n]

    X_reg = np.vstack([X_b, X_n])
    y_reg = np.array([0] * len(X_b) + [1] * len(X_n))   # 0=benign, 1=attack

    np.save(os.path.join(artefact_dir, "regression_X.npy"),           X_reg)
    np.save(os.path.join(artefact_dir, "regression_y_is_attack.npy"), y_reg)


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

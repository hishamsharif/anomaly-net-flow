"""
Initial training pipeline — runs once when no active model exists.

Reads bundled CIC-IDS2017 CSV via DuckDB, trains XGBoost + Vanilla AE,
saves versioned artefacts, returns metrics.
"""
from __future__ import annotations
import json
import os
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler

from training.data_loader import load_initial

_META_COLS = {"Label", "Flow ID", "Timestamp", "Source IP", "Destination IP"}


def run(
    artefact_dir: str,
    bundled_dir: str,
    classifier_type: str = "xgboost",
    detector_type: str = "vanilla_ae",
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """
    Train on CIC-IDS2017, save artefacts to artefact_dir, return metrics dict.
    artefact_dir must already exist (caller creates it with the version ID).
    """
    def progress(msg: str):
        if on_progress:
            on_progress(msg)

    # ── 1. Load dataset ───────────────────────────────────────────────────
    progress("Loading CIC-IDS2017 CSV files …")
    df = load_initial(bundled_dir)
    label_col = next((c for c in df.columns if c.strip() == "Label"), None)
    if label_col is None:
        raise ValueError("CIC-IDS2017 CSV must contain a 'Label' column.")

    # ── 2. Sort by Timestamp for day-based ordering ───────────────────────
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

    # ── 3. Time-ordered 80/10/10 split (Mon-Thu train / Fri val / Fri-pm test)
    progress(f"Splitting {len(X_raw):,} rows (80% train / 10% val / 10% test) …")
    n = len(X_raw)
    train_end, val_end = int(n * 0.8), int(n * 0.9)

    X_train_raw, y_train = X_raw.iloc[:train_end].values, labels.iloc[:train_end].values
    X_val_raw, y_val = X_raw.iloc[train_end:val_end].values, labels.iloc[train_end:val_end].values
    X_test_raw, y_test = X_raw.iloc[val_end:].values, labels.iloc[val_end:].values

    # ── 4. Variance threshold + RobustScaler (fit on train only) ─────────
    progress("Fitting preprocessing (VarianceThreshold + RobustScaler) …")
    vt = VarianceThreshold()
    X_train = vt.fit_transform(X_train_raw)
    X_val = vt.transform(X_val_raw)
    X_test = vt.transform(X_test_raw)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── 5. SMOTE on train split only ──────────────────────────────────────
    progress("Applying SMOTE to training split …")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # ── 6. Train classifier ───────────────────────────────────────────────
    progress(f"Training {classifier_type} classifier …")
    from app.ml.registry import get_classifier, get_detector
    clf_cls = get_classifier(classifier_type)
    clf = clf_cls()
    clf.fit(X_train_bal, y_train_bal)

    # ── 7. Evaluate classifier on test set ───────────────────────────────
    progress("Evaluating classifier on test split …")
    y_pred = clf.predict(X_test)
    if isinstance(y_pred[0], tuple):        # predict() returns (label, conf) tuples
        y_pred = [p[0] for p in y_pred]
    metrics = _compute_metrics(y_test, y_pred)
    progress(f"Test  F1={metrics['f1']:.4f}  FAR={metrics['far']:.4f}  AMR={metrics['amr']:.4f}")

    # ── 8. Train anomaly detector on benign train flows only ──────────────
    progress(f"Training {detector_type} anomaly detector on benign flows …")
    X_benign_train = X_train[y_train == "BENIGN"]
    det_cls = get_detector(detector_type)
    det = det_cls()
    det.fit(X_benign_train)

    # ── 9. Save artefacts ─────────────────────────────────────────────────
    progress("Saving model artefacts …")
    os.makedirs(artefact_dir, exist_ok=True)
    clf_path = os.path.join(artefact_dir, "classifier.joblib")
    det_path = os.path.join(artefact_dir, "detector.joblib")
    scaler_path = os.path.join(artefact_dir, "scaler.joblib")
    vt_path = os.path.join(artefact_dir, "variance_threshold.joblib")
    meta_path = os.path.join(artefact_dir, "metadata.json")

    clf.save(clf_path)
    det.save(det_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(vt, vt_path)

    metadata = {
        "feature_names": list(X_raw.columns),
        "classes": list(clf.classes_) if hasattr(clf, "classes_") else [],
        "ae_threshold": float(det.threshold) if hasattr(det, "threshold") else None,
        "train_rows": int(len(X_train_bal)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        **metrics,
        **metadata,
        "artefact_dir": artefact_dir,
    }


def _compute_metrics(y_true, y_pred) -> dict:
    """Compute F1 (macro), FAR, AMR for binary BENIGN vs ATTACK framing."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_macro = report.get("macro avg", {}).get("f1-score", 0.0)

    # Binary: BENIGN=negative, everything else=positive (attack)
    y_bin_true = np.array(["ATTACK" if v != "BENIGN" else "BENIGN" for v in y_true])
    y_bin_pred = np.array(["ATTACK" if v != "BENIGN" else "BENIGN" for v in y_pred])

    labels = ["BENIGN", "ATTACK"]
    cm = confusion_matrix(y_bin_true, y_bin_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false alarm rate
    amr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # attack miss rate

    return {"f1": float(f1_macro), "far": float(far), "amr": float(amr)}

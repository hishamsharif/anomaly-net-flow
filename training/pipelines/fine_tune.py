"""
Fine-tune pipeline.

XGBoost:       full cumulative retrain on all labelled_flows up to the new savepoint
               (no weight carry-over; trees are additive).
MLPAutoencoder: load previous weights -> freeze encoder N epochs -> fine-tune on
               incremental rows + replay buffer -> unfreeze.
"""
from __future__ import annotations
import json
import os
import shutil
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from training.data_loader import (
    load_cumulative,
    load_incremental,
    load_replay_buffer,
    get_max_flow_id,
)

_META_COLS = {"Label", "label", "Flow ID", "Timestamp", "Source IP",
              "Destination IP", "features_json", "source", "id",
              "predicted_label", "model_version_id", "analyst_id", "created_at"}


def run(
    prev_artefact_dir: str,
    artefact_dir: str,
    prev_savepoint_max_id: int,
    classifier_type: str = "xgboost",
    detector_type: str = "mlp_ae",
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """
    Fine-tune from the label store. Save new artefacts to artefact_dir.
    """
    def progress(msg: str):
        if on_progress:
            on_progress(msg)

    from api.config import settings
    from core.registry import get_classifier, get_detector

    # Load preprocessing artefacts from previous version
    vt = joblib.load(os.path.join(prev_artefact_dir, "variance_threshold.joblib"))

    def _load_opt(fname):
        p = os.path.join(prev_artefact_dir, fname)
        return joblib.load(p) if os.path.exists(p) else None

    corr_mask        = _load_opt("corr_mask.joblib")
    feature_selector = _load_opt("feature_selector.joblib")

    # Prefer new dual scalers; fall back to legacy single scaler
    clf_scaler = _load_opt("clf_scaler.joblib") or _load_opt("scaler.joblib")
    det_scaler = _load_opt("det_scaler.joblib") or _load_opt("scaler.joblib")

    new_max_id = get_max_flow_id()
    progress(f"New savepoint max_flow_id = {new_max_id}")

    # XGBoost: cumulative retrain
    progress("Loading all labelled flows for cumulative retrain ...")
    df_cumulative = load_cumulative(new_max_id)
    X_clf, y_clf = _extract(df_cumulative, vt, corr_mask, feature_selector, clf_scaler)
    progress(f"  {len(X_clf):,} rows")

    clf_cls = get_classifier(classifier_type)
    clf = clf_cls()
    progress(f"Training {classifier_type} ...")
    clf.fit(X_clf, y_clf)

    # AE: incremental rows + replay buffer
    df_inc    = load_incremental(prev_savepoint_max_id, new_max_id)
    df_replay = load_replay_buffer(prev_savepoint_max_id, settings.ae_replay_buffer_fraction)
    X_inc, y_inc = _extract(df_inc,    vt, corr_mask, feature_selector, det_scaler)
    X_rep, y_rep = _extract(df_replay, vt, corr_mask, feature_selector, det_scaler)

    X_inc_benign = X_inc[y_inc == "BENIGN"]
    X_rep_benign = X_rep[y_rep == "BENIGN"] if len(X_rep) else None
    progress(f"Fine-tuning {detector_type}: {len(X_inc_benign)} new benign + "
             f"{len(X_rep_benign) if X_rep_benign is not None else 0} replay benign ...")

    det_cls = get_detector(detector_type)
    det = det_cls.load(os.path.join(prev_artefact_dir, "detector.joblib"))
    det.fine_tune(X_inc_benign, X_rep_benign if X_rep_benign is not None and len(X_rep_benign) else None)

    # Evaluate on cumulative test rows
    from training.pipelines.initial import _compute_metrics
    n = len(X_clf)
    test_start = int(n * 0.9)
    if test_start < n:
        y_pred_arr = [clf.predict(X_clf[i:i+1])[0] for i in range(test_start, n)]
        if y_pred_arr and isinstance(y_pred_arr[0], tuple):
            y_pred_arr = [p[0] for p in y_pred_arr]
        metrics = _compute_metrics(y_clf[test_start:], y_pred_arr)
    else:
        metrics = {"f1": 0.0, "far": 0.0, "amr": 0.0}
    progress(f"Test  F1={metrics['f1']:.4f}  FAR={metrics['far']:.4f}  AMR={metrics['amr']:.4f}")

    # Save artefacts
    progress("Saving artefacts ...")
    os.makedirs(artefact_dir, exist_ok=True)
    clf.save(os.path.join(artefact_dir, "classifier.joblib"))
    det.save(os.path.join(artefact_dir, "detector.joblib"))

    # Copy all preprocessing artefacts from previous version
    for fname in ("variance_threshold.joblib", "corr_mask.joblib",
                  "feature_selector.joblib", "clf_scaler.joblib", "det_scaler.joblib",
                  "scaler.joblib"):
        src = os.path.join(prev_artefact_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, artefact_dir)

    prev_meta_path = os.path.join(prev_artefact_dir, "metadata.json")
    prev_meta = {}
    if os.path.exists(prev_meta_path):
        with open(prev_meta_path) as f:
            prev_meta = json.load(f)

    meta = {
        **prev_meta,
        "ae_threshold":           float(det.threshold) if hasattr(det, "threshold") else None,
        "cumulative_rows":        int(len(df_cumulative)),
        "incremental_rows":       int(len(df_inc)),
        "new_savepoint_max_id":   int(new_max_id),
    }
    with open(os.path.join(artefact_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {**metrics, **meta, "artefact_dir": artefact_dir, "new_savepoint_max_id": int(new_max_id)}


def _extract(df: pd.DataFrame, vt, corr_mask, feature_selector, scaler):
    label_col = next((c for c in df.columns if c in {"label", "Label"}), None)
    y = df[label_col].values if label_col else np.array(["UNKNOWN"] * len(df))
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .replace([float("inf"), float("-inf")], 0)
        .values
    )
    X = vt.transform(X)
    if corr_mask is not None:
        X = X[:, corr_mask]
    if feature_selector is not None:
        X = feature_selector.transform(X)
    if scaler is not None:
        X = scaler.transform(X)
    return X, y

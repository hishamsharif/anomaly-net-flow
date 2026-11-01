"""
AE retraining pipeline (AE-only, periodic).

XGBoost: frozen at the base CIC-IDS2017 artefact — always copied from prev_artefact_dir,
         never retrained here. Inference always uses the same base XGBoost model.
MLPAutoencoder: load previous weights → freeze encoder N epochs → fine-tune on
                incremental analyst-labelled flows + replay buffer → unfreeze.
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
    AE-only retraining from analyst-labelled flows. XGBoost is always copied
    unchanged from prev_artefact_dir — it is never retrained here.
    """
    def progress(msg: str):
        if on_progress:
            on_progress(msg)

    from api.config import settings
    from core.registry import get_detector

    # Load preprocessing artefacts from previous version
    vt = joblib.load(os.path.join(prev_artefact_dir, "variance_threshold.joblib"))

    def _load_opt(fname):
        p = os.path.join(prev_artefact_dir, fname)
        return joblib.load(p) if os.path.exists(p) else None

    corr_mask        = _load_opt("corr_mask.joblib")
    feature_selector = _load_opt("feature_selector.joblib")

    # Prefer new dual scalers; fall back to legacy single scaler
    det_scaler = _load_opt("det_scaler.joblib") or _load_opt("scaler.joblib")

    new_max_id = get_max_flow_id()
    progress(f"New savepoint max_flow_id = {new_max_id}")
    progress("Fetching incremental analyst-labelled flows ...")

    # AE: incremental rows (analyst verdicts since last savepoint) + replay buffer
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

    # AE candidate regression gate: compare candidate vs incumbent on fixed snapshot.
    # Snapshot created during initial training: regression_X.npy + regression_y_is_attack.npy
    reg_metrics = _ae_regression_gate(
        prev_artefact_dir, det, det_scaler, progress
    )
    metrics = {"f1": 0.0, "far": 0.0, "amr": 0.0}
    if reg_metrics:
        # Expose gate recall/FAR as primary metrics for this AE-only version
        metrics["recall"] = reg_metrics.get("ae_gate_candidate_recall", 0.0)
        metrics["far"]    = reg_metrics.get("ae_gate_candidate_far", 0.0)
        metrics.update(reg_metrics)
    progress(f"AE retrain recall={metrics.get('recall', 0):.4f}  FAR={metrics.get('far', 0):.4f}")

    # Save artefacts
    progress("Saving artefacts ...")
    os.makedirs(artefact_dir, exist_ok=True)

    # Copy frozen base classifier — never retrained
    progress("Copying frozen base XGBoost classifier ...")
    base_clf_src = os.path.join(prev_artefact_dir, "classifier.joblib")
    if os.path.exists(base_clf_src):
        shutil.copy2(base_clf_src, os.path.join(artefact_dir, "classifier.joblib"))
    else:
        raise RuntimeError("Base classifier.joblib not found in prev_artefact_dir — cannot proceed.")

    det.save(os.path.join(artefact_dir, "detector.joblib"))

    # Copy all preprocessing artefacts and feature schema from previous version
    for fname in ("variance_threshold.joblib", "corr_mask.joblib",
                  "feature_selector.joblib", "clf_scaler.joblib", "det_scaler.joblib",
                  "scaler.joblib", "feature_schema.json",
                  "regression_X.npy", "regression_y_is_attack.npy"):
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
        "incremental_rows":       int(len(df_inc)),
        "new_savepoint_max_id":   int(new_max_id),
        "xgboost_frozen":         True,
        **(reg_metrics or {}),
    }
    with open(os.path.join(artefact_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {**metrics, **meta, "artefact_dir": artefact_dir, "new_savepoint_max_id": int(new_max_id)}


def _ae_regression_gate(
    prev_artefact_dir: str,
    candidate_det,
    det_scaler,
    progress,
) -> dict | None:
    """
    Compare candidate AE vs incumbent on the fixed regression snapshot saved
    during initial training.  Logs a WARNING if recall drops > 5% or FAR rises
    > 10%.  Returns a dict of gate metrics to include in fine-tune metadata.
    Skips silently if no regression snapshot exists.
    """
    reg_X_path = os.path.join(prev_artefact_dir, "regression_X.npy")
    reg_y_path = os.path.join(prev_artefact_dir, "regression_y_is_attack.npy")
    if not (os.path.exists(reg_X_path) and os.path.exists(reg_y_path)):
        progress("AE regression gate: no snapshot found — skipping gate check.")
        return None

    X_reg = np.load(reg_X_path)
    y_reg = np.load(reg_y_path)          # 0=benign, 1=attack

    # Rescale if a scaler is available (snapshot is stored pre-scaled in det space)
    # — snapshot was saved from X_det_val/test which are already scaled, so no transform needed.

    # Load incumbent from previous artefact
    incumbent_path = os.path.join(prev_artefact_dir, "detector.joblib")
    if not os.path.exists(incumbent_path):
        progress("AE regression gate: no incumbent detector found — skipping.")
        return None

    try:
        from core.registry import get_detector
        incumbent_cls = type(candidate_det)
        incumbent = incumbent_cls.load(incumbent_path)
    except Exception as e:
        progress(f"AE regression gate: could not load incumbent ({e}) — skipping.")
        return None

    def _eval(det) -> tuple[float, float]:
        scores = det.score_batch(X_reg)
        thresh = det.threshold or 0.0
        flags  = (scores > thresh).astype(int)
        attack_rows = y_reg == 1
        benign_rows = y_reg == 0
        recall = float(flags[attack_rows].mean()) if attack_rows.sum() > 0 else 0.0
        far    = float(flags[benign_rows].mean()) if benign_rows.sum() > 0 else 0.0
        return recall, far

    inc_recall, inc_far   = _eval(incumbent)
    cand_recall, cand_far = _eval(candidate_det)

    recall_drop = inc_recall - cand_recall   # positive = candidate is worse
    far_increase = cand_far - inc_far         # positive = candidate raises FAR

    gate_pass = recall_drop <= 0.05 and far_increase <= 0.10
    status = "PASS" if gate_pass else "WARN"

    msg = (
        f"AE regression gate [{status}]: "
        f"incumbent recall={inc_recall:.3f} FAR={inc_far:.3f}  |  "
        f"candidate recall={cand_recall:.3f} FAR={cand_far:.3f}  |  "
        f"Δrecall={recall_drop:+.3f} ΔFAR={far_increase:+.3f}"
    )
    progress(msg)
    if not gate_pass:
        progress(
            "  WARNING: candidate AE fails quality gate. "
            "Proceeding with promotion — review metrics before serving production traffic."
        )

    return {
        "ae_gate_status":           status,
        "ae_gate_incumbent_recall": round(inc_recall, 4),
        "ae_gate_incumbent_far":    round(inc_far, 4),
        "ae_gate_candidate_recall": round(cand_recall, 4),
        "ae_gate_candidate_far":    round(cand_far, 4),
        "ae_gate_recall_drop":      round(recall_drop, 4),
        "ae_gate_far_increase":     round(far_increase, 4),
    }


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

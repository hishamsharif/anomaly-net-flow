#!/usr/bin/env python3
"""
AnomalyNetFlow — Standalone inference script.

Usage:
    python predict.py <csv_file>                        # auto-detect latest model
    python predict.py <csv_file> --model models/xgboost_mlp_ae_v8
    python predict.py <csv_file> --out results.csv      # save predictions to file

No API server needed. Loads artefacts directly and runs the 3-stage pipeline.

GPU is used automatically if CUDA is available; silently falls back to CPU.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd


# ── 1. Resolve model directory ────────────────────────────────────────────────

def find_latest_model(models_dir: str = "models") -> str:
    if not os.path.isdir(models_dir):
        sys.exit(f"ERROR: models/ directory not found at '{models_dir}'.\n"
                 "Run initial training first: bash ops.sh train initial")
    versions = sorted(
        [d for d in os.listdir(models_dir)
         if os.path.isfile(os.path.join(models_dir, d, "metadata.json"))],
        key=lambda d: os.path.getmtime(os.path.join(models_dir, d, "metadata.json")),
        reverse=True,
    )
    if not versions:
        sys.exit("ERROR: No trained model found in models/.\n"
                 "Run initial training first: bash ops.sh train initial")
    return os.path.join(models_dir, versions[0])


# ── 2. Load artefacts ─────────────────────────────────────────────────────────

def load_model(artefact_dir: str):
    with open(os.path.join(artefact_dir, "metadata.json")) as f:
        meta = json.load(f)

    clf_type = meta.get("classifier_type") or _infer_type(artefact_dir, "classifier")
    det_type = meta.get("detector_type")   or _infer_type(artefact_dir, "detector")

    from core.registry import get_classifier, get_detector
    clf = get_classifier(clf_type).load(os.path.join(artefact_dir, "classifier.joblib"))
    det = get_detector(det_type).load(os.path.join(artefact_dir, "detector.joblib"))

    vt = joblib.load(os.path.join(artefact_dir, "variance_threshold.joblib"))

    def _opt(fname):
        p = os.path.join(artefact_dir, fname)
        return joblib.load(p) if os.path.exists(p) else None

    corr_mask        = _opt("corr_mask.joblib")
    feature_selector = _opt("feature_selector.joblib")
    clf_scaler       = _opt("clf_scaler.joblib")
    det_scaler       = _opt("det_scaler.joblib")
    if clf_scaler is None and det_scaler is None:
        legacy = _opt("scaler.joblib")
        clf_scaler = det_scaler = legacy

    return {
        "clf": clf, "det": det,
        "vt": vt, "corr_mask": corr_mask,
        "feature_selector": feature_selector,
        "clf_scaler": clf_scaler, "det_scaler": det_scaler,
        "feature_names": meta.get("feature_names", []),
        "ae_threshold": meta.get("ae_threshold"),
        "clf_type": clf_type, "det_type": det_type,
        "meta": meta,
    }


def _infer_type(artefact_dir: str, role: str) -> str:
    name = os.path.basename(artefact_dir).lower()
    if role == "classifier":
        if "random_forest" in name: return "random_forest"
        return "xgboost"
    # detector
    if "isolation_forest" in name: return "isolation_forest"
    if "transfer_ae" in name:      return "transfer_ae"
    return "mlp_ae"


# ── 3. Preprocessing ──────────────────────────────────────────────────────────

_META_COLS = {"Label", "Flow ID", "Timestamp", "Source IP", "Destination IP"}

def preprocess(df: pd.DataFrame, model: dict, path: str) -> np.ndarray:
    feature_names = model["feature_names"]
    available = [c for c in feature_names if c in df.columns]
    missing   = [c for c in feature_names if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} expected features not in CSV; filling with 0")

    X = (df.reindex(columns=feature_names, fill_value=0)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(0)
           .replace([float("inf"), float("-inf")], 0)
           .values)

    X = model["vt"].transform(X)
    if model["corr_mask"] is not None:
        X = X[:, model["corr_mask"]]
    if model["feature_selector"] is not None:
        X = model["feature_selector"].transform(X)
    scaler = model["clf_scaler"] if path == "clf" else model["det_scaler"]
    if scaler is not None:
        X = scaler.transform(X)
    return X


# ── 4. Stage 1 — Rule engine ──────────────────────────────────────────────────

def rule_verdict(row: dict) -> str:
    dst  = row.get("Destination Port", -1)
    dur  = row.get("Flow Duration", 0)
    pkts = row.get("Total Fwd Packets", 0)
    if dst in (53, 67, 68) and pkts <= 1 and dur < 100:
        return "BENIGN"
    return "PASS"


# ── 5. Consensus ──────────────────────────────────────────────────────────────

def consensus(clf_label: str, clf_conf: float, is_anom: bool) -> str:
    if clf_label != "BENIGN" and clf_conf >= 0.6:
        return "KNOWN_ATTACK"
    if is_anom:
        return "NOVEL_ATTACK"
    return "SAFE"


# ── 6. Main inference loop ────────────────────────────────────────────────────

def run_inference(csv_path: str, model: dict) -> pd.DataFrame:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no GPU detected)"))

    print(f"  Loading CSV …")
    df = pd.read_csv(csv_path, low_memory=False)
    # Strip whitespace from column names (CIC-IDS2017 has trailing spaces)
    df.columns = df.columns.str.strip()
    print(f"  {len(df):,} flows loaded")

    # Stage 1 — Rule engine (row-by-row, uses raw features)
    stage1_labels = [rule_verdict(row) for row in df.to_dict("records")]
    pass_mask = np.array([v == "PASS" for v in stage1_labels])
    n_fast = (~pass_mask).sum()
    print(f"  Stage 1: {n_fast:,} flows cleared by rule engine → {pass_mask.sum():,} to AI stages")

    results = []
    if pass_mask.sum() > 0:
        df_pass = df[pass_mask].reset_index(drop=True)

        # Stage 2 — Classifier
        X_clf = preprocess(df_pass, model, "clf")
        raw_pred = model["clf"].predict(X_clf)
        if raw_pred and isinstance(raw_pred[0], tuple):
            clf_labels  = [p[0] for p in raw_pred]
            clf_confs   = [p[1] for p in raw_pred]
        else:
            clf_labels = list(raw_pred)
            clf_confs  = [1.0] * len(clf_labels)

        # Stage 3 — Anomaly detector
        X_det = preprocess(df_pass, model, "det")
        anom_scores = model["det"].score_batch(X_det)
        threshold   = model["det"].threshold or model["meta"].get("ae_threshold", 0.0)
        is_anoms    = anom_scores > threshold

        for i in range(len(df_pass)):
            verdict = consensus(clf_labels[i], clf_confs[i], is_anoms[i])
            results.append({
                "flow_idx":     df_pass.index[i] if hasattr(df_pass, "index") else i,
                "stage":        2 if verdict == "KNOWN_ATTACK" else 3 if verdict == "NOVEL_ATTACK" else 2,
                "clf_label":    clf_labels[i],
                "clf_conf":     round(float(clf_confs[i]), 4),
                "anom_score":   round(float(anom_scores[i]), 6),
                "is_anomaly":   bool(is_anoms[i]),
                "verdict":      verdict,
            })

    # Add rule-engine BENIGN rows
    rule_rows = [{"flow_idx": i, "stage": 1, "clf_label": "BENIGN",
                  "clf_conf": 1.0, "anom_score": 0.0, "is_anomaly": False,
                  "verdict": "SAFE"}
                 for i, v in enumerate(stage1_labels) if v == "BENIGN"]

    all_rows = rule_rows + results
    out_df = pd.DataFrame(all_rows).sort_values("flow_idx").reset_index(drop=True)
    return out_df


# ── 7. CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AnomalyNetFlow standalone inference — no API server needed")
    parser.add_argument("csv", help="Path to CIC-IDS2017-format CSV file")
    parser.add_argument("--model", default=None,
                        help="Path to model artefacts directory "
                             "(default: latest trained model in models/)")
    parser.add_argument("--out", default=None,
                        help="Save predictions to this CSV file")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top flagged flows to print (default: 20)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"ERROR: CSV file not found: {args.csv}")

    artefact_dir = args.model or find_latest_model()
    print(f"\nModel:  {artefact_dir}")

    with open(os.path.join(artefact_dir, "metadata.json")) as f:
        meta = json.load(f)
    clf_display = (meta.get("classifier_type") or _infer_type(artefact_dir, "classifier"))
    det_display = (meta.get("detector_type")   or _infer_type(artefact_dir, "detector"))
    print(f"Trained: {clf_display} + {det_display}")
    print(f"Features: {meta.get('n_features_final', '?')} "
          f"(from {meta.get('n_features_initial', '?')} raw)")
    print(f"AE threshold (p{int(meta.get('anomaly_threshold_percentile', 95))}): "
          f"{meta.get('ae_threshold', '?')}")
    print()

    print("Loading artefacts …")
    t0 = time.time()
    model = load_model(artefact_dir)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print()

    print("Running 3-stage inference …")
    t0 = time.time()
    results = run_inference(args.csv, model)
    elapsed = time.time() - t0

    # Summary
    counts = results["verdict"].value_counts()
    total  = len(results)
    print(f"\n{'='*55}")
    print(f"  Results: {total:,} flows in {elapsed:.1f}s "
          f"({total/elapsed:,.0f} flows/s)")
    print(f"{'='*55}")
    for verdict in ["SAFE", "KNOWN_ATTACK", "NOVEL_ATTACK"]:
        n = counts.get(verdict, 0)
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {verdict:<15} {n:>7,}  ({pct:5.1f}%)  {bar}")
    print(f"{'='*55}")

    # Top flagged flows
    flagged = results[results["verdict"] != "SAFE"].nlargest(args.top, "anom_score")
    if not flagged.empty:
        print(f"\nTop {min(args.top, len(flagged))} flagged flows:")
        print(flagged[["flow_idx","verdict","clf_label","clf_conf","anom_score"]].to_string(index=False))

    if args.out:
        results.to_csv(args.out, index=False)
        print(f"\nSaved {len(results):,} predictions → {args.out}")

    print()


if __name__ == "__main__":
    main()

"""
FeatureEngineeringPipeline — reusable 4-stage feature engineering pipeline.

Stages (all fitted on the train split, applied to val/test):
  1. VarianceThreshold  — drop near-constant features
  2. Correlation pruning — drop one from each highly-correlated pair
  3. MISelection (opt)  — keep top-K by mutual information with label
  4. Scaler (per-path)  — separate scalers for classifier and detector paths

The pipeline is serialised alongside the model artefacts so the same
transformation is applied identically at inference time.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.preprocessing import (
    MaxAbsScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def make_scaler(scaler_type: str) -> Any:
    """Return a fresh unfitted scaler, or None when scaler_type='none'."""
    match scaler_type.lower():
        case "robust":   return RobustScaler()
        case "standard": return StandardScaler()
        case "quantile": return QuantileTransformer(output_distribution="normal",
                                                    random_state=42)
        case "maxabs":   return MaxAbsScaler()
        case "none":     return None
        case _:
            raise ValueError(
                f"Unknown scaler_type '{scaler_type}'. "
                "Choose: robust | standard | quantile | maxabs | none"
            )


@dataclass
class FEResult:
    """Output of FeatureEngineeringPipeline.fit_transform()."""
    # Feature arrays after all stages (unscaled — scaling is path-specific)
    X_train_sel: np.ndarray
    X_val_sel:   np.ndarray
    X_test_sel:  np.ndarray
    # Fitted artefacts for serialisation
    vt:               VarianceThreshold
    corr_mask:        np.ndarray     # bool mask applied after VT
    feature_selector: Any            # SelectKBest | None
    # Human-readable names at each stage
    feature_names_selected: list[str]
    # Summary counters
    n_initial:  int
    n_after_vt: int
    n_after_corr: int
    n_final:    int


class FeatureEngineeringPipeline:
    """
    Stateless factory — call fit_transform() to get a FEResult.
    All state lives in FEResult; serialise with save_artefacts().
    """

    def __init__(
        self,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
        feature_selection_k: int = 0,
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.feature_selection_k = feature_selection_k

    def fit_transform(
        self,
        X_train_raw: np.ndarray,
        X_val_raw:   np.ndarray,
        X_test_raw:  np.ndarray,
        y_train:     np.ndarray,
        all_feature_names: list[str],
        on_progress: Callable[[str], None] | None = None,
    ) -> FEResult:
        def p(msg: str):
            if on_progress:
                on_progress(msg)

        n_initial = X_train_raw.shape[1]

        # ── Stage 1: Variance threshold ────────────────────────────────────
        p(f"Step 1/3 — VarianceThreshold(threshold={self.variance_threshold})"
          f" on {n_initial} features …")
        vt = VarianceThreshold(threshold=self.variance_threshold)
        X_train = vt.fit_transform(X_train_raw)
        X_val   = vt.transform(X_val_raw)
        X_test  = vt.transform(X_test_raw)
        n_after_vt = X_train.shape[1]
        vt_names = [all_feature_names[i] for i in np.where(vt.get_support())[0]]
        p(f"  → {n_initial - n_after_vt} dropped  ({n_after_vt} remaining)")

        # ── Stage 2: Correlation pruning ───────────────────────────────────
        p(f"Step 2/3 — Correlation pruning (|r| > {self.correlation_threshold}) …")
        corr = pd.DataFrame(X_train).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop_cols = {col for col in upper.columns
                     if any(upper[col] > self.correlation_threshold)}
        corr_mask = np.array([i not in drop_cols for i in range(X_train.shape[1])])
        X_train = X_train[:, corr_mask]
        X_val   = X_val[:, corr_mask]
        X_test  = X_test[:, corr_mask]
        n_after_corr = X_train.shape[1]
        sel_names = [vt_names[i] for i in np.where(corr_mask)[0]]
        p(f"  → {n_after_vt - n_after_corr} dropped  ({n_after_corr} remaining)")

        # ── Stage 3: MI SelectKBest (optional) ────────────────────────────
        feature_selector = None
        if self.feature_selection_k > 0:
            k = min(self.feature_selection_k, X_train.shape[1])
            p(f"Step 3/3 — SelectKBest(mutual_info_classif, k={k}) …")
            feature_selector = SelectKBest(mutual_info_classif, k=k)
            X_train = feature_selector.fit_transform(X_train, y_train)
            X_val   = feature_selector.transform(X_val)
            X_test  = feature_selector.transform(X_test)
            sel_names = [sel_names[i] for i in np.where(feature_selector.get_support())[0]]
            p(f"  → {n_after_corr - k} dropped  ({k} remaining)")
        else:
            p(f"Step 3/3 — MI selection skipped (FEATURE_SELECTION_K=0); "
              f"keeping all {n_after_corr}")

        n_final = X_train.shape[1]
        p(f"Feature engineering: {n_initial} → {n_final} "
          f"({n_initial - n_final} removed total)")

        return FEResult(
            X_train_sel=X_train,
            X_val_sel=X_val,
            X_test_sel=X_test,
            vt=vt,
            corr_mask=corr_mask,
            feature_selector=feature_selector,
            feature_names_selected=sel_names,
            n_initial=n_initial,
            n_after_vt=n_after_vt,
            n_after_corr=n_after_corr,
            n_final=n_final,
        )

    # ── Serialisation ──────────────────────────────────────────────────────

    @staticmethod
    def save_artefacts(result: FEResult, artefact_dir: str) -> None:
        joblib.dump(result.vt,        os.path.join(artefact_dir, "variance_threshold.joblib"))
        joblib.dump(result.corr_mask, os.path.join(artefact_dir, "corr_mask.joblib"))
        if result.feature_selector is not None:
            joblib.dump(result.feature_selector,
                        os.path.join(artefact_dir, "feature_selector.joblib"))

    @staticmethod
    def apply_scaler(
        result: FEResult,
        scaler_type: str,
        artefact_dir: str,
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit a scaler on result.X_train_sel, transform val/test, save to artefact_dir.
        Returns (X_train_scaled, X_val_scaled, X_test_scaled).
        """
        scaler = make_scaler(scaler_type)
        if scaler is not None:
            X_tr = scaler.fit_transform(result.X_train_sel)
            X_v  = scaler.transform(result.X_val_sel)
            X_te = scaler.transform(result.X_test_sel)
            joblib.dump(scaler, os.path.join(artefact_dir, filename))
        else:
            X_tr, X_v, X_te = (result.X_train_sel, result.X_val_sel, result.X_test_sel)
        return X_tr, X_v, X_te

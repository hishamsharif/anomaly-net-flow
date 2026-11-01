"""
Read-only pipeline dashboard endpoints.

These power the Dataset / Preprocess / EDA / Feature Engineering tabs in the UI.
All data is derived from the active model's metrics_json (saved during training)
plus a live directory scan for the dataset tab.
"""
import os

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.db.crud import get_active_model_version
from api.config import settings

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/dataset")
async def dataset_info():
    """
    Scan the dataset directory and return file names + sizes.
    Works even before training — just shows what CSV files are present.
    """
    data_dir = settings.bundled_dataset_dir
    files = []
    total_size = 0
    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(data_dir, fname)
            if not os.path.isfile(fpath):
                # Broken symlink — file visible in directory but target missing
                # (common in Docker when the symlink target path isn't mounted)
                files.append({"name": fname, "size_mb": None, "broken_link": True})
                continue
            size = os.path.getsize(fpath)
            total_size += size
            files.append({
                "name": fname,
                "size_mb": round(size / (1024 * 1024), 1),
            })
    return {
        "data_dir": data_dir,
        "csv_count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 1),
        "files": files,
    }


@router.get("/stats")
async def pipeline_stats(db: AsyncSession = Depends(get_db)):
    """
    Return pipeline statistics from the active model's metrics_json.
    Used by Preprocess, EDA, and Feature Engineering tabs.
    Returns null fields when no model has been trained yet.
    """
    mv = await db.run_sync(get_active_model_version)
    if mv is None:
        return {"model_id": None, "stats": None}

    m = mv.metrics_json or {}
    return {
        "model_id": mv.id,
        "classifier_type": mv.classifier_type,
        "detector_type": mv.detector_type,
        "stats": {
            # Split sizes
            "train_rows_raw":          m.get("train_rows_raw"),
            "train_rows_after_smote":  m.get("train_rows"),
            "val_rows":                m.get("val_rows"),
            "test_rows":               m.get("test_rows"),
            # Class distribution (before SMOTE)
            "class_distribution_train": m.get("class_distribution_train", {}),
            "classes":                 m.get("classes", []),
            # SMOTE
            "smote_before_total":      m.get("smote_before_total"),
            "smote_after_total":       m.get("smote_after_total"),
            "smote_n_minority":        m.get("smote_n_minority"),
            "smote_target_per_class":  m.get("smote_target_per_class"),
            # Feature engineering funnel
            "n_features_initial":      m.get("n_features_initial"),
            "n_features_after_vt":     m.get("n_features_after_vt"),
            "n_features_after_corr":   m.get("n_features_after_corr"),
            "n_features_final":        m.get("n_features_final"),
            "variance_threshold":        m.get("variance_threshold"),
            "correlation_threshold":     m.get("correlation_threshold"),
            "feature_selection_k":       m.get("feature_selection_k"),
            "feature_selection_method":  m.get("feature_selection_method"),
            "clf_scaler_type":           m.get("clf_scaler_type"),
            "det_scaler_type":           m.get("det_scaler_type"),
            # Top features by classifier importance
            "top_features":            m.get("top_features", []),
            "selected_feature_names":  m.get("selected_feature_names", []),
            # Evaluation extras
            "recall":                        m.get("recall"),
            "pr_auc":                        m.get("pr_auc"),
            "known_attack_f1":               m.get("known_attack_f1"),
            "novel_attack_detection_rate":   m.get("novel_attack_detection_rate"),
            "novel_attack_test_rows":        m.get("novel_attack_test_rows"),
            "novel_attack_detected":         m.get("novel_attack_detected"),
            # Training config
            "ae_threshold":            m.get("ae_threshold"),
            "training_duration_s":     m.get("training_duration_s"),
        },
    }

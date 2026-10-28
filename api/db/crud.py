"""
Synchronous DB helpers for the background training thread.
FastAPI routes use async SQLAlchemy directly; training threads use these.
"""
from __future__ import annotations
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from api.db.models import ModelVersion, TrainingSavepoint, TrainingRun


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── ModelVersion ──────────────────────────────────────────────────────────────

def create_model_version(
    db: Session,
    classifier_type: str,
    detector_type: str,
    status: str = "training",
) -> ModelVersion:
    """Create a ModelVersion row. artefact_path and name are set afterwards via
    set_model_version_path_and_name() once the auto-increment ID is known."""
    mv = ModelVersion(
        classifier_type=classifier_type,
        detector_type=detector_type,
        artefact_path="",
        status=status,
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


def set_model_version_path_and_name(
    db: Session, version_id: int, artefact_path: str, name: str
) -> ModelVersion:
    """Update artefact_path and name once the versioned directory has been determined."""
    mv = db.get(ModelVersion, version_id)
    mv.artefact_path = artefact_path
    mv.name = name
    db.commit()
    db.refresh(mv)
    return mv


def activate_model_version(db: Session, version_id: int, metrics: dict) -> ModelVersion:
    """Set this version active, archive any previous active version."""
    prev = db.query(ModelVersion).filter(ModelVersion.status == "active").first()
    if prev:
        prev.status = "archived"

    mv = db.get(ModelVersion, version_id)
    mv.status = "active"
    mv.metrics_json = metrics
    db.commit()
    db.refresh(mv)
    return mv


def fail_model_version(db: Session, version_id: int, error: str) -> None:
    mv = db.get(ModelVersion, version_id)
    if mv:
        mv.status = "failed"
        mv.metrics_json = {"error": error}
        db.commit()


def get_active_model_version(db: Session) -> ModelVersion | None:
    return db.query(ModelVersion).filter(ModelVersion.status == "active").first()


def list_model_versions(db: Session) -> list[ModelVersion]:
    return db.query(ModelVersion).order_by(ModelVersion.id.desc()).all()


# ── TrainingSavepoint ─────────────────────────────────────────────────────────

def create_savepoint(
    db: Session, model_version_id: int, max_flow_id: int, row_count: int
) -> TrainingSavepoint:
    sp = TrainingSavepoint(
        model_version_id=model_version_id,
        max_flow_id=max_flow_id,
        row_count=row_count,
    )
    db.add(sp)
    db.commit()
    db.refresh(sp)
    return sp


def get_savepoint_for_version(db: Session, version_id: int) -> TrainingSavepoint | None:
    return (
        db.query(TrainingSavepoint)
        .filter(TrainingSavepoint.model_version_id == version_id)
        .first()
    )


# ── TrainingRun ───────────────────────────────────────────────────────────────

def create_training_run(
    db: Session, run_id: str, mode: str, model_version_id: int
) -> TrainingRun:
    run = TrainingRun(
        celery_task_id=run_id,   # reused column — stores our UUID run_id
        mode=mode,
        model_version_id=model_version_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def finish_training_run(
    db: Session, run_id: str, row_count: int, error: str | None = None
) -> None:
    run = (
        db.query(TrainingRun).filter(TrainingRun.celery_task_id == run_id).first()
    )
    if run:
        run.finished_at = _now()
        run.row_count = row_count
        run.error = error
        db.commit()

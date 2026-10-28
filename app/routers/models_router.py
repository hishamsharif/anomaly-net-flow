from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from app.database import get_db
from app.db.models import ModelVersion

router = APIRouter(prefix="/models", tags=["models"])


class ModelVersionOut(BaseModel):
    id: int
    name: str | None
    classifier_type: str
    detector_type: str
    artefact_path: str
    status: str
    metrics: dict[str, Any]
    created_at: str

    @classmethod
    def from_orm(cls, mv: ModelVersion) -> "ModelVersionOut":
        return cls(
            id=mv.id,
            name=mv.name,
            classifier_type=mv.classifier_type,
            detector_type=mv.detector_type,
            artefact_path=mv.artefact_path,
            status=mv.status,
            metrics=mv.metrics_json or {},
            created_at=mv.created_at.isoformat(),
        )


@router.get("/active")
async def get_active_model(db: AsyncSession = Depends(get_db)):
    """
    Returns the active model version or {"model": null}.
    UI uses this on load: null → show 'Train Initial Model' button.
    """
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.status == "active")
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        return {"model": None}
    return {"model": ModelVersionOut.from_orm(mv)}


@router.get("/")
async def list_models(db: AsyncSession = Depends(get_db)):
    """All model versions, newest first."""
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.id.desc())
    )
    versions = result.scalars().all()
    return {"models": [ModelVersionOut.from_orm(mv) for mv in versions]}


@router.post("/{model_id}/promote")
async def promote_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Manually set a model version as active, archive the current active."""
    target = await db.get(ModelVersion, model_id)
    if target is None:
        raise HTTPException(404, f"Model version {model_id} not found")
    if target.status == "training":
        raise HTTPException(400, "Cannot promote a model that is still training")

    # Archive current active
    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.status == "active")
        .values(status="archived")
    )

    target.status = "active"
    await db.commit()
    await db.refresh(target)

    # Reload into model_store
    from app.learning.model_store import model_store
    import os
    if os.path.isdir(target.artefact_path):
        model_store.load_from_dir(
            version_id=target.id,
            artefact_dir=target.artefact_path,
            classifier_type=target.classifier_type,
            detector_type=target.detector_type,
            metrics=target.metrics_json or {},
        )

    return {"promoted": ModelVersionOut.from_orm(target)}

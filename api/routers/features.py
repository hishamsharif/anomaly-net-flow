from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db

router = APIRouter(prefix="/features", tags=["features"])


@router.get("/")
async def get_feature_registry(db: AsyncSession = Depends(get_db)):
    """
    Single source of truth for the feature list required by the active model.
    UI and inference adapters read from here — no hardcoded column lists anywhere.
    Returns empty lists if no model has been trained yet.
    """
    import json, os
    from sqlalchemy import select
    from api.db.models import ModelVersion

    result = await db.execute(
        select(ModelVersion).where(ModelVersion.status == "active")
    )
    mv = result.scalar_one_or_none()

    if mv is None:
        return {"model_active": False, "required_features": [], "selected_features": []}

    metadata_path = os.path.join(mv.artefact_path, "metadata.json")
    if not os.path.isfile(metadata_path):
        return {"model_active": True, "required_features": [], "selected_features": []}

    with open(metadata_path) as f:
        meta = json.load(f)

    return {
        "model_active": True,
        "model_version_id": mv.id,
        "required_features": meta.get("feature_names", []),
        "selected_features": meta.get("selected_feature_names", []),
        "n_features_initial": meta.get("n_features_initial"),
        "n_features_final": meta.get("n_features_final"),
    }

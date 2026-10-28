from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

router = APIRouter(prefix="/features", tags=["features"])


@router.get("/")
async def get_feature_registry(db: AsyncSession = Depends(get_db)):
    """
    Single source of truth for required / optional feature split.
    UI and inference service read from here — no hardcoded feature lists.
    """
    # TODO: query FeatureRegistry for active model version
    raise HTTPException(501, "Not yet implemented")

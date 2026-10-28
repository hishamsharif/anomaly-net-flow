from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

router = APIRouter(prefix="/alerts", tags=["alerts"])


class FeedbackRequest(BaseModel):
    # The label the analyst believes is correct
    correct_label: str
    # Whether the model prediction was right or wrong
    # analyst_confirmed | analyst_override
    verdict: str


@router.get("/")
async def list_alerts(limit: int = 50, offset: int = 0, db: AsyncSession = Depends(get_db)):
    # TODO: query Alert table
    raise HTTPException(501, "Not yet implemented")


@router.post("/{alert_id}/feedback")
async def submit_feedback(
    alert_id: int,
    req: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Record analyst verdict.
    Both analyst_confirmed and analyst_override are appended to labelled_flows
    so that fine-tuning benefits from both signals.
    Increments the verdict count toward the K fine-tune threshold.
    """
    if req.verdict not in ("analyst_confirmed", "analyst_override"):
        raise HTTPException(400, "verdict must be 'analyst_confirmed' or 'analyst_override'")
    # TODO: update Alert, append to labelled_flows, return updated metrics
    raise HTTPException(501, "Not yet implemented")


@router.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    """
    Verdict count toward the fine-tune threshold.
    Full TP/FP metrics arrive in the inference pipeline phase.
    """
    from sqlalchemy import select, func
    from app.db.models import Alert
    result = await db.execute(
        select(func.count()).select_from(Alert).where(Alert.analyst_verdict.isnot(None))
    )
    verdict_count = result.scalar_one() or 0
    return {"verdict_count": verdict_count}

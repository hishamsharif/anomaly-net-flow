from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db

router = APIRouter(prefix="/alerts", tags=["alerts"])


class FeedbackRequest(BaseModel):
    # The label the analyst believes is correct
    correct_label: str
    # Whether the model prediction was right or wrong
    # analyst_confirmed | analyst_override
    verdict: str


@router.get("/")
async def list_alerts(limit: int = 50, offset: int = 0, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    from api.db.models import Alert
    result = await db.execute(
        select(Alert).order_by(Alert.id.desc()).offset(offset).limit(limit)
    )
    alerts = result.scalars().all()
    return {
        "alerts": [
            {
                "id": a.id,
                "flow_id": a.flow_id,
                "consensus": a.consensus,
                "severity": a.severity,
                "rule_verdict": a.rule_verdict,
                "predicted_label": a.predicted_label,
                "classifier_confidence": a.classifier_confidence,
                "ae_score": a.ae_score,
                "analyst_verdict": a.analyst_verdict,
                "analyst_label": a.analyst_label,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in alerts
        ],
        "total": len(alerts),
        "offset": offset,
        "limit": limit,
    }


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

    from sqlalchemy import select, func
    from datetime import datetime, timezone
    from api.db.models import Alert, LabelledFlow

    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if alert is None:
        raise HTTPException(404, f"Alert {alert_id} not found")

    # Append verdict to label store
    lf = LabelledFlow(
        features_json={},
        label=req.correct_label,
        source=req.verdict,
        predicted_label=alert.predicted_label,
        model_version_id=alert.model_version_id,
    )
    db.add(lf)
    await db.flush()

    # Update alert with analyst verdict
    alert.analyst_verdict = req.verdict
    alert.analyst_label = req.correct_label
    alert.labelled_flow_id = lf.id
    alert.feedback_at = datetime.now(timezone.utc)
    await db.commit()

    # Return updated verdict count
    count_result = await db.execute(
        select(func.count()).select_from(Alert).where(Alert.analyst_verdict.isnot(None))
    )
    verdict_count = count_result.scalar_one() or 0
    return {"alert_id": alert_id, "verdict": req.verdict, "verdict_count": verdict_count}


@router.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    """
    Verdict count toward the fine-tune threshold.
    Full TP/FP metrics arrive in the inference pipeline phase.
    """
    from sqlalchemy import select, func
    from api.db.models import Alert
    result = await db.execute(
        select(func.count()).select_from(Alert).where(Alert.analyst_verdict.isnot(None))
    )
    verdict_count = result.scalar_one() or 0
    return {"verdict_count": verdict_count}

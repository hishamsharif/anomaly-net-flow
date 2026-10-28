from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.learning.registry import CLASSIFIERS, ANOMALY_DETECTORS
from app import tasks

router = APIRouter(prefix="/training", tags=["training"])


class TrainingStartRequest(BaseModel):
    mode: str = "initial"           # initial | fine_tune
    classifier_type: str = "xgboost"
    detector_type: str = "mlp_ae"


class TrainingStartResponse(BaseModel):
    run_id: str
    mode: str
    status: str = "started"
    events_url: str


@router.post("/start", response_model=TrainingStartResponse, status_code=202)
async def start_training(req: TrainingStartRequest, db: AsyncSession = Depends(get_db)):
    if req.mode not in ("initial", "fine_tune"):
        raise HTTPException(400, f"mode must be 'initial' or 'fine_tune', got '{req.mode}'")
    if req.classifier_type not in CLASSIFIERS:
        raise HTTPException(400, f"Unknown classifier '{req.classifier_type}'")
    if req.detector_type not in ANOMALY_DETECTORS:
        raise HTTPException(400, f"Unknown detector '{req.detector_type}'")

    run_id = await tasks.dispatch(req.mode, req.classifier_type, req.detector_type)
    return TrainingStartResponse(
        run_id=run_id,
        mode=req.mode,
        events_url=f"/api/training/events/{run_id}",
    )


@router.get("/events/{run_id}")
async def training_events(run_id: str):
    """SSE stream — yields progress / done / error events for this run."""
    return StreamingResponse(
        tasks.sse_events(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if proxied
        },
    )


@router.get("/available-types")
async def available_types():
    return {
        "classifiers": list(CLASSIFIERS.keys()),
        "detectors": list(ANOMALY_DETECTORS.keys()),
    }

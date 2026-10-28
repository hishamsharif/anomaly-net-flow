from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.db.models import TrainingRun
from core.registry import CLASSIFIERS, ANOMALY_DETECTORS
from api import tasks

router = APIRouter(prefix="/training", tags=["training"])

_DEFAULT_ESTIMATE = {"initial": (1500, "~25 min"), "fine_tune": (480, "~8 min")}


async def _estimate_seconds(db: AsyncSession, mode: str) -> tuple[int, str]:
    """Query avg duration from the last 5 completed runs of the same mode."""
    result = await db.execute(
        select(TrainingRun)
        .where(
            TrainingRun.mode == mode,
            TrainingRun.finished_at.is_not(None),
            TrainingRun.error.is_(None),
        )
        .order_by(TrainingRun.id.desc())
        .limit(5)
    )
    runs = result.scalars().all()
    if runs:
        durations = [
            (r.finished_at - r.started_at).total_seconds()
            for r in runs
            if r.finished_at and r.started_at
        ]
        if durations:
            avg = int(sum(durations) / len(durations))
            mins = max(1, round(avg / 60))
            return avg, f"~{mins} min"
    secs, label = _DEFAULT_ESTIMATE.get(mode, (900, "~15 min"))
    return secs, label


class TrainingStartRequest(BaseModel):
    mode: str = "initial"           # initial | fine_tune
    classifier_type: str = "xgboost"
    detector_type: str = "mlp_ae"


class TrainingStartResponse(BaseModel):
    run_id: str
    mode: str
    acknowledged: bool = True
    estimated_seconds: int = 0
    estimated_label: str = ""
    events_url: str


@router.post("/start", response_model=TrainingStartResponse, status_code=202)
async def start_training(req: TrainingStartRequest, db: AsyncSession = Depends(get_db)):
    if req.mode not in ("initial", "fine_tune"):
        raise HTTPException(400, f"mode must be 'initial' or 'fine_tune', got '{req.mode}'")
    if req.classifier_type not in CLASSIFIERS:
        raise HTTPException(400, f"Unknown classifier '{req.classifier_type}'")
    if req.detector_type not in ANOMALY_DETECTORS:
        raise HTTPException(400, f"Unknown detector '{req.detector_type}'")

    est_secs, est_label = await _estimate_seconds(db, req.mode)
    run_id = await tasks.dispatch(req.mode, req.classifier_type, req.detector_type)
    return TrainingStartResponse(
        run_id=run_id,
        mode=req.mode,
        estimated_seconds=est_secs,
        estimated_label=est_label,
        events_url=f"/api/training/events/{run_id}",
    )


@router.get("/status/{run_id}")
async def training_status(run_id: str):
    """Polling endpoint — returns current step % when SSE is not connected."""
    state = tasks.get_step_state(run_id)
    if state is None:
        raise HTTPException(404, f"No active training run '{run_id}'")
    return state


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
    """
    Lists the available model types for each pipeline stage.
    supervised_classifiers  = Stage 2 (trained with attack labels — identifies known categories)
    anomaly_detectors       = Stage 3 (trained on benign-only — flags novel/zero-day patterns)
    """
    return {
        "supervised_classifiers": list(CLASSIFIERS.keys()),
        "anomaly_detectors": list(ANOMALY_DETECTORS.keys()),
        # Legacy keys kept for backwards compatibility
        "classifiers": list(CLASSIFIERS.keys()),
        "detectors": list(ANOMALY_DETECTORS.keys()),
    }

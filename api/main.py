import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.database import init_db
from api.routers import predict, training, models_router, alerts, features


def _migrate_db_location_on_startup() -> None:
    """Move anomaly_net_flow.db from project root → data/ if the old file still exists."""
    old = "anomaly_net_flow.db"
    new = os.path.join("data", "anomaly_net_flow.db")
    if os.path.isfile(old) and not os.path.isfile(new):
        os.makedirs("data", exist_ok=True)
        os.rename(old, new)
        print(f"[startup] Migrated database: {old} → {new}")


def _mark_stale_training_on_startup() -> None:
    """
    Mark any ModelVersion rows left in 'training' status as 'failed'.
    These are always orphans — the training thread cannot survive a process restart.
    """
    from api.database import get_sync_db
    from api.db.models import ModelVersion
    db = get_sync_db()
    try:
        stale = db.query(ModelVersion).filter(ModelVersion.status == "training").all()
        if stale:
            for mv in stale:
                mv.status = "failed"
                mv.metrics_json = {"error": "Training was interrupted by an API restart"}
            db.commit()
            print(f"[startup] Marked {len(stale)} orphaned 'training' model(s) as 'failed'")
    except Exception as exc:
        print(f"[startup] Warning: could not clean up stale training rows: {exc}")
    finally:
        db.close()


def _load_active_model_on_startup() -> None:
    """
    Called during lifespan startup.
    Queries SQLite for the active ModelVersion and loads its artefacts into the
    in-memory model_store so inference is available immediately after restart.
    """
    from api.database import get_sync_db
    from api.db.crud import get_active_model_version
    from core.store import model_store

    db = get_sync_db()
    try:
        mv = get_active_model_version(db)
        if mv is None:
            print("[startup] No active model found — UI will prompt for initial training.")
            return

        artefact_dir = mv.artefact_path
        if not os.path.isdir(artefact_dir):
            print(f"[startup] Active model v{mv.id} artefact directory missing: {artefact_dir}")
            return

        print(f"[startup] Loading model v{mv.id} from {artefact_dir} …")
        model_store.load_from_dir(
            version_id=mv.id,
            artefact_dir=artefact_dir,
            classifier_type=mv.classifier_type,
            detector_type=mv.detector_type,
            metrics=mv.metrics_json or {},
        )
        print(f"[startup] Model v{mv.id} ready  "
              f"(F1={mv.metrics_json.get('f1', '?'):.4f})")
    except Exception as exc:
        print(f"[startup] Warning: could not load active model: {exc}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _migrate_db_location_on_startup()
    await init_db()
    _mark_stale_training_on_startup()
    _load_active_model_on_startup()
    yield


app = FastAPI(
    title="AnomalyNetFlow",
    description="Incremental network intrusion detection with human-in-the-loop labelling.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(models_router.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(features.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}

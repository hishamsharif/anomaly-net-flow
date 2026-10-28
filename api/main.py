import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.database import init_db
from api.routers import predict, training, models_router, alerts, features


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
    await init_db()
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

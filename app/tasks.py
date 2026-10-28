"""
In-process background training runner.

Training runs in a ThreadPoolExecutor (CPU-bound, doesn't block the event loop).
Progress events are pushed to a per-run asyncio.Queue and streamed to the UI via SSE.
DB writes use a sync SQLAlchemy session (safe alongside the async session on SQLite).
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="training")

# run_id → asyncio.Queue of {"event": str, "data": str}
_queues: dict[str, asyncio.Queue] = {}
_qlock = threading.Lock()


# ── Queue helpers ─────────────────────────────────────────────────────────────

def _get_or_create_queue(run_id: str) -> asyncio.Queue:
    with _qlock:
        if run_id not in _queues:
            _queues[run_id] = asyncio.Queue()
        return _queues[run_id]


def _put(run_id: str, event: str, data: Any, loop: asyncio.AbstractEventLoop) -> None:
    q = _get_or_create_queue(run_id)
    loop.call_soon_threadsafe(q.put_nowait, {"event": event, "data": str(data)})


def _cleanup_later(run_id: str, delay: float = 120.0) -> None:
    def _rm():
        time.sleep(delay)
        with _qlock:
            _queues.pop(run_id, None)
    threading.Thread(target=_rm, daemon=True).start()


# ── Training workers ──────────────────────────────────────────────────────────

def _run_initial(run_id: str, classifier_type: str, detector_type: str) -> None:
    from app.config import settings
    from app.database import get_sync_db
    from app.db.crud import (
        create_model_version, activate_model_version,
        fail_model_version, create_savepoint, create_training_run,
        finish_training_run,
    )
    from app.ml.model_store import model_store
    from training.pipelines.initial import run as pipeline_run
    from training.data_loader import get_max_flow_id

    loop = asyncio.get_event_loop()

    def progress(msg: str):
        _put(run_id, "progress", msg, loop)

    db = get_sync_db()
    version_id = None
    try:
        # Create ModelVersion row immediately so we have an ID for the artefact dir
        artefact_dir = os.path.join(settings.model_artefact_dir, f"v{run_id[:8]}_initial")
        mv = create_model_version(db, classifier_type, detector_type, artefact_dir)
        version_id = mv.id
        create_training_run(db, run_id, "initial", version_id)

        result = pipeline_run(
            artefact_dir=artefact_dir,
            bundled_dir=settings.bundled_dataset_dir,
            classifier_type=classifier_type,
            detector_type=detector_type,
            on_progress=progress,
        )

        # Record savepoint (max_flow_id at training start = 0 for initial)
        sp = create_savepoint(db, version_id, 0, result.get("train_rows", 0))

        metrics = {
            "f1": result["f1"],
            "far": result["far"],
            "amr": result["amr"],
            "train_rows": result.get("train_rows", 0),
            "val_rows": result.get("val_rows", 0),
            "test_rows": result.get("test_rows", 0),
            "ae_threshold": result.get("ae_threshold"),
            "savepoint_id": sp.id,
        }
        mv = activate_model_version(db, version_id, metrics)
        finish_training_run(db, run_id, result.get("train_rows", 0))

        # Load into in-memory store so inference is immediately available
        progress("Loading model into memory …")
        model_store.load_from_dir(
            version_id=mv.id,
            artefact_dir=artefact_dir,
            classifier_type=classifier_type,
            detector_type=detector_type,
            metrics=metrics,
        )

        _put(run_id, "done", f"Training complete — model v{mv.id} active", loop)

    except Exception as exc:
        err = str(exc)
        progress(f"Error: {err}")
        if version_id:
            fail_model_version(db, version_id, err)
            finish_training_run(db, run_id, 0, error=err)
        traceback.print_exc()
        _put(run_id, "error", err, loop)
    finally:
        db.close()
        _cleanup_later(run_id)


def _run_fine_tune(run_id: str, classifier_type: str, detector_type: str) -> None:
    from app.config import settings
    from app.database import get_sync_db
    from app.db.crud import (
        create_model_version, activate_model_version,
        fail_model_version, create_savepoint, create_training_run,
        finish_training_run, get_active_model_version, get_savepoint_for_version,
    )
    from app.ml.model_store import model_store
    from training.pipelines.fine_tune import run as pipeline_run

    loop = asyncio.get_event_loop()

    def progress(msg: str):
        _put(run_id, "progress", msg, loop)

    db = get_sync_db()
    version_id = None
    try:
        # Fetch previous active version for artefact path + savepoint
        prev_mv = get_active_model_version(db)
        if prev_mv is None:
            raise RuntimeError("No active model — run initial training first.")
        prev_sp = get_savepoint_for_version(db, prev_mv.id)
        prev_max_id = prev_sp.max_flow_id if prev_sp else 0

        artefact_dir = os.path.join(settings.model_artefact_dir, f"v{run_id[:8]}_ft")
        mv = create_model_version(db, classifier_type, detector_type, artefact_dir)
        version_id = mv.id
        create_training_run(db, run_id, "fine_tune", version_id)

        result = pipeline_run(
            prev_artefact_dir=prev_mv.artefact_path,
            artefact_dir=artefact_dir,
            prev_savepoint_max_id=prev_max_id,
            classifier_type=classifier_type,
            detector_type=detector_type,
            on_progress=progress,
        )

        new_max_id = result.get("new_savepoint_max_id", 0)
        sp = create_savepoint(db, version_id, new_max_id, result.get("cumulative_rows", 0))

        metrics = {
            "f1": result["f1"],
            "far": result["far"],
            "amr": result["amr"],
            "cumulative_rows": result.get("cumulative_rows", 0),
            "incremental_rows": result.get("incremental_rows", 0),
            "ae_threshold": result.get("ae_threshold"),
            "savepoint_id": sp.id,
            "prev_version_id": prev_mv.id,
        }
        mv = activate_model_version(db, version_id, metrics)
        finish_training_run(db, run_id, result.get("cumulative_rows", 0))

        progress("Loading updated model into memory …")
        model_store.load_from_dir(
            version_id=mv.id,
            artefact_dir=artefact_dir,
            classifier_type=classifier_type,
            detector_type=detector_type,
            metrics=metrics,
        )

        _put(run_id, "done", f"Fine-tune complete — model v{mv.id} active", loop)

    except Exception as exc:
        err = str(exc)
        progress(f"Error: {err}")
        if version_id:
            fail_model_version(db, version_id, err)
            finish_training_run(db, run_id, 0, error=err)
        traceback.print_exc()
        _put(run_id, "error", err, loop)
    finally:
        db.close()
        _cleanup_later(run_id)


# ── Public API ────────────────────────────────────────────────────────────────

async def dispatch(mode: str, classifier_type: str, detector_type: str) -> str:
    """Submit a training job to the thread pool. Returns a run_id."""
    run_id = str(uuid.uuid4())
    _get_or_create_queue(run_id)  # create queue before thread starts

    target = _run_initial if mode == "initial" else _run_fine_tune
    _executor.submit(target, run_id, classifier_type, detector_type)
    return run_id


async def sse_events(run_id: str):
    """Async generator yielding SSE-formatted strings."""
    q = _get_or_create_queue(run_id)
    while True:
        try:
            item = await asyncio.wait_for(q.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield "event: heartbeat\ndata: ping\n\n"
            continue

        yield f"event: {item['event']}\ndata: {item['data']}\n\n"

        if item["event"] in ("done", "error"):
            break

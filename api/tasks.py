"""
In-process background training runner.

Training runs in a ThreadPoolExecutor (CPU-bound, doesn't block the event loop).
Progress events are pushed to a per-run asyncio.Queue and streamed to the UI via SSE.
DB writes use a sync SQLAlchemy session (safe alongside the async session on SQLite).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_logger = logging.getLogger("anomaly_net_flow.training")

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="training")

# run_id → asyncio.Queue of {"event": str, "data": str}
_queues: dict[str, asyncio.Queue] = {}
_qlock = threading.Lock()

# run_id → {"pct": int, "mode": str, "started_at": float}
_step_state: dict[str, dict] = {}
_slock = threading.Lock()


# ── Queue helpers ─────────────────────────────────────────────────────────────

def _get_or_create_queue(run_id: str) -> asyncio.Queue:
    with _qlock:
        if run_id not in _queues:
            _queues[run_id] = asyncio.Queue()
        return _queues[run_id]


def _put(run_id: str, event: str, data: Any, loop: asyncio.AbstractEventLoop) -> None:
    q = _get_or_create_queue(run_id)
    loop.call_soon_threadsafe(q.put_nowait, {"event": event, "data": str(data)})
    if event == "step":
        with _slock:
            if run_id in _step_state:
                _step_state[run_id]["pct"] = int(data)


def _cleanup_later(run_id: str, delay: float = 600.0) -> None:
    def _rm():
        time.sleep(delay)
        with _qlock:
            _queues.pop(run_id, None)
        with _slock:
            _step_state.pop(run_id, None)
    threading.Thread(target=_rm, daemon=True).start()


# ── Training workers ──────────────────────────────────────────────────────────

_CLF_DISPLAY = {
    "xgboost": "XGBoost",
    "random_forest": "RF",
}
_DET_DISPLAY = {
    "mlp_ae": "MLP-AE",
    "isolation_forest": "IF",
}

# Ordered (keyword, percent) pairs — first match wins.
# The 'done' event always sends 100; these cover the pipeline body.
_STEP_INITIAL: list[tuple[str, int]] = [
    ("Loading CIC-IDS2017",      5),
    ("Splitting ",               12),
    ("Step 1/3",                 20),
    ("Step 2/3",                 30),
    ("Step 3/3",                 38),
    ("Feature engineering:",     45),
    ("Scaler --",                52),
    ("Applying SMOTE",           60),
    ("Training ",                68),   # "Training xgboost classifier …"
    ("Evaluating classifier",    78),
    ("anomaly detector",         84),   # "Training mlp_ae anomaly detector …"
    ("Top-20 features",          88),
    ("Saving model",             92),
    ("Loading model",            97),
]
_STEP_FINE_TUNE: list[tuple[str, int]] = [
    ("Fetching incremental",     5),
    ("Loading previous",        10),
    ("Step 1/3",                18),
    ("Step 2/3",                28),
    ("Step 3/3",                36),
    ("Feature engineering:",    44),
    ("Scaler --",               50),
    ("Retraining ",             58),    # "Retraining xgboost …"
    ("Evaluating",              70),
    ("Fine-tuning",             78),    # "Fine-tuning AE …"
    ("Replay buffer",           84),
    ("Saving",                  90),
    ("Loading model",           97),
]


def _make_step_emitter(steps: list[tuple[str, int]], run_id: str, loop: asyncio.AbstractEventLoop):
    """Return a function that, given a message, emits a step event for the first matching keyword."""
    emitted: set[int] = set()

    def emit_step(msg: str) -> None:
        for keyword, pct in steps:
            if keyword.lower() in msg.lower() and pct not in emitted:
                emitted.add(pct)
                _put(run_id, "step", str(pct), loop)  # also updates _step_state via _put
                break

    return emit_step


def _version_name(clf: str, det: str, vid: int) -> str:
    c = _CLF_DISPLAY.get(clf, clf.replace("_", " ").title())
    d = _DET_DISPLAY.get(det, det.replace("_", " ").title())
    return f"{c} + {d} — v{vid}"


def _version_dir(clf: str, det: str, vid: int) -> str:
    return f"{clf}_{det}_v{vid}"


def _run_initial(run_id: str, classifier_type: str, detector_type: str,
                 loop: asyncio.AbstractEventLoop) -> None:
    from api.config import settings
    from api.database import get_sync_db
    from api.db.crud import (
        create_model_version, set_model_version_path_and_name,
        activate_model_version, fail_model_version,
        create_savepoint, create_training_run, finish_training_run,
    )
    from core.store import model_store
    from training.pipelines.initial import run as pipeline_run

    emit_step = _make_step_emitter(_STEP_INITIAL, run_id, loop)

    def progress(msg: str):
        _logger.info("[%s] %s", run_id[:8], msg)
        emit_step(msg)
        _put(run_id, "progress", msg, loop)

    db = get_sync_db()
    version_id = None
    try:
        # Create row first to obtain the auto-increment ID
        mv = create_model_version(db, classifier_type, detector_type)
        version_id = mv.id
        # Now we know the ID — build the human-readable name and dir
        artefact_dir = os.path.join(
            settings.model_artefact_dir,
            _version_dir(classifier_type, detector_type, version_id),
        )
        name = _version_name(classifier_type, detector_type, version_id)
        set_model_version_path_and_name(db, version_id, artefact_dir, name)
        progress(f"Registered {name}")
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

        _put(run_id, "step", "100", loop)
        _put(run_id, "done", f"Training complete — {name} active", loop)

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


def _run_fine_tune(run_id: str, classifier_type: str, detector_type: str,
                   loop: asyncio.AbstractEventLoop) -> None:
    from api.config import settings
    from api.database import get_sync_db
    from api.db.crud import (
        create_model_version, set_model_version_path_and_name,
        activate_model_version, fail_model_version,
        create_savepoint, create_training_run, finish_training_run,
        get_active_model_version, get_savepoint_for_version,
    )
    from core.store import model_store
    from training.pipelines.fine_tune import run as pipeline_run

    emit_step = _make_step_emitter(_STEP_FINE_TUNE, run_id, loop)

    def progress(msg: str):
        _logger.info("[%s] %s", run_id[:8], msg)
        emit_step(msg)
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

        mv = create_model_version(db, classifier_type, detector_type)
        version_id = mv.id
        artefact_dir = os.path.join(
            settings.model_artefact_dir,
            _version_dir(classifier_type, detector_type, version_id),
        )
        name = _version_name(classifier_type, detector_type, version_id)
        set_model_version_path_and_name(db, version_id, artefact_dir, name)
        progress(f"Registered {name}")
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

        _put(run_id, "step", "100", loop)
        _put(run_id, "done", f"Fine-tune complete — {name} active", loop)

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
    with _slock:
        _step_state[run_id] = {"pct": 0, "mode": mode, "started_at": time.time()}

    # Capture the running loop here (async context) so threads can post events
    # back to it via call_soon_threadsafe without calling get_event_loop() from
    # a worker thread (which is deprecated / unreliable in Python 3.10+).
    loop = asyncio.get_running_loop()

    target = _run_initial if mode == "initial" else _run_fine_tune
    _executor.submit(target, run_id, classifier_type, detector_type, loop)
    return run_id


def get_step_state(run_id: str) -> dict | None:
    """Return current step state for polling when SSE is not connected."""
    with _slock:
        s = _step_state.get(run_id)
        return dict(s) if s is not None else None


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

import os
import tempfile
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.config import settings
from api.db.models import Alert
from core.store import model_store

router = APIRouter(prefix="/predict", tags=["inference"])

_ADAPTER_MAP = {
    ".csv": "csv",
    ".pcap": "pcap",
    ".pcapng": "pcap",
}

_SEVERITY: dict[str, str] = {
    "SAFE": "info",
    "KNOWN_ATTACK": "high",
    "NOVEL_ATTACK": "critical",
}


def _rule_verdict(row) -> str:
    """Stage 1: lightweight heuristics. Returns 'BENIGN' (skip AI) or 'PASS' (proceed)."""
    dst = row.get("Destination Port", -1)
    dur = row.get("Flow Duration", 0)
    pkts = row.get("Total Fwd Packets", 0)
    # Trusted: loopback DNS/DHCP, single-packet, sub-100ms — almost certainly not an attack
    if dst in (53, 67, 68) and pkts <= 1 and dur < 100:
        return "BENIGN"
    return "PASS"


def _consensus(clf_label: str, clf_conf: float, is_anom: bool) -> str:
    if clf_label != "BENIGN" and clf_conf >= 0.6:
        return "KNOWN_ATTACK"
    if is_anom:
        return "NOVEL_ATTACK"
    return "SAFE"


@router.post("/batch")
async def predict_batch(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    adapter_key = _ADAPTER_MAP.get(ext)
    if not adapter_key:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use .csv, .pcap or .pcapng")

    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    t_extract_start = time.monotonic()
    try:
        if adapter_key == "csv":
            from inference.adapters.csv_adapter import CSVInferenceAdapter
            df = CSVInferenceAdapter().load(tmp_path)
        else:
            from inference.adapters.pcap_adapter import PcapAdapter
            df = PcapAdapter().load(tmp_path)
    finally:
        os.unlink(tmp_path)
    extraction_ms = int((time.monotonic() - t_extract_start) * 1000)

    if df.empty:
        raise HTTPException(422, "No flows could be extracted from the uploaded file")

    model = model_store.get()
    if model is None:
        raise HTTPException(503, "No trained model loaded — run initial training first")

    t_preproc_start = time.monotonic()
    # Align to model's raw feature list, filling any gaps with 0
    for col in model.feature_names:
        if col not in df.columns:
            df[col] = 0.0
    X_raw = df[model.feature_names].fillna(0).values.astype(np.float32)
    X_clf = model.preprocess_clf(X_raw)
    X_det = model.preprocess_det(X_raw)
    preprocessing_ms = int((time.monotonic() - t_preproc_start) * 1000)

    t_infer_start = time.monotonic()
    # Stage 2 — supervised classifier (batch)
    clf_results: list[tuple[str, float]] = model.classifier.predict(X_clf)

    # Stage 3 — anomaly detector (batch-efficient)
    detector = model.detector
    if hasattr(detector, "score_batch"):
        ae_scores: np.ndarray = detector.score_batch(X_det)
        threshold = getattr(detector, "threshold", 0.0) or 0.0
        anomaly_flags = (ae_scores > threshold).tolist()
    else:
        pairs = [detector.is_anomaly(X_det[i]) for i in range(len(X_det))]
        anomaly_flags = [p[0] for p in pairs]
        ae_scores = np.array([p[1] for p in pairs])
    inference_ms = int((time.monotonic() - t_infer_start) * 1000)

    db_alerts: list[Alert] = []
    response_alerts: list[dict] = []
    summary: dict[str, int] = {}
    both_flagged_count = 0  # classifier says attack AND AE flags anomaly

    for i, (clf_label, clf_conf) in enumerate(clf_results):
        # Stage 1 — rule engine
        rule = _rule_verdict(df.iloc[i])
        ai_skipped = rule == "BENIGN"

        if ai_skipped:
            consensus = "SAFE"
            pred_label = None
            conf_val = None
            ae_val = None
        else:
            is_anom = bool(anomaly_flags[i])
            consensus = _consensus(clf_label, float(clf_conf), is_anom)
            pred_label = clf_label
            conf_val = float(clf_conf)
            ae_val = float(ae_scores[i])
            if clf_label != "BENIGN" and float(clf_conf) >= 0.6 and is_anom:
                both_flagged_count += 1

        severity = _SEVERITY[consensus]
        flow_id = str(uuid.uuid4())

        db_alerts.append(Alert(
            flow_id=flow_id,
            consensus=consensus,
            severity=severity,
            rule_verdict=rule,
            predicted_label=pred_label,
            classifier_confidence=conf_val,
            ae_score=ae_val,
            ai_skipped=ai_skipped,
            model_version_id=model.version_id,
        ))
        response_alerts.append({
            "flow_id": flow_id,
            "consensus": consensus,
            "severity": severity,
            "predicted_label": pred_label,
            "classifier_confidence": conf_val,
            "ae_score": ae_val,
            "ai_skipped": ai_skipped,
        })
        summary[consensus] = summary.get(consensus, 0) + 1

    db.add_all(db_alerts)
    await db.commit()

    return {
        "total": len(response_alerts),
        "model_version_id": model.version_id,
        "summary": {
            **summary,
            "both_flagged_count": both_flagged_count,
        },
        "processing": {
            "extraction_ms":    extraction_ms,
            "preprocessing_ms": preprocessing_ms,
            "inference_ms":     inference_ms,
            "total_ms":         extraction_ms + preprocessing_ms + inference_ms,
        },
        "alerts": response_alerts,
    }

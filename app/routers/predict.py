import io
import os
import tempfile
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings

router = APIRouter(prefix="/predict", tags=["inference"])

_ADAPTER_MAP = {
    ".csv": "csv",
    ".pcap": "pcap",
    ".pcapng": "pcap",
}


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

    # Write to temp file (PcapAdapter needs a path; CSVAdapter uses StringIO)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        if adapter_key == "csv":
            from training.adapters.csv_inference import CSVInferenceAdapter
            df = CSVInferenceAdapter().load(tmp_path)
        else:
            from training.adapters.pcap_adapter import PcapAdapter
            df = PcapAdapter().load(tmp_path)
    finally:
        os.unlink(tmp_path)

    if df.empty:
        raise HTTPException(422, "No flows could be extracted from the uploaded file")

    # TODO: run 3-stage pipeline over df rows, insert alerts, return results
    return {"total": len(df), "message": "Inference pipeline not yet wired — stub"}



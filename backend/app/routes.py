from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from .services.yolo_inference import predict_buildings

router = APIRouter(prefix="/api", tags=["processing"])
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Accept an aerial/drone image and store it for later processing."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    job_id = uuid4().hex
    destination = UPLOAD_DIR / f"{job_id}{suffix}"
    content = await file.read()
    destination.write_bytes(content)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "stored_path": str(destination),
        "status": "uploaded",
    }


@router.post("/predict/{job_id}")
def predict_image(job_id: str):
    """Run building-footprint segmentation for a previously uploaded image."""
    matches = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Upload job not found")

    try:
        result = predict_buildings(matches[0])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"job_id": job_id, "status": "processed", **result}

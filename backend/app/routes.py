from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from .services.geometry import mask_to_polygon_features
from .services.topology import validate_features
from .services.yolo_inference import predict_buildings

router = APIRouter(prefix="/api", tags=["processing"])
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Accept an aerial/drone image and store it for processing."""
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
def predict_image(
    job_id: str,
    confidence: float = Query(0.25, ge=0.0, le=1.0),
):
    """Run building-footprint segmentation and basic geometry validation."""
    matches = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Upload job not found")

    image_path = matches[0]
    try:
        result = predict_buildings(image_path, confidence=confidence)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Uploaded image could not be decoded")
        image_height, image_width = image.shape[:2]
        features = mask_to_polygon_features(
            [item["polygon"] for item in result["detections"]],
            image_width,
            image_height,
        )
        validation = validate_features(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "job_id": job_id,
        "status": "processed",
        "confidence_threshold": confidence,
        "image": {"width": image_width, "height": image_height},
        "features": features,
        "validation": validation,
        **result,
    }

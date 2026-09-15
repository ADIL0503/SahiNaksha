from pathlib import Path
from typing import Any


class SegmentationService:
    """Interface for building-footprint segmentation models.

    The prototype deliberately keeps model loading separate from the API so a
    trained YOLO segmentation model can be plugged in without changing routes.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.model_path is None:
            raise RuntimeError("No segmentation model configured")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Segmentation model not found: {self.model_path}")

        # Model loading is intentionally deferred until a real model is supplied.
        # This keeps the base API runnable on machines without GPU/Ultralytics.
        raise NotImplementedError(
            "YOLO segmentation model loading will be enabled after the model artifact is added."
        )

    def predict(self, image_path: str) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Segmentation model is not loaded")
        return {"image": image_path, "features": []}

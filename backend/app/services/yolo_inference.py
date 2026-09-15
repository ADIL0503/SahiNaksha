from __future__ import annotations

from pathlib import Path
from typing import Any


class YOLOInferenceService:
    """Run building-footprint segmentation with a trained Ultralytics model.

    The Ultralytics import is lazy so the API can still start when the optional
    inference dependency/model is not installed yet.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is required for YOLO inference. "
                    "Install training/requirements-yolo.txt."
                ) from exc

            if not self.model_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(
        self,
        image_path: str | Path,
        confidence: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Return detected building polygons in pixel coordinates."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        model = self._load_model()
        results = model.predict(source=str(path), conf=confidence, verbose=False)
        detections: list[dict[str, Any]] = []

        for result in results:
            masks = getattr(result, "masks", None)
            if masks is None or masks.xy is None:
                continue

            classes = getattr(result, "boxes", None)
            class_ids = classes.cls.tolist() if classes is not None else []
            confidences = classes.conf.tolist() if classes is not None else []

            for index, polygon in enumerate(masks.xy):
                points = [[float(x), float(y)] for x, y in polygon]
                detections.append(
                    {
                        "class_id": int(class_ids[index]) if index < len(class_ids) else 0,
                        "confidence": float(confidences[index]) if index < len(confidences) else confidence,
                        "polygon": points,
                    }
                )

        return detections

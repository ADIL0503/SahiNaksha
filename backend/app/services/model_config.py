from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_PATH = Path("backend/models/sahinaksha-building-seg.pt")


def get_model_path() -> Path:
    """Return the configured YOLO model path.

    SAHINAKSHA_MODEL_PATH can point to a local mounted model in deployment.
    The repository default is used when the variable is not set.
    """
    configured = os.getenv("SAHINAKSHA_MODEL_PATH")
    return Path(configured) if configured else DEFAULT_MODEL_PATH

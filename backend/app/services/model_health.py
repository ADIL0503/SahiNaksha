from __future__ import annotations

from typing import Any

from .model_config import get_model_path


def model_status() -> dict[str, Any]:
    """Return deployment-safe status information for the configured model."""
    path = get_model_path()
    return {
        "configured": True,
        "model_path": str(path),
        "available": path.is_file(),
    }

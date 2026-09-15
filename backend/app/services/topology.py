from typing import Any


def validate_features(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Run lightweight GeoJSON structure checks before GIS export.

    Full geometric validity checks will be added with Shapely in the GIS stage.
    """
    errors: list[str] = []

    for index, feature in enumerate(features):
        if feature.get("type") != "Feature":
            errors.append(f"Feature {index}: type must be 'Feature'")
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            errors.append(f"Feature {index}: missing geometry")
            continue
        if geometry.get("type") != "Polygon":
            errors.append(f"Feature {index}: expected Polygon geometry")
        if not geometry.get("coordinates"):
            errors.append(f"Feature {index}: empty coordinates")

    return {
        "valid": not errors,
        "feature_count": len(features),
        "errors": errors,
    }

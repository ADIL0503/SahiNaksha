from typing import Any


def polygon_feature(coordinates: list[list[list[float]]], properties: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a GeoJSON Polygon feature from coordinate rings."""
    if not coordinates or not coordinates[0]:
        raise ValueError("Polygon coordinates cannot be empty")

    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates,
        },
    }


def feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": features,
    }

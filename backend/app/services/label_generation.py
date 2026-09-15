from __future__ import annotations

from typing import Any, Iterable

from .georeference import polygon_map_to_pixels


def _clip_normalized(value: float) -> float:
    return min(1.0, max(0.0, value))


def map_polygons_to_yolo_segments(
    polygons: Iterable[list[list[float]]],
    image_width: int,
    image_height: int,
    transform: Iterable[float],
    class_id: int = 0,
) -> list[str]:
    """Convert map-coordinate polygon rings into YOLO segmentation label lines.

    Each output line follows YOLO segmentation format:
    ``class_id x1 y1 x2 y2 ...`` with coordinates normalized to [0, 1].
    Polygon coordinates are expected in the same CRS as the raster transform.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if class_id < 0:
        raise ValueError("class_id must be non-negative")

    lines: list[str] = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue

        pixel_polygon = polygon_map_to_pixels(polygon, transform)
        normalized: list[float] = []
        for x, y in pixel_polygon:
            normalized.extend(
                [
                    _clip_normalized(x / image_width),
                    _clip_normalized(y / image_height),
                ]
            )

        if len(normalized) < 6:
            continue

        values = " ".join(f"{value:.6f}" for value in normalized)
        lines.append(f"{class_id} {values}")

    return lines


def feature_collection_to_yolo_segments(
    feature_collection: dict[str, Any],
    image_width: int,
    image_height: int,
    transform: Iterable[float],
    class_id: int = 0,
) -> list[str]:
    """Convert Polygon features in a GeoJSON FeatureCollection to YOLO labels."""
    features = feature_collection.get("features", [])
    polygons: list[list[list[float]]] = []

    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Polygon":
            continue

        coordinates = geometry.get("coordinates") or []
        if not coordinates:
            continue

        # YOLO's single-segment representation uses the exterior ring.
        exterior = coordinates[0]
        if isinstance(exterior, list) and len(exterior) >= 3:
            polygons.append(exterior)

    return map_polygons_to_yolo_segments(
        polygons,
        image_width,
        image_height,
        transform,
        class_id=class_id,
    )

from __future__ import annotations

from typing import Any


def mask_to_polygon_features(
    masks: list[list[list[float]]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Convert pixel-coordinate segmentation polygons into GeoJSON features.

    The returned coordinates are still in image pixels. Geographic
    georeferencing is intentionally a separate step because it depends on the
    source raster CRS and affine transform.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    features: list[dict[str, Any]] = []
    for index, polygon in enumerate(masks):
        if len(polygon) < 3:
            continue

        ring = [[float(x), float(y)] for x, y in polygon]
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        features.append(
            {
                "type": "Feature",
                "properties": {"id": index, "class": "building"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    return features

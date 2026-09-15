from __future__ import annotations

from typing import Iterable


def pixel_to_map(
    x: float,
    y: float,
    transform: Iterable[float],
) -> tuple[float, float]:
    """Convert pixel coordinates to map coordinates using an affine transform.

    Transform order follows GDAL/rasterio's six affine coefficients:
    (a, b, c, d, e, f), where X = a*x + b*y + c and
    Y = d*x + e*y + f.
    """
    a, b, c, d, e, f = transform
    return a * x + b * y + c, d * x + e * y + f


def polygon_pixels_to_map(
    polygon: list[list[float]],
    transform: Iterable[float],
) -> list[list[float]]:
    return [list(pixel_to_map(x, y, transform)) for x, y in polygon]

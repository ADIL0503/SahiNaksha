from __future__ import annotations

from typing import Iterable


def pixel_to_map(
    x: float,
    y: float,
    transform: Iterable[float],
) -> tuple[float, float]:
    """Convert pixel coordinates to map coordinates using a raster affine transform.

    Transform order follows GDAL/rasterio's six affine coefficients:
    (a, b, c, d, e, f), where X = a*x + b*y + c and
    Y = d*x + e*y + f.
    """
    a, b, c, d, e, f = transform
    return a * x + b * y + c, d * x + e * y + f


def map_to_pixel(
    x: float,
    y: float,
    transform: Iterable[float],
) -> tuple[float, float]:
    """Convert map coordinates to fractional pixel coordinates.

    This uses the inverse of the raster affine transform and therefore works
    with rotated/skewed rasters as well as north-up rasters, provided the
    transform is invertible.
    """
    a, b, c, d, e, f = transform
    determinant = a * e - b * d
    if determinant == 0:
        raise ValueError("Raster affine transform is not invertible")

    dx = x - c
    dy = y - f
    return (
        (e * dx - b * dy) / determinant,
        (-d * dx + a * dy) / determinant,
    )


def polygon_pixels_to_map(
    polygon: list[list[float]],
    transform: Iterable[float],
) -> list[list[float]]:
    """Convert a polygon ring from pixel coordinates to map coordinates."""
    return [list(pixel_to_map(x, y, transform)) for x, y in polygon]


def polygon_map_to_pixels(
    polygon: list[list[float]],
    transform: Iterable[float],
) -> list[list[float]]:
    """Convert a polygon ring from map coordinates to pixel coordinates."""
    return [list(map_to_pixel(x, y, transform)) for x, y in polygon]

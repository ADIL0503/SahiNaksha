from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box, shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create YOLO segmentation labels from GeoJSONL building footprints and a georeferenced raster."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input georeferenced raster (GeoTIFF recommended).")
    parser.add_argument("--buildings", required=True, type=Path, help="GeoJSONL or GeoJSONL.GZ building footprints.")
    parser.add_argument("--output", required=True, type=Path, help="YOLO dataset output directory.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--class-id", default=0, type=int)
    parser.add_argument("--min-area-pixels", default=4.0, type=float)
    return parser.parse_args()


def iter_features(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            if item.get("type") == "Feature":
                yield item


def polygon_to_yolo_line(geometry: Any, dataset_bounds, transform, width: int, height: int, class_id: int, min_area_pixels: float) -> str | None:
    geom = shape(geometry)
    if geom.is_empty:
        return None

    clipped = geom.intersection(dataset_bounds)
    if clipped.is_empty:
        return None

    # A footprint may become a MultiPolygon after clipping. Keep each exterior
    # component as its own YOLO segment line.
    parts = list(clipped.geoms) if clipped.geom_type == "MultiPolygon" else [clipped]
    lines: list[str] = []

    for part in parts:
        if part.geom_type != "Polygon" or part.area <= 0:
            continue
        pixel_area = abs(part.area / abs(transform.a * transform.e - transform.b * transform.d))
        if pixel_area < min_area_pixels:
            continue

        coords: list[tuple[float, float]] = []
        for x_map, y_map in part.exterior.coords:
            x_pixel, y_pixel = (~transform) * (x_map, y_map)
            coords.append((x_pixel, y_pixel))

        normalized: list[float] = []
        for x_pixel, y_pixel in coords:
            x_norm = min(1.0, max(0.0, x_pixel / width))
            y_norm = min(1.0, max(0.0, y_pixel / height))
            normalized.extend([x_norm, y_norm])

        if len(normalized) >= 6:
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized))

    return "\n".join(lines) if lines else None


def main() -> None:
    args = parse_args()
    if args.class_id < 0:
        raise ValueError("--class-id must be non-negative")
    if args.min_area_pixels < 0:
        raise ValueError("--min-area-pixels must be non-negative")

    image_dir = args.output / "images" / args.split
    label_dir = args.output / "labels" / args.split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.image) as raster:
        if raster.crs is None:
            raise ValueError("Input raster has no CRS; use a georeferenced raster.")

        width, height = raster.width, raster.height
        raster_bounds = box(*raster.bounds)
        transform = raster.transform

        label_lines: list[str] = []
        for feature in iter_features(args.buildings):
            geometry = feature.get("geometry")
            if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
                continue
            try:
                line = polygon_to_yolo_line(
                    geometry,
                    raster_bounds,
                    transform,
                    width,
                    height,
                    args.class_id,
                    args.min_area_pixels,
                )
            except (TypeError, ValueError):
                continue
            if line:
                label_lines.extend(line.splitlines())

    image_destination = image_dir / args.image.name
    shutil.copy2(args.image, image_destination)

    label_destination = label_dir / f"{args.image.stem}.txt"
    label_destination.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    print(f"Image: {image_destination}")
    print(f"Labels: {label_destination}")
    print(f"Building segments: {len(label_lines)}")


if __name__ == "__main__":
    main()

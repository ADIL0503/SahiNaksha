from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Iterator

import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import box, shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create YOLO segmentation labels from GeoJSONL building footprints and a georeferenced raster."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input georeferenced raster (GeoTIFF recommended).")
    parser.add_argument("--buildings", required=True, type=Path, help="GeoJSONL or GeoJSONL.GZ building footprints.")
    parser.add_argument("--buildings-crs", default="EPSG:4326", help="CRS of building coordinates; defaults to EPSG:4326 for GlobalML footprints.")
    parser.add_argument("--output", required=True, type=Path, help="YOLO dataset output directory.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--class-id", default=0, type=int)
    parser.add_argument("--min-area-pixels", default=4.0, type=float)
    return parser.parse_args()


def iter_features(path: Path) -> Iterator[dict[str, Any]]:
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


def polygon_to_yolo_lines(
    geometry: Any,
    dataset_bounds,
    transform,
    width: int,
    height: int,
    class_id: int,
    min_area_pixels: float,
) -> list[str]:
    geom = shape(geometry)
    if geom.is_empty:
        return []

    clipped = geom.intersection(dataset_bounds)
    if clipped.is_empty:
        return []

    parts = list(clipped.geoms) if clipped.geom_type == "MultiPolygon" else [clipped]
    lines: list[str] = []
    pixel_area_per_map_unit = abs(transform.a * transform.e - transform.b * transform.d)
    if pixel_area_per_map_unit == 0:
        raise ValueError("Raster transform has zero pixel area")

    for part in parts:
        if part.geom_type != "Polygon" or part.area <= 0:
            continue

        pixel_area = part.area / pixel_area_per_map_unit
        if pixel_area < min_area_pixels:
            continue

        coords = list(part.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords.pop()

        normalized: list[float] = []
        for x_map, y_map in coords:
            x_pixel, y_pixel = (~transform) * (x_map, y_map)
            normalized.extend([
                min(1.0, max(0.0, x_pixel / width)),
                min(1.0, max(0.0, y_pixel / height)),
            ])

        if len(normalized) >= 6:
            lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized))

    return lines


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
        raster_crs = raster.crs
        transform = raster.transform
        label_lines: list[str] = []

        for feature in iter_features(args.buildings):
            geometry = feature.get("geometry")
            if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
                continue
            try:
                geometry_in_raster_crs = transform_geom(
                    args.buildings_crs,
                    raster_crs,
                    geometry,
                    precision=-1,
                )
                label_lines.extend(
                    polygon_to_yolo_lines(
                        geometry_in_raster_crs,
                        raster_bounds,
                        transform,
                        width,
                        height,
                        args.class_id,
                        args.min_area_pixels,
                    )
                )
            except (TypeError, ValueError):
                continue

    image_destination = image_dir / args.image.name
    shutil.copy2(args.image, image_destination)

    label_destination = label_dir / f"{args.image.stem}.txt"
    label_destination.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )

    print(f"Image: {image_destination}")
    print(f"Labels: {label_destination}")
    print(f"Building segments: {len(label_lines)}")


if __name__ == "__main__":
    main()

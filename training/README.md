# SahiNaksha YOLO training

This directory contains the building-footprint segmentation training pipeline.

## Install

```bash
python -m pip install -r training/requirements-yolo.txt
```

## Prepare the dataset

Use a georeferenced raster (GeoTIFF recommended) and building footprints in GeoJSONL or GeoJSONL.GZ format.

```bash
python training/prepare_yolo_dataset.py --image /path/to/image.tif --buildings /path/to/buildings.geojsonl.gz --output training/dataset --split train
```

Create validation data from a separate image/scene:

```bash
python training/prepare_yolo_dataset.py --image /path/to/validation.tif --buildings /path/to/buildings.geojsonl.gz --output training/dataset --split val
```

## Train

```bash
python training/train_yolo.py --data training/sahinaksha.yaml --model yolo11n-seg.pt --epochs 50 --imgsz 640 --batch 8
```

The training script lets Ultralytics select an available accelerator by default. Use `--device cpu` to force CPU training or provide a GPU device such as `0`.

## Dataset layout

```text
training/dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Class `0` is `building`.

## Important limitation

This model detects building footprints for a computer-vision prototype. A building footprint is not automatically a legal cadastral parcel boundary. Final cadastral decisions require authoritative cadastral data and human review.

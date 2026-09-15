from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SahiNaksha building-footprint YOLO segmentation model.")
    parser.add_argument("--data", type=Path, default=Path("training/sahinaksha.yaml"))
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="CUDA device such as 0, or cpu. Defaults to Ultralytics auto-selection.")
    parser.add_argument("--project", type=Path, default=Path("training/runs"))
    parser.add_argument("--name", default="sahinaksha-building-seg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.batch == 0:
        raise ValueError("--batch cannot be zero")

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

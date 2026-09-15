from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a SahiNaksha YOLO segmentation dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("training/dataset"))
    parser.add_argument("--split", choices=["train", "val"], default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    images = {p.stem for p in image_dir.iterdir() if p.is_file()}
    labels = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = sorted(images - labels)
    orphan_labels = sorted(labels - images)
    invalid_lines = 0
    total_segments = 0

    for label_path in label_dir.glob("*.txt"):
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            values = line.split()
            if len(values) < 7 or len(values[1:]) % 2 != 0:
                invalid_lines += 1
                continue
            try:
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
            except ValueError:
                invalid_lines += 1
                continue
            if class_id != 0 or any(value < 0 or value > 1 for value in coordinates):
                invalid_lines += 1
                continue
            total_segments += 1

    print(f"Split: {args.split}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Valid segments: {total_segments}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Orphan labels: {len(orphan_labels)}")
    print(f"Invalid label lines: {invalid_lines}")

    if missing_labels or orphan_labels or invalid_lines:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

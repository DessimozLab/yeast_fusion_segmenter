#!/usr/bin/env python3
"""Render class-coloured YOLO segmentation contours for a prepared dataset."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml


PALETTE = (
    (0, 255, 0),    # f: green
    (255, 255, 0),  # h: cyan
    (0, 165, 255),  # lmcf: orange
    (255, 0, 255),  # lmsgfp: magenta
    (255, 0, 0),    # lsgfp: blue
    (0, 255, 255),  # dip: yellow
    (0, 0, 255),    # d2: red
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create overlays of YOLO segmentation labels for every dataset image."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data", help="Prepared YOLO dataset.yaml")
    input_group.add_argument("--dataset", help="Saved dataset-info JSON")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for train/val/test overlay PNGs and manifest.csv",
    )
    parser.add_argument("--thickness", type=int, default=2, help="Contour thickness in pixels")
    return parser.parse_args()


def resolve_data_yaml(args):
    if args.data:
        return Path(args.data).resolve()
    import json

    info = json.loads(Path(args.dataset).read_text())
    return Path(info["yolo_data"]).resolve()


def resolve_split_dir(config, yaml_path, split):
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    split_value = Path(config[split])
    # Prepared datasets use e.g. train: train/images. A list file is not
    # accepted because this utility intentionally renders every split image.
    if split_value.suffix.lower() in {".txt", ".list"}:
        raise ValueError(f"{split} points to an image list, not a directory: {split_value}")
    return root / split_value


def read_segments(label_path, width, height):
    if not label_path.exists() or not label_path.read_text().strip():
        return []
    segments = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"Malformed segmentation label at {label_path}:{line_number}")
        class_id = int(fields[0])
        coordinates = np.asarray(fields[1:], dtype=np.float32).reshape(-1, 2)
        coordinates[:, 0] *= width
        coordinates[:, 1] *= height
        contour = np.rint(coordinates).astype(np.int32).reshape(-1, 1, 2)
        segments.append((class_id, contour))
    return segments


def draw_overlay(image_path, label_path, output_path, class_names, thickness):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    segments = read_segments(label_path, image.shape[1], image.shape[0])
    for class_id, contour in segments:
        color = PALETTE[class_id % len(PALETTE)]
        cv2.polylines(image, [contour], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    if not segments:
        cv2.putText(image, "no annotation", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image)
    return len(segments)


def main():
    args = parse_args()
    yaml_path = resolve_data_yaml(args)
    config = yaml.safe_load(yaml_path.read_text())
    class_names = {int(key): value for key, value in config["names"].items()}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "manifest.csv").open("w", newline="") as manifest_file:
        manifest = csv.DictWriter(manifest_file, fieldnames=("split", "image", "label", "segments", "overlay"))
        manifest.writeheader()
        for split in ("train", "val", "test"):
            image_dir = resolve_split_dir(config, yaml_path, split)
            split_output = output_dir / split
            split_output.mkdir(exist_ok=True)
            image_paths = sorted(image_dir.glob("*.png"))
            expected_names = {image_path.name for image_path in image_paths}
            for stale_overlay in split_output.glob("*.png"):
                if stale_overlay.name not in expected_names:
                    stale_overlay.unlink()
            for image_path in image_paths:
                label_path = image_dir.parent / "labels" / f"{image_path.stem}.txt"
                output_path = split_output / image_path.name
                segments = draw_overlay(image_path, label_path, output_path, class_names, args.thickness)
                manifest.writerow({
                    "split": split,
                    "image": str(image_path.resolve()),
                    "label": str(label_path.resolve()),
                    "segments": segments,
                    "overlay": str(output_path.resolve()),
                })

    with (output_dir / "legend.txt").open("w") as legend:
        for class_id, class_name in sorted(class_names.items()):
            legend.write(f"{class_id}: {class_name}; BGR={PALETTE[class_id % len(PALETTE)]}\n")


if __name__ == "__main__":
    main()

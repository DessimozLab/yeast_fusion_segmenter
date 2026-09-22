#!/usr/bin/env python3
"""Run prediction and labelled test evaluation for each model/dataset pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODELS = {
    "40x_only": "models/40x_only_yolov8n_seg.pt",
    "all_images": "models/all_images_yolov8n_seg.pt",
    "all_czi": "models/all_czi_yolov8n_seg.pt",
}
DEFAULT_DATASETS = {
    "all_czi": "data/dataset_info/all_czi.json",
    "all_images": "data/dataset_info/all_images.json",
}


def labelled_test_count(data_yaml: Path) -> tuple[int, int]:
    root = data_yaml.parent / "test"
    labels = list((root / "labels").glob("*.txt"))
    return sum(bool(label.read_text().strip()) for label in labels), len(labels)


def metric_value(results, key: str) -> float | None:
    value = results.results_dict.get(key)
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-evaluate trained YOLO models")
    parser.add_argument("--device", default="0")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="evaluation/cross_dataset_metrics.json")
    parser.add_argument("--models", nargs="+", choices=tuple(DEFAULT_MODELS), default=tuple(DEFAULT_MODELS))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DEFAULT_DATASETS), default=tuple(DEFAULT_DATASETS))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_payload: dict[str, object] = {"protocol": {"split": "test", "img_size": args.img_size, "device": args.device}, "results": []}

    for model_name in args.models:
        model_path = DEFAULT_MODELS[model_name]
        model = YOLO(model_path)
        for dataset_name in args.datasets:
            info_path = DEFAULT_DATASETS[dataset_name]
            info = json.loads(Path(info_path).read_text())
            data_yaml = Path(info["yolo_data"])
            labelled, total = labelled_test_count(data_yaml)
            # Save visual annotations separately for every cross pair.
            model.predict(
                source=str(data_yaml.parent / "test" / "images"),
                imgsz=args.img_size,
                device=args.device,
                save=True,
                project="evaluation/predictions",
                name=f"{model_name}_on_{dataset_name}",
                exist_ok=True,
                verbose=False,
            )
            metrics = model.val(
                data=str(data_yaml), split="test", imgsz=args.img_size,
                batch=args.batch_size, device=args.device, workers=2,
                project="evaluation/metrics", name=f"{model_name}_on_{dataset_name}",
                exist_ok=True, verbose=False,
            )
            results_payload["results"].append({
                "model": model_name,
                "model_path": str(Path(model_path).resolve()),
                "dataset": dataset_name,
                "dataset_info": str(Path(info_path).resolve()),
                "labelled_test_images": labelled,
                "total_test_images": total,
                "box_map50": metric_value(metrics, "metrics/mAP50(B)"),
                "box_map50_95": metric_value(metrics, "metrics/mAP50-95(B)"),
                "mask_map50": metric_value(metrics, "metrics/mAP50(M)"),
                "mask_map50_95": metric_value(metrics, "metrics/mAP50-95(M)"),
                "precision": metric_value(metrics, "metrics/precision(B)"),
                "recall": metric_value(metrics, "metrics/recall(B)"),
            })
    output.write_text(json.dumps(results_payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

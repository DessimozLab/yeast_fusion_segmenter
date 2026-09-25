#!/usr/bin/env python3
"""Create visual CZI-mask orientation candidates and record manual approvals."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image_dataset import MicroscopyImageDataset
from prepare_yolo_data import (
    ORIENTATION_TRANSFORMS,
    apply_mask_orientation,
    center_crop_or_pad,
    load_annotation_mask,
    split_mask,
)


PALETTE = ((0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255),
           (255, 0, 0), (0, 255, 255), (0, 0, 255))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--samples', required=True,
                        help='Comma-separated <magnification>/<sample-id> keys')
    parser.add_argument('--output-dir', default='validation/manual_orientation_review')
    parser.add_argument('--overrides', default='data/manual_orientation_overrides.json')
    parser.add_argument('--set', dest='selections', action='append', default=[],
                        help='Approve KEY=TRANSFORM, e.g. 40x/p1-1g7-08=flip_ud')
    return parser.parse_args()


def load_overrides(path):
    source = Path(path)
    if not source.exists():
        return {'schema_version': 1, 'mask_transforms': {}}
    payload = json.loads(source.read_text())
    if payload.get('schema_version') != 1 or not isinstance(payload.get('mask_transforms'), dict):
        raise ValueError(f'Invalid override file: {source}')
    return payload


def write_selections(path, selections):
    payload = load_overrides(path)
    for selection in selections:
        if '=' not in selection:
            raise ValueError(f'Expected KEY=TRANSFORM, got {selection}')
        key, transform = selection.split('=', 1)
        if transform not in ORIENTATION_TRANSFORMS or '/' not in key:
            raise ValueError(f'Invalid orientation selection: {selection}')
        payload['mask_transforms'][key] = transform
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + '\n')


def draw_candidate(rgb, mask, transform):
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for class_id, class_mask in enumerate(split_mask(mask, crop=mask.shape[0])):
        contours, _ = cv2.findContours((class_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, PALETTE[class_id], 2, cv2.LINE_AA)
    cv2.putText(canvas, transform, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    if args.selections:
        write_selections(args.overrides, args.selections)

    requested = [item.strip() for item in args.samples.split(',') if item.strip()]
    dataset = MicroscopyImageDataset(args.raw_root)
    records = {f'{record.magnification}/{record.sample_id}': record for record in dataset.records}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in requested:
        record = records.get(key)
        if record is None or record.annotation_path is None:
            raise ValueError(f'No annotated raw record for {key}')
        png_path = dataset.png_root / record.magnification / f'{record.sample_id}.png'
        if not png_path.exists():
            dataset.materialize_pngs()
        with Image.open(png_path) as image:
            rgb = center_crop_or_pad(np.asarray(image.convert('RGB')), size=1024)
        mask = center_crop_or_pad(load_annotation_mask(record.annotation_path, record.sample_id), size=1024)
        candidates = [draw_candidate(rgb, apply_mask_orientation(mask, transform), transform)
                      for transform in ORIENTATION_TRANSFORMS]
        panel = np.vstack((np.hstack(candidates[:2]), np.hstack(candidates[2:])))
        output = output_dir / f'{record.magnification}_{record.sample_id}_candidates.png'
        cv2.imwrite(str(output), panel)
        print(output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Export manually orientation-corrected HDF5 annotation copies.

The source HDF5 files are copied byte-for-byte first; only the values of image
datasets in the copy are transformed. Group layout, dataset names, dtypes,
compression, chunking, and attributes therefore retain the input protocol.
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from image_dataset import MicroscopyImageDataset
from prepare_yolo_data import ORIENTATION_TRANSFORMS, apply_mask_orientation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--overrides', required=True)
    parser.add_argument('--output-root', required=True,
                        help='Derived root containing <magnification>/annotations copies')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def transform_dataset_values(values, transform):
    """Apply a spatial transform to each 2-D image in an HDF5 dataset."""
    values = np.asarray(values)
    if values.ndim < 2:
        return values
    if transform == 'orig':
        return values
    if transform == 'flip_ud':
        return np.flip(values, axis=-2)
    if transform == 'flip_lr':
        return np.flip(values, axis=-1)
    if transform == 'flip_udlr':
        return np.flip(np.flip(values, axis=-2), axis=-1)
    raise ValueError(f'Unknown orientation transform: {transform}')


def export_mask(source, destination, transform, overwrite=False):
    if destination.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite derived mask: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with h5py.File(destination, 'r+') as handle:
        datasets = []
        handle.visititems(lambda name, item: datasets.append(item) if isinstance(item, h5py.Dataset) else None)
        for dataset in datasets:
            if dataset.ndim >= 2:
                dataset[...] = transform_dataset_values(dataset[...], transform)


def main():
    args = parse_args()
    payload = json.loads(Path(args.overrides).read_text())
    if payload.get('schema_version') != 1 or not isinstance(payload.get('mask_transforms'), dict):
        raise ValueError('Invalid orientation override JSON')
    overrides = payload['mask_transforms']
    if any(value not in ORIENTATION_TRANSFORMS for value in overrides.values()):
        raise ValueError('Override JSON contains an invalid transform')

    dataset = MicroscopyImageDataset(args.raw_root)
    records = {f'{record.magnification}/{record.sample_id}': record for record in dataset.annotated_records()}
    exported = []
    for key, transform in sorted(overrides.items()):
        record = records.get(key)
        if record is None:
            raise ValueError(f'No annotated raw record for override: {key}')
        destination = Path(args.output_root) / record.magnification / 'annotations' / record.annotation_path.name
        export_mask(record.annotation_path, destination, transform, args.overwrite)
        exported.append({'key': key, 'transform': transform, 'source': str(record.annotation_path), 'output': str(destination)})
    manifest = Path(args.output_root) / 'manual_orientation_hdf5_manifest.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'schema_version': 1, 'exports': exported}, indent=2) + '\n')
    print(manifest)


if __name__ == '__main__':
    main()

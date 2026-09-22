"""Regression tests for the notebook's seven-class HDF5 annotation encoding."""

from pathlib import Path
import tempfile

import numpy as np

import h5py

from prepare_yolo_data import (
    CLASS_NAMES,
    align_czi_mask_orientation,
    center_crop_or_pad,
    load_annotation_mask,
    mask_to_contour_file,
    split_mask,
)


def test_notebook_bins_preserve_all_seven_phenotypes():
    mask = np.zeros((14, 14), dtype=np.uint16)
    for class_id in range(len(CLASS_NAMES)):
        y = class_id * 2
        mask[y:y + 2, 0:2] = class_id * 1000 + 1

    class_masks = split_mask(mask, crop=14)

    assert len(class_masks) == 7
    for class_id, class_mask in enumerate(class_masks):
        expected = class_id * 1000 + 1
        assert expected in np.unique(class_mask)
        assert all(value in (0, expected) for value in np.unique(class_mask))


def test_yolo_contours_include_every_non_background_class():
    mask = np.zeros((28, 28), dtype=np.uint16)
    for class_id in range(len(CLASS_NAMES)):
        y = class_id * 4
        mask[y:y + 3, 0:3] = class_id * 1000 + 1

    with tempfile.TemporaryDirectory() as directory:
        label_path = Path(directory) / 'labels.txt'
        mask_to_contour_file(mask, label_path)
        labels = {int(line.split()[0]) for line in label_path.read_text().splitlines()}

    assert labels == set(range(len(CLASS_NAMES)))


def test_center_crop_matches_notebook_geometry():
    image = np.arange(36, dtype=np.uint16).reshape(6, 6)

    cropped = center_crop_or_pad(image, size=4)

    assert np.array_equal(cropped, image[1:5, 1:5])


def test_materialized_tiff_frame_uses_matching_hdf5_frame():
    with tempfile.TemporaryDirectory() as directory:
        h5_path = Path(directory) / 'sample__mask.h5'
        with h5py.File(h5_path, 'w') as h5:
            group = h5.create_group('FOV0')
            group.create_dataset('T0', data=np.full((2, 2), 1001, dtype=np.uint16))
            group.create_dataset('T1', data=np.full((2, 2), 3001, dtype=np.uint16))

        mask = load_annotation_mask(h5_path, 'sample__f0001')

    assert np.array_equal(mask, np.full((2, 2), 3001, dtype=np.uint16))


def test_czi_orientation_check_applies_clear_vertical_mask_flip():
    mask = np.zeros((100, 100), dtype=np.uint16)
    mask[10:30, 20:45] = 1001
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    # The image boundary is where the vertically flipped mask belongs.
    rgb[70:90, 20:45, 1] = 255

    aligned, orientation, scores = align_czi_mask_orientation(mask, rgb)

    assert orientation == 'flip_ud'
    assert np.array_equal(aligned, np.flipud(mask))
    assert scores['flip_ud'][0] > scores['orig'][0] + 0.10

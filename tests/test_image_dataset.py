"""Tests for deterministic raw microscopy discovery and PNG conversion."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image

from image_dataset import ImageRecord, MicroscopyImageDataset, requires_vertical_flip


class TestMicroscopyImageDataset(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "raw"
        tiff = self.root / "other" / "images" / "tiff"
        annotations = self.root / "other" / "annotations"
        tiff.mkdir(parents=True)
        annotations.mkdir(parents=True)
        Image.fromarray(np.full((8, 8), 20, dtype=np.uint16)).save(tiff / "fusion-a-001__bf.tiff")
        Image.fromarray(np.full((8, 8), 50, dtype=np.uint16)).save(tiff / "fusion-a-001__gfp.tiff")
        Image.fromarray(np.full((8, 8), 100, dtype=np.uint16)).save(tiff / "fusion-a-001__rfp.tiff")
        with h5py.File(annotations / "fusion-a-001__mask.h5", "w") as handle:
            handle.create_dataset("mask", data=np.zeros((8, 8), dtype=np.uint16))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_pairs_annotation_and_materializes_rgb_png(self):
        dataset = MicroscopyImageDataset(self.root)
        self.assertEqual(len(dataset.records), 1)
        record = dataset.records[0]
        self.assertEqual(record.annotation_path.name, "fusion-a-001__mask.h5")
        self.assertEqual(record.metadata()["annotation_path"], str(record.annotation_path))

        converted = dataset.materialize_pngs()
        self.assertEqual(len(converted), 1)
        with Image.open(converted[0].sources["png"]) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (8, 8))

    def test_unrecognised_names_fail_fast(self):
        bad = self.root / "other" / "images" / "tiff" / "Fusion A BF.tif"
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(bad)
        with self.assertRaisesRegex(ValueError, "Invalid TIFF filename"):
            MicroscopyImageDataset(self.root)

    def test_tiff_without_brightfield_fails(self):
        (self.root / "other" / "images" / "tiff" / "fusion-a-001__bf.tiff").unlink()
        with self.assertRaisesRegex(ValueError, "BF channel"):
            MicroscopyImageDataset(self.root)

    def test_saved_dataset_info_round_trips(self):
        dataset = MicroscopyImageDataset(self.root, magnifications=("other",))
        info_path = dataset.save_dataset_info(
            Path(self.tempdir.name) / "dataset-info.json",
            name="unit-test",
            yolo_data=Path(self.tempdir.name) / "yolo" / "dataset.yaml",
        )
        info = MicroscopyImageDataset.load_dataset_info(info_path)
        self.assertEqual(info["name"], "unit-test")
        self.assertEqual(info["magnifications"], ["other"])
        self.assertEqual(info["source_formats"], ["czi", "tiff"])
        self.assertEqual(len(info["records"]), 1)

    def test_all_annotations_are_paired(self):
        dataset = MicroscopyImageDataset(self.root)
        self.assertEqual(dataset.annotated_records(), dataset.records)

    def test_known_40x_orientation_correction_has_one_exception(self):
        corrected = ImageRecord("p1-1e3-13", "40x", "czi", {"czi": Path("example.czi")})
        exception = ImageRecord("p1-1g2-09", "40x", "czi", {"czi": Path("exception.czi")})
        other = ImageRecord("sample", "other", "czi", {"czi": Path("other.czi")})

        self.assertTrue(requires_vertical_flip(corrected))
        self.assertFalse(requires_vertical_flip(exception))
        self.assertFalse(requires_vertical_flip(other))
        self.assertTrue(corrected.metadata()["vertical_flip_for_conversion"])

    def test_corrupt_cached_png_is_rematerialized(self):
        dataset = MicroscopyImageDataset(self.root)
        cached = dataset.png_root / "other" / "fusion-a-001.png"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"not a PNG")

        converted = dataset.materialize_pngs()

        with Image.open(converted[0].sources["png"]) as image:
            image.verify()

    def test_orphan_annotation_fails_fast(self):
        annotations = self.root / "other" / "annotations"
        with h5py.File(annotations / "orphan__mask.h5", "w") as handle:
            handle.create_dataset("mask", data=np.zeros((8, 8), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "Annotation has no matching image"):
            MicroscopyImageDataset(self.root)


if __name__ == "__main__":
    unittest.main()

"""Canonical microscopy-image discovery and PNG conversion.

The loader deliberately accepts only the filenames described in
``DATASET_NAMING.md``.  This makes pairing channels and masks deterministic:
we never pair files merely because their sorted positions happen to agree.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterator, Literal

import numpy as np
from PIL import Image, ImageSequence


MAGNIFICATIONS = ("40x", "other")
CHANNELS = ("bf", "gfp", "rfp")
# Fiji presents the known 40x CZI series inverted vertically.  The exception
# was acquired with the corrected orientation already applied at the scope.
_VERTICAL_FLIP_40X_EXCEPTION = "p1-1g2-09"
_CZI_RE = re.compile(r"^(?P<id>[a-z0-9][a-z0-9-]*)\.czi$", re.IGNORECASE)
_TIFF_RE = re.compile(
    r"^(?P<id>[a-z0-9][a-z0-9-]*)__?(?P<channel>bf|gfp|rfp)\.tiff?$",
    re.IGNORECASE,
)
_MASK_RE = re.compile(r"^(?P<id>[a-z0-9][a-z0-9-]*)__mask\.h(?:df)?5$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    """One acquisition and the optional HDF5 annotation paired by sample id."""

    sample_id: str
    magnification: Literal["40x", "other"]
    source_format: Literal["czi", "tiff"]
    sources: dict[str, Path]
    annotation_path: Path | None = None

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable metadata, including the HDF5 path when present."""
        return {
            "sample_id": self.sample_id,
            "magnification": self.magnification,
            "source_format": self.source_format,
            "sources": {key: str(value) for key, value in self.sources.items()},
            "annotation_path": str(self.annotation_path) if self.annotation_path else None,
            "vertical_flip_for_conversion": requires_vertical_flip(self),
        }


def requires_vertical_flip(record: ImageRecord) -> bool:
    """Return whether this raw acquisition needs the known 40x Fiji correction."""
    return (
        record.magnification == "40x"
        and record.sample_id != _VERTICAL_FLIP_40X_EXCEPTION
    )


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def _as_rgb(array: np.ndarray) -> np.ndarray:
    """Convert a 2-D channel or image with channels to an RGB uint8 array."""
    array = np.asarray(array)
    if array.ndim == 2:
        channel = _to_uint8(array)
        return np.repeat(channel[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, received {array.shape}")
    if array.shape[-1] not in (1, 2, 3, 4):
        candidates = [axis for axis, length in enumerate(array.shape) if length in (1, 2, 3, 4)]
        if not candidates:
            raise ValueError(f"Cannot identify a channel axis in {array.shape}")
        array = np.moveaxis(array, candidates[0], -1)
    array = array[..., :3]
    if array.shape[-1] == 1:
        return np.repeat(_to_uint8(array), 3, axis=2)
    if array.shape[-1] == 2:
        array = np.concatenate((array, np.zeros((*array.shape[:2], 1), dtype=array.dtype)), axis=2)
    return np.stack([_to_uint8(array[..., index]) for index in range(3)], axis=2)


def _is_valid_png(path: Path) -> bool:
    """Return whether a cached materialized PNG can be decoded completely."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


class MicroscopyImageDataset:
    """Discover canonical raw data and materialize model-ready PNG files.

    ``root`` must contain ``<magnification>/images/{czi,tiff}`` and optional
    ``<magnification>/annotations`` folders.  PNGs are generated outside the
    raw tree by default, so conversion never overwrites the scientific source.
    """

    def __init__(
        self,
        root: str | Path,
        png_root: str | Path | None = None,
        magnifications: tuple[str, ...] = MAGNIFICATIONS,
        source_formats: tuple[str, ...] = ("czi", "tiff"),
    ):
        self.root = Path(root).resolve()
        self.png_root = Path(png_root).resolve() if png_root else self.root.parent / "derived" / "png"
        invalid = set(magnifications).difference(MAGNIFICATIONS)
        if invalid:
            raise ValueError(f"Unknown magnification selection: {sorted(invalid)}")
        self.magnifications = tuple(magnifications)
        invalid_formats = set(source_formats).difference({"czi", "tiff"})
        if invalid_formats:
            raise ValueError(f"Unknown source format selection: {sorted(invalid_formats)}")
        self.source_formats = tuple(source_formats)
        self.records = self._discover()

    def __iter__(self) -> Iterator[ImageRecord]:
        return iter(self.records)

    def annotated_records(self) -> list[ImageRecord]:
        """Return only records with a deterministically paired HDF5 annotation."""
        return [record for record in self.records if record.annotation_path is not None]

    def _discover(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        for magnification in self.magnifications:
            base = self.root / magnification
            masks = self._masks(base / "annotations")
            if "czi" in self.source_formats:
                records.extend(self._czi_records(base, magnification, masks))
            if "tiff" in self.source_formats:
                records.extend(self._tiff_records(base, magnification, masks))
        identities = [(record.magnification, record.sample_id) for record in records]
        if len(identities) != len(set(identities)):
            raise ValueError("A sample id may appear only once per magnification")
        # A full dataset must account for every annotation.  A format-filtered
        # view deliberately omits annotations belonging to the other format.
        if set(self.source_formats) == {"czi", "tiff"}:
            paired_annotations = {
                (record.magnification, record.sample_id)
                for record in records
                if record.annotation_path is not None
            }
            available_annotations = {
                (magnification, sample_id)
                for magnification in self.magnifications
                for sample_id in self._masks(self.root / magnification / "annotations")
            }
            orphaned = available_annotations.difference(paired_annotations)
            if orphaned:
                display = ", ".join(f"{mag}/{sample}" for mag, sample in sorted(orphaned))
                raise ValueError(f"Annotation has no matching image record: {display}")
        return sorted(records, key=lambda record: (record.magnification, record.sample_id, record.source_format))

    @staticmethod
    def _masks(directory: Path) -> dict[str, Path]:
        if not directory.exists():
            return {}
        masks: dict[str, Path] = {}
        for path in directory.iterdir():
            match = _MASK_RE.match(path.name)
            if match:
                sample_id = match["id"].lower()
                if sample_id in masks:
                    raise ValueError(f"Duplicate annotation for {sample_id}: {path} and {masks[sample_id]}")
                masks[sample_id] = path
        return masks

    def _czi_records(self, base: Path, magnification: str, masks: dict[str, Path]) -> list[ImageRecord]:
        directory = base / "images" / "czi"
        if not directory.exists():
            return []
        records = []
        for path in directory.iterdir():
            match = _CZI_RE.match(path.name)
            if not match:
                if path.suffix.lower() == ".czi":
                    raise ValueError(f"Invalid CZI filename: {path.name}")
                continue
            sample_id = match["id"].lower()
            records.append(ImageRecord(sample_id, magnification, "czi", {"czi": path}, masks.get(sample_id)))
        return records

    def _tiff_records(self, base: Path, magnification: str, masks: dict[str, Path]) -> list[ImageRecord]:
        directory = base / "images" / "tiff"
        if not directory.exists():
            return []
        grouped: dict[str, dict[str, Path]] = {}
        for path in directory.iterdir():
            match = _TIFF_RE.match(path.name)
            if not match:
                if path.suffix.lower() in {".tif", ".tiff"}:
                    raise ValueError(f"Invalid TIFF filename: {path.name}")
                continue
            sample_id, channel = match["id"].lower(), match["channel"].lower()
            if channel in grouped.setdefault(sample_id, {}):
                raise ValueError(f"Duplicate {channel} TIFF for {sample_id}")
            grouped[sample_id][channel] = path
        missing_bf = [sample_id for sample_id, channels in grouped.items() if "bf" not in channels]
        if missing_bf:
            raise ValueError(f"TIFF samples need a BF channel: {', '.join(sorted(missing_bf))}")
        return [
            ImageRecord(sample_id, magnification, "tiff", channels, masks.get(sample_id))
            for sample_id, channels in grouped.items()
        ]

    def write_manifest(self, path: str | Path | None = None) -> Path:
        """Write discovered records, with their HDF5 paths, to a JSON manifest."""
        output = Path(path) if path else self.root / "manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps([record.metadata() for record in self.records], indent=2) + "\n")
        return output

    def save_dataset_info(
        self,
        path: str | Path,
        *,
        name: str,
        yolo_data: str | Path | None = None,
        split: dict[str, object] | None = None,
        annotation_classes: dict[int, str] | None = None,
    ) -> Path:
        """Persist a selected dataset definition for reproducible training.

        The JSON includes exact source/annotation metadata and, once prepared,
        the YOLO YAML path.  ``train_yolo.py --dataset <this file>`` resolves
        that path without requiring the caller to rediscover the raw files.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data_path = Path(yolo_data).resolve() if yolo_data else None
        payload = {
            "schema_version": 1,
            "name": name,
            "raw_root": str(self.root),
            "png_root": str(self.png_root),
            "magnifications": list(self.magnifications),
            "source_formats": list(self.source_formats),
            "yolo_data": str(data_path) if data_path else None,
            "split": split or {},
            "annotation_classes": annotation_classes or {},
            "records": [record.metadata() for record in self.records],
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        return output

    @staticmethod
    def load_dataset_info(path: str | Path) -> dict[str, object]:
        """Load and minimally validate a dataset definition saved by this class."""
        source = Path(path)
        payload = json.loads(source.read_text())
        required = {"schema_version", "name", "magnifications", "records", "yolo_data"}
        missing = required.difference(payload)
        if payload.get("schema_version") != 1 or missing:
            raise ValueError(f"Invalid dataset info file {source}: missing {sorted(missing)}")
        return payload

    def materialize_pngs(self, overwrite: bool = False) -> list[ImageRecord]:
        """Convert every acquisition to deterministic PNG(s) and return their records.

        A multi-page TIFF yields one PNG per frame.  A CZI yields one PNG using
        ImageJ/Fiji.  Returned records retain the original annotation path.
        """
        converted: list[ImageRecord] = []
        for record in self.records:
            # CZI has one deterministic output frame.  Skipping an existing
            # PNG avoids reopening ImageJ/Fiji on resumable dataset builds.
            czi_destination = self.png_root / record.magnification / f"{record.sample_id}.png"
            if record.source_format == "czi" and _is_valid_png(czi_destination) and not overwrite:
                converted.append(
                    ImageRecord(record.sample_id, record.magnification, "czi", {"png": czi_destination}, record.annotation_path)
                )
                continue
            if record.source_format == "tiff" and not overwrite:
                # TIFF frame count is available from its BF header, so a
                # resumed build can avoid decoding complete stacks already
                # materialized before an interrupted run.
                with Image.open(record.sources["bf"]) as image:
                    frame_count = getattr(image, "n_frames", 1)
                destinations = [
                    self.png_root / record.magnification /
                    f"{record.sample_id}{'' if frame_count == 1 else f'__f{index:04d}'}.png"
                    for index in range(frame_count)
                ]
                if all(_is_valid_png(destination) for destination in destinations):
                    converted.extend(
                        ImageRecord(
                            record.sample_id + ("" if frame_count == 1 else f"__f{index:04d}"),
                            record.magnification,
                            "tiff",
                            {"png": destination},
                            record.annotation_path,
                        )
                        for index, destination in enumerate(destinations)
                    )
                    continue
            frames = self._load_czi(record) if record.source_format == "czi" else self._load_tiff(record)
            for frame_index, image in enumerate(frames):
                if requires_vertical_flip(record):
                    image = np.flipud(image).copy()
                suffix = "" if len(frames) == 1 else f"__f{frame_index:04d}"
                destination = self.png_root / record.magnification / f"{record.sample_id}{suffix}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not _is_valid_png(destination):
                    Image.fromarray(image).save(destination)
                converted.append(ImageRecord(record.sample_id + suffix, record.magnification, "tiff", {"png": destination}, record.annotation_path))
        return converted

    @staticmethod
    def _load_tiff(record: ImageRecord) -> list[np.ndarray]:
        channel_frames: dict[str, list[np.ndarray]] = {}
        for channel, path in record.sources.items():
            with Image.open(path) as image:
                channel_frames[channel] = [_to_uint8(np.asarray(frame)) for frame in ImageSequence.Iterator(image)]
        count = min(len(frames) for frames in channel_frames.values())
        frames = []
        for index in range(count):
            bf = channel_frames["bf"][index]
            if bf.ndim == 3:
                frames.append(_as_rgb(bf))
                continue
            frames.append(np.stack([channel_frames.get(channel, [bf] * count)[index] for channel in CHANNELS], axis=2))
        return frames

    _imagej = None

    @classmethod
    def _load_czi(cls, record: ImageRecord) -> list[np.ndarray]:
        try:
            import imagej
        except ImportError as error:  # pragma: no cover - requires Fiji runtime
            raise ImportError("CZI conversion requires pyimagej in the yeast_fusion_segmenter environment") from error
        if cls._imagej is None:
            cls._imagej = imagej.init("sc.fiji:fiji", mode="headless")
        dataset = cls._imagej.io().open(str(record.sources["czi"]))
        return [_as_rgb(cls._imagej.py.from_java(dataset))]

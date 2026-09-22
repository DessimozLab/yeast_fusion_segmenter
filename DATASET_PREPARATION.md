# Preparing a microscopy dataset

This guide is the supported route from raw microscopy files to YOLO training or annotation. **The naming and layout rules in [DATASET_NAMING.md](DATASET_NAMING.md) are mandatory. Dataset discovery intentionally fails on malformed TIFF/CZI names rather than risking incorrect image/mask pairing.**

## 1. Activate the project environment

CZI conversion uses Fiji through `pyimagej`, so run preparation and annotation in the project mamba environment:

```bash
mamba activate yeast_fusion_segmenter
```

## 2. Place files in the canonical layout

Put raw source files under `data/raw`. Keep 40× acquisitions separate from all other magnifications:

```text
data/raw/
  40x/images/czi/<sample-id>.czi
  40x/annotations/<sample-id>__mask.h5
  other/images/tiff/<sample-id>__bf.tiff
  other/images/tiff/<sample-id>__gfp.tiff
  other/images/tiff/<sample-id>__rfp.tiff
  other/annotations/<sample-id>__mask.h5
```

TIFF `__gfp` and `__rfp` channels are optional; `__bf` is required. HDF5 masks are optional, but an image without one is inference-only and is placed in the test split by the training-preparation command.

### Phenotype labels

The annotation conversion preserves the seven classes from
`segment_retrain(1).ipynb`: `f`, `h`, `lmcf`, `lmsgfp`, `lsgfp`, `dip`, and
`d2` (IDs 0–6 respectively). Values are read in 1000-wide bins; see
[DATASET_NAMING.md](DATASET_NAMING.md#annotation-classes). **The class order
must be respected.** Older three-class datasets have collapsed lysis
phenotypes and must be rebuilt before training or evaluation.

For raw datasets, the builder also applies the notebook's 1024px center
crop/pad to both the converted RGB image and its HDF5 mask before extracting
contours. This paired operation is required: cropping a CZI image and mask
differently invalidates spatial labels.

For CZI inputs, preparation also runs the notebook's per-file orientation
sanity check. It compares HDF5 mask boundaries against fluorescence-image
edges for the original, vertical-flip, horizontal-flip, and 180° masks, and
uses a flipped mask only when it substantially improves alignment. This is
required because some CZI/HDF5 pairs have an inverted vertical orientation.

## 3. Migrate this repository’s legacy files (one time)

The repository has already been reorganized into `data/raw`: 7 CZI records at 40× and 39 records at other magnifications. For a fresh legacy checkout, review the migration first, then apply it:

```bash
python organize_raw_images.py
python organize_raw_images.py --apply
```

The script moves the authoritative source folders only. It leaves generated datasets and duplicate working copies alone.

## 4. Validate discovery and write metadata

Every `ImageRecord` includes its source path(s), magnification, source format, and `annotation_path`. Write this metadata before a long conversion run:

```bash
python - <<'PY'
from image_dataset import MicroscopyImageDataset

dataset = MicroscopyImageDataset("data/raw")
dataset.write_manifest()
print(f"Validated {len(dataset.records)} acquisitions")
PY
```

This writes `data/raw/manifest.json`. Inspect records with a missing `annotation_path` before expecting them to contribute training labels.

## 5. Convert raw images to PNG and prepare a YOLO dataset

Use the canonical `raw` format. The dataset class normalizes TIFF channels and writes PNG frames to `data/derived/png`; CZI files are opened with ImageJ/Fiji and converted there as well. Raw scientific files are not overwritten.

```bash
python prepare_yolo_data.py \
  --input-dir data/raw \
  --file-format raw \
  --output-dir datasets/experiment_01
```

The command creates YOLO `train`, `val`, and `test` folders and a dataset YAML file. Records with an HDF5 annotation are converted to YOLO segmentation labels; unannotated records receive an empty label and start in the test split.

### Split controls

The builder makes a deterministic split of *annotated* images using
`--random-seed` (default `42`). `--val-split` and `--test-split` are fractions
of annotated images; the remainder is train. They must sum to less than one.
Unannotated records are always inference-only and remain in `test`, regardless
of those fractions. The selected fractions and seed are stored in
`dataset-info.json`.

```bash
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --output-dir data/yolo_datasets/custom --val-split 0.15 --test-split 0.15 \
  --random-seed 7
```

### Build examples

All examples start from the canonical `data/raw` layout. `--file-format raw`
selects `MicroscopyImageDataset`; `--source-format` filters its CZI/TIFF
records, and `--magnification` filters the 40× group.

**All CZI images, 70% train / 15% validation / 15% test**

```bash
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --source-format czi --magnification all \
  --output-dir data/yolo_datasets/czi_70_15_15 \
  --dataset-info data/dataset_info/czi_70_15_15.json \
  --val-split 0.15 --test-split 0.15 --random-seed 42
```

**40× CZI images only, 80% train / 20% validation / 0% held-out annotated test**

```bash
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --source-format czi --magnification 40x \
  --output-dir data/yolo_datasets/czi_40x_80_20 \
  --dataset-info data/dataset_info/czi_40x_80_20.json \
  --val-split 0.20 --test-split 0 --random-seed 42
```

**All TIFF stacks, 80% train / 10% validation / 10% test**

```bash
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --source-format tiff --magnification all \
  --output-dir data/yolo_datasets/tiff_80_10_10 \
  --dataset-info data/dataset_info/tiff_80_10_10.json \
  --val-split 0.10 --test-split 0.10 --random-seed 42
```

**Mixed CZI + TIFF dataset, 60% train / 20% validation / 20% test**

```bash
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --source-format all --magnification all \
  --output-dir data/yolo_datasets/mixed_60_20_20 \
  --dataset-info data/dataset_info/mixed_60_20_20.json \
  --val-split 0.20 --test-split 0.20 --random-seed 7
```

The actual count may differ slightly from the stated percentage because splits
use whole image frames. Check the final printed counts and the saved split
metadata before training.

To make the two maintained training datasets, use:

```bash
python prepare_yolo_data.py --input-dir data/raw --file-format raw --magnification 40x \
  --output-dir data/yolo_datasets/40x_only --dataset-info data/dataset_info/40x_only.json
python prepare_yolo_data.py --input-dir data/raw --file-format raw --magnification all \
  --output-dir data/yolo_datasets/all_images --dataset-info data/dataset_info/all_images.json
python prepare_yolo_data.py --input-dir data/raw --file-format raw --magnification all --source-format czi \
  --output-dir data/yolo_datasets/all_czi --dataset-info data/dataset_info/all_czi.json
```

Train from either persisted dataset definition (rather than manually locating
its YAML) with `python train_yolo.py --dataset data/dataset_info/40x_only.json`
or `python train_yolo.py --dataset data/dataset_info/all_images.json`.

The prepared repository datasets currently contain:

| Dataset definition | PNG frames | Train / validation / test |
| --- | ---: | ---: |
| `data/dataset_info/40x_only.json` | 7 | 5 / 1 / 1 |
| `data/dataset_info/all_images.json` | 112 | 92 / 3 / 17 |
| `data/dataset_info/all_czi.json` | 21 | 5 / 1 / 15 |

The 40× and CZI-only datasets share the same five annotated 40× training
images. They contain no `dip` (class 5) examples. `all_images` includes TIFF
and CZI data and does contain `dip` training labels, so it is the maintained
choice when all seven notebook phenotype classes are required.

### Dataset-size and augmentation limitation

There is no universal image count for cellular instance segmentation: the
required number depends on phenotype rarity, cell density, imaging modality,
and how much biological/acquisition variation the model must generalize over.
For perspective, the original U-Net microscopy challenge result used 35 images
with strong augmentation, whereas broader microscopy benchmarks contain
hundreds to thousands of independently acquired images (for example,
[LIVECell](https://www.nature.com/articles/s41592-021-01249-6) has 5,239
images and 1.69 million annotated cells). Treat the current 26 annotated
`all_images` training PNGs as a pipeline smoke-test dataset, not a sufficient
scientific training corpus.

As a practical starting target, collect at least hundreds of independent image
fields overall, with substantial instance counts for **every** phenotype and a
frozen, independently acquired validation/test set. Rotations, flips, color
jitter, mosaics, and crops can reduce overfitting to orientation or intensity;
they cannot create missing `dip` examples, novel lysis morphologies, microscope
settings, batches, or biological replicates. Never count augmented derivatives
as independent images, and split by acquisition/experiment before augmentation
to avoid train/test leakage.

## 6. Annotate canonical raw data

Use `--format raw` to ensure annotation goes through the same loader and PNG conversion path:

```bash
python annotate_images.py \
  --model yolov8s-seg_yfusion.pt \
  --input data/raw \
  --format raw \
  --output results.csv
```

Do not point training or annotation at legacy folders after migration. Do not rename one member of a TIFF channel set or its mask independently: matching is by the exact `<sample-id>`, not by order or partial filename matching.

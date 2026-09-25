# Guide for coding agents

This repository segments yeast fusion microscopy images with YOLOv8 instance
segmentation. Preserve the notebook-compatible, seven-class data contract.
Read `DATASET_NAMING.md`, `DATASET_PREPARATION.md`, and `TRAINING_CLI.md`
before changing image discovery, annotation conversion, or model interfaces.

## Non-negotiable data contract

**Filenames are part of the dataset API.** Do not implement fuzzy matching,
alphabetical pairing, or silent fallback for malformed names: discovery,
conversion, training, and annotation must fail rather than pair a wrong mask.

```text
data/raw/
  40x/
    images/czi/<sample-id>.czi
    images/tiff/<sample-id>__bf.tiff
    images/tiff/<sample-id>__gfp.tiff       # optional
    images/tiff/<sample-id>__rfp.tiff       # optional
    annotations/<sample-id>__mask.h5        # optional; .hdf5 is accepted
  other/
    images/czi/...
    images/tiff/...
    annotations/...
```

- `<sample-id>` is lowercase letters, digits, and single hyphens only.
- Extensions are lowercase; a sample ID is unique within its magnification
  across CZI and TIFF sources.
- `40x` contains only 40× acquisitions; every other magnification belongs in
  `other`.
- Raw CZI/TIFF/HDF5 files are immutable. Generated PNGs belong under
  `data/derived/png`, and generated YOLO data under `data/yolo_datasets`.
- An image without a mask is inference-only. Do not fabricate a biological
  ground-truth label for it.

## Dataset implementation rules

`MicroscopyImageDataset` in `image_dataset.py` is the sole raw-data entry
point. It discovers records, resolves the deterministic `annotation_path`,
materializes model-ready PNGs, and saves/loads dataset-info JSON. Call it
instead of globbing raw files in training, annotation, or inference code.

- CZI reading uses ImageJ/Fiji through `pyimagej`; run CZI conversion in the
  `yeast_fusion_segmenter` mamba environment.
- TIFF data use the required BF channel plus optional GFP/RFP channels. The
  generated RGB channel order is BF/GFP/RFP; channel normalization happens
  before writing PNGs.
- Stack frames receive a deterministic `__f0000`-style suffix. If a PNG frame
  is named with this suffix, select its matching HDF5 `T<n>` frame, not the
  first mask frame.
- `prepare_yolo_data.py --file-format raw` applies the notebook's 1024-pixel
  crop/pad jointly to the RGB PNG and mask before contour extraction: center
  crop/pad for CZI, upper-left crop/pad for TIFF. Never crop an image and its
  mask independently.
- CZI masks undergo the notebook's deterministic boundary-alignment check
  across original, vertical-flip, horizontal-flip, and 180° orientations. A
  flip is applied only when overlap improves by more than 0.10 and mean edge
  distance decreases. Do not remove this check or replace it with filename
  heuristics; some source CZI files require a vertical mask flip.
- The current mixed CZI-holdout dataset quarantines `p1-1g7-08` and
  `p1-3c12-15` after visual orientation QC. Build it with
  `--exclude-samples p1-1g7-08,p1-3c12-15`; do not delete or modify raw files.
  Its validation/test source IDs are `p1-1g2-09` and `p1-1e3-13` respectively.
- Use `manual_orientation_review_app.py` for a human-approved mask transform
  when automatic CZI alignment is ambiguous. It records exact keys in
  `data/manual_orientation_overrides.json`; pass that file through
  `prepare_yolo_data.py --orientation-overrides` during a rebuild to export
  corrected PNG/YOLO labels directly. Derived HDF5 copies are fallback only
  for tools that explicitly require HDF5.
- Rebuilding an output dataset replaces that output directory. Do not point it
  at user source data or a broad directory.

### Fixed annotation classes

The class IDs and order are immutable:

| ID | Class | HDF5 pixel rule |
| ---: | --- | --- |
| 0 | `f` | `0 <= value < 1000`, excluding zero background |
| 1 | `h` | `1000 <= value < 2000` |
| 2 | `lmcf` | `2000 <= value < 3000` |
| 3 | `lmsgfp` | `3000 <= value < 4000` |
| 4 | `lsgfp` | `4000 <= value < 5000` |
| 5 | `dip` | `5000 <= value < 6000` |
| 6 | `d2` | `6000 <= value < 7000` |

Do not collapse, reorder, or renumber these classes. Older three-class data
and checkpoints are historical segmentation baselines only and are invalid for
phenotype classification.

## Building datasets

Activate the supported environment before CZI operations:

```bash
mamba activate yeast_fusion_segmenter
```

Create a manifest or inspect records programmatically:

```python
from image_dataset import MicroscopyImageDataset

dataset = MicroscopyImageDataset("data/raw")
dataset.write_manifest()
png_records = dataset.materialize_pngs()
```

Build reproducible datasets through the CLI. The dataset-info JSON is the
durable contract passed to training and batch inference.

```bash
# 40× CZI only
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --magnification 40x --source-format czi \
  --output-dir data/yolo_datasets/40x_only \
  --dataset-info data/dataset_info/40x_only.json \
  --val-split 0.10 --test-split 0.10 --random-seed 42

# All CZI images
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --magnification all --source-format czi \
  --output-dir data/yolo_datasets/all_czi \
  --dataset-info data/dataset_info/all_czi.json \
  --val-split 0.10 --test-split 0.10 --random-seed 42

# All TIFF and CZI images
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --magnification all --source-format all \
  --output-dir data/yolo_datasets/all_images \
  --dataset-info data/dataset_info/all_images.json \
  --val-split 0.10 --test-split 0.10 --random-seed 42
```

Splits are deterministic by `--random-seed`. Validation/test fractions apply
to non-empty YOLO labels, so unannotated or empty-mask images cannot consume a
small supervised hold-out. Raw images with no HDF5 pairing stay in `test` for
inference. Inspect split and per-class coverage before interpreting metrics.

## YOLO data layout and YAML

The builder writes:

```text
data/yolo_datasets/<name>/
  train/images/*.png   train/labels/*.txt
  val/images/*.png     val/labels/*.txt
  test/images/*.png    test/labels/*.txt
  dataset.yaml
```

Each label file contains normalized YOLO segmentation polygons:
`<class-id> x1 y1 x2 y2 ...`. `dataset.yaml` has this shape:

```yaml
path: /absolute/path/to/data/yolo_datasets/<name>
train: train/images
val: val/images
test: test/images
names:
  0: f
  1: h
  2: lmcf
  3: lmsgfp
  4: lsgfp
  5: dip
  6: d2
```

`train_yolo.py --annotated-only` writes an adjacent `annotated_only/dataset.yaml`
whose splits are text files listing only images with non-empty labels. Prefer
`--dataset data/dataset_info/<name>.json` over directly passing YAML: it
retains raw-image and HDF5 provenance. Use `--data` only for a standalone
standard YOLO dataset.

## Training and augmentation

The default CLI uses `yolov8n-seg.pt`, 100 epochs, 1024px images, batch 8,
device `0`, and four workers. Default training augmentation/hyperparameters in
`train_yolo.py` are: HSV H/S/V 0.01/0.01/0.01; rotation ±180°; translate 0.1;
scale 0.1; shear 0.1; vertical/horizontal flips 0.5/0.5; mosaic 0.2; mixup
0.0; perspective 0.0. Optimizer-related defaults are `lr0=0.001`,
`lrf=0.0001`, momentum 0.5, weight decay 0.0001, and 3 warmup epochs.
Override them only with a versioned `--hyp` YAML and document the change.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images.json --annotated-only \
  --model yolov8n-seg.pt --epochs 50 --batch-size 4 --img-size 1024 \
  --device 0 --workers 2 --output models/all_images_yolov8n_seg_7class_50e.pt
```

Training writes Ultralytics artifacts under
`runs/segment/yolo_training/models/` and copies the best checkpoint to
`--output`. Use a new output/run name for every run; the CLI intentionally
does not reuse an existing run directory.

For every materially changed dataset, train a fresh iteration from the
pretrained base model with a new output name. Do not continue from a prior
experiment checkpoint; document the dataset-info, overrides, exclusions, and
evaluation in [ITERATIVE_TRAINING_WORKFLOW.md](ITERATIVE_TRAINING_WORKFLOW.md).

To reproduce the final notebook model run, add `--notebook-protocol`. It uses
`yolov8s-seg.pt`, 1,000 epochs, batch 20, eight workers, `nbs=32`, and the
notebook's 180° rotation/flip settings with mosaic, mixup, and copy-paste
disabled. Explicit CLI values override those defaults for a short test.

### Dataset-size limitation

Do not present augmentation as a substitute for biological data. Small
microscopy examples can demonstrate a pipeline (U-Net's original challenge
used 35 images with strong augmentation), but broad generalization benchmarks
use hundreds to thousands of independently acquired fields; for example,
[LIVECell](https://www.nature.com/articles/s41592-021-01249-6) contains 5,239
images and 1.69 million annotated cells. The current 26 annotated mixed-dataset
training PNGs are a smoke-test baseline only. Aim for hundreds of independent
fields overall, with substantial instances per phenotype and an acquisition- or
experiment-level held-out test set. Rotation, flips, HSV jitter, mosaic, and
crops do not add missing phenotypes, biological replicates, modalities, or
microscope conditions. Split before augmentation and never report augmented
derivatives as independent samples.

## Evaluation and inference

Evaluate only labelled data using the dataset object and `--annotated-only`:

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images.json --annotated-only --evaluate \
  --eval-split test --model models/all_images_yolov8n_seg_7class_50e.pt \
  --img-size 1024 --batch-size 4 --device 0 --workers 2
```

This produces a new run under `runs/segment/yolo_evaluation/`. Do not report
metrics that include unannotated inference-only images as if they were ground
truth.

For a single file or a raw directory, use the annotation CLI (CZI requires the
mamba/ImageJ environment):

```bash
python annotate_images.py --model models/all_images_yolov8n_seg_7class_50e.pt \
  --input path/to/image_or_directory --format auto --output predictions.csv \
  --imgsz 1024 --crop 1024
```

For a prepared dataset, prefer provenance-preserving batch inference. The
combined CSV includes dataset/split, sample ID, magnification, source paths,
and paired annotation path for every detected instance:

```bash
# All prepared images, including inference-only images
CUDA_VISIBLE_DEVICES=0 python batch_predict.py \
  --dataset data/dataset_info/all_images.json --split all \
  --model models/all_images_yolov8n_seg_7class_50e.pt \
  --output_csv predictions/all_images.csv

# Annotated held-out test images only
CUDA_VISIBLE_DEVICES=0 python batch_predict.py \
  --dataset data/dataset_info/all_images.json --split test --annotated-only \
  --model models/all_images_yolov8n_seg_7class_50e.pt \
  --output_csv predictions/all_images_test.csv
```

Do not provide both `--dataset` and `--input_dir` to `batch_predict.py`.
Use `--zoom` only when overlapping-crop inference is intended; its output
contains crop IDs and crop coordinates and should not be treated as unique
whole-image detections without downstream deduplication.

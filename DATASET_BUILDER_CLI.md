# Dataset builder CLI

`build_dataset.py` is the canonical command for creating a versioned YOLO
dataset from the strict `data/raw` CZI/TIFF/HDF5 layout. It always invokes the
shared `MicroscopyImageDataset` and the notebook-compatible seven-class
conversion path. It writes both a prepared YOLO dataset and a durable
dataset-info JSON that is passed unchanged to training, evaluation, batch
annotation, and the browser label editor.

**Raw file naming must follow [DATASET_NAMING.md](DATASET_NAMING.md) exactly.**
The tool fails on malformed or ambiguous names instead of guessing an
image/mask pairing.

## Build a dataset

```bash
mamba activate yeast_fusion_segmenter

python build_dataset.py --name iteration-003 \
  --raw-root data/raw --source-format all --magnification all \
  --val-split 0.15 --test-split 0.15 --random-seed 42
```

This writes:

```text
data/yolo_datasets/iteration-003/
  train/images/*.png   train/labels/*.txt
  val/images/*.png     val/labels/*.txt
  test/images/*.png    test/labels/*.txt
  dataset.yaml
data/dataset_info/iteration-003.json
```

The output JSON records raw source and HDF5 paths, selected magnification and
source formats, exclusions, exact split policy, class map, PNG provenance, and
the absolute `dataset.yaml` path. It is the dataset contract; do not hand-edit
it or substitute a YAML file when provenance matters.

## Common build variants

```bash
# Only 40× CZI acquisitions.
python build_dataset.py --name czi-40x-v1 --magnification 40x \
  --source-format czi --val-split 0.2 --test-split 0 --random-seed 42

# All CZI acquisitions with named acquisition-level holdouts.
python build_dataset.py --name all-czi-heldout-v1 --source-format czi \
  --val-samples p1-1g2-09 --test-samples p1-1e3-13 \
  --val-split 0 --test-split 0

# Mixed TIFF/CZI dataset while quarantining visually unresolved sources.
python build_dataset.py --name mixed-reviewed-v1 --source-format all \
  --exclude-samples p1-1g7-08,p1-3c12-15 \
  --orientation-overrides data/manual_orientation_overrides.json \
  --val-split 0.1 --test-split 0.1 --random-seed 42
```

The default paths derive from `--name`. Supply `--output-dir` and/or
`--dataset-info` only when an alternative versioned location is required.
The tool refuses to overwrite an existing output or JSON by default. Use a new
name for a new iteration. `--resume` is only for an interrupted build;
`--replace` explicitly permits rebuilding a disposable derived output.

## Reuse the dataset-info JSON

```bash
# Train only on non-empty, HDF5-backed labels.
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/iteration-003.json --annotated-only \
  --notebook-protocol --output models/iteration-003_yolov8s.pt

# Evaluate the held-out labelled subset.
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/iteration-003.json --annotated-only --evaluate \
  --eval-split test --model models/iteration-003_yolov8s.pt

# Batch-annotate prepared image PNGs with retained raw/HDF5 provenance.
CUDA_VISIBLE_DEVICES=0 python batch_predict.py \
  --dataset data/dataset_info/iteration-003.json --split all \
  --model models/iteration-003_yolov8s.pt \
  --output_csv predictions/iteration-003.csv

# Hand-correct prepared training labels before a fresh training run.
python manual_yolo_annotation_app.py \
  --dataset data/dataset_info/iteration-003.json --split train
```

`annotate_images.py` remains the single-file/raw-directory annotation command.
For a prepared dataset object, use `batch_predict.py --dataset`; it is the
annotation workflow that retains dataset split and source/annotation metadata.

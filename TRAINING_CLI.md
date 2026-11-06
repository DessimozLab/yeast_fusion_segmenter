# Training from a prepared dataset

[Documentation index](README.MD#documentation) ·
[Build a dataset](DATASET_BUILDER_CLI.md) ·
[Next: annotation and review](ANNOTATION_LIFECYCLE.md) ·
[Model results](MODEL_PERFORMANCE.md) ·
[Matched pilot](ANNOTATION_PILOT.md)

Use the `--dataset` argument to train from a saved dataset definition. It is
the preferred interface because it records the exact raw records, HDF5 masks,
magnification/source-format filters, split seed, and YOLO YAML used for the
run.

Create that definition with the canonical builder; it writes
`data/dataset_info/<name>.json` by default. See
[DATASET_BUILDER_CLI.md](DATASET_BUILDER_CLI.md) for source-format filters,
splits, named holdouts, and orientation overrides.

```bash
python build_dataset.py --name all-images-v2 --raw-root data/raw \
  --source-format all --magnification all \
  --val-split 0.1 --test-split 0.1 --random-seed 42
```

```bash
mamba activate yeast_fusion_segmenter

# 40× CZI data only
python train_yolo.py --dataset data/dataset_info/40x_only.json

# Every TIFF and CZI acquisition
python train_yolo.py --dataset data/dataset_info/all_images.json

# Every CZI acquisition
python train_yolo.py --dataset data/dataset_info/all_czi.json
```

`--data path/to/dataset.yaml` remains available for a plain YOLO YAML, but it
does not preserve the raw-data provenance held by a dataset-info JSON.

## Common options

Version-2 datasets enforce inventories, hashes and valid human approvals.
`--data` cannot bypass the adjacent candidate-review contract. Evaluation always
filters to nonempty labels, including when `--annotated-only` is omitted.
Use `--seed 42` and `--run-dir data/derived/<experiment>/training` to make
protocols/artifact placement explicit. Each run saves `protocol_metrics.json`
with dataset/checkpoint hashes, effective settings, versions and per-class
box/mask metrics. An existing `--output` checkpoint cannot be overwritten.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images.json \
  --model yolov8n-seg.pt \
  --epochs 100 --batch-size 8 --img-size 1024 --device 0 \
  --workers 4 --output models/all_images_yolov8n_seg.pt
```

- `--model`: pretrained segmentation checkpoint to fine-tune.
- `--epochs`: maximum epoch safety cap (default 1000), not a convergence claim.
- `--batch-size`, `--img-size`: training batch and input size.
- `--device 0`: use CUDA GPU 0; use `--device cpu` only when CUDA is absent.
- `--output`: durable copy of the run’s best checkpoint. Training logs and
  artifacts remain in `runs/segment/`.
- `--hyp`: optional YAML override for augmentation and optimizer settings.

## Seeded training to convergence

All training uses `--seed 42` by default and `deterministic=True`. Each run
starts fresh; do not continue an earlier experiment checkpoint for these analyses.
Determinism is bounded by the hardware and software versions recorded in
`protocol_metrics.json`, not a guarantee of bitwise equality across systems.

The trainer monitors the **sum of all validation loss components**, including
box, segmentation, classification and DFL loss, after warmup. It uses PyTorch
[ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html):

- `--lr-patience 10`: reduce LR after more than 10 epochs without a relative
  loss improvement of `1e-4`.
- `--lr-factor 0.5`: multiply LR by 0.5 at a plateau.
- `--min-lr 0.000001`: learning-rate floor.
- `--convergence-patience 40`: stop after 40 epochs without significant loss
  improvement **while at the floor**. An improvement resets this wait.
- `--epochs 1000`: safety cap. `convergence.json` distinguishes `converged`
  from `epoch_cap_reached`; never describe a capped run as converged.

The old fixed LR decay and mAP early stopper are replaced. AdamW is now the
explicit default optimizer so Ultralytics' automatic optimizer cannot override
the requested initial LR. A versioned `--hyp` can select another explicit
optimizer, but not `auto`, a different seed/stopping policy, or classification
`dropout`. The legacy `lrf` value is recorded but no longer controls fixed decay;
the LR floor comes from `--min-lr`. Warmup retains Ultralytics' minimum 100 batches, so it can last more
than three epochs on tiny datasets. Its LR ramp is not a validation plateau.
The usual `best.pt` remains the checkpoint with the best **validation fitness**,
not necessarily minimum loss. Neither test loss nor test AP controls training.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all-images-v2.json --annotated-only \
  --model yolov8s-seg.pt --seed 42 --device 0 --batch-size 4 \
  --epochs 1000 --lr-patience 10 --lr-factor 0.5 \
  --min-lr 0.000001 --convergence-patience 40 \
  --run-dir data/derived/convergence-v1/training \
  --output models/convergence-v1.pt
```

Currently use one device (`--device 0`, or `cpu` for a software smoke test).
Multi-GPU/DDP is rejected rather than running unsynchronized loss schedulers.
Long experiment runs are not launched merely by running the regression suite.

## Cross-validation analysis

`--cross-validate` defaults to **10 repeated grouped holdout runs**, not ten
folds. This permits CV on datasets with fewer than ten independent acquisitions.
It requires a version-2 training/compiled `--dataset`, at least two annotated
training acquisitions, and a nonempty labelled frozen test set. Use the explicit
[legacy migration](ANNOTATION_LIFECYCLE.md#version-2-contracts-and-legacy-migration)
if necessary; candidate predictions are rejected.

For run index `i=0..9`, both split and training seeds are predefined as
`--seed + i` (42–51 by default). Reserve `--cv-val-fraction 0.2` of the
**original training acquisitions** for validation, rounded up, with at least
one acquisition left for training. Sibling stack frames, including empty ones,
stay together. Only nonempty labels contribute to supervised training/evaluation.
There is no phenotype stratification; inspect class support in each run.

The original validation acquisitions are excluded from CV entirely. Frozen
test acquisitions stay test in every run, and inference-only pool images never
become ground truth. The command copies only derived PNG/YOLO files into fresh
version-2 datasets, preserving raw identities/hashes, approvals and teacher
lineage; it never rebuilds raw masks or changes the source split.

```bash
# Default: 10 runs, seed schedule 42–51; run on CUDA 0.
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all-images-v2.json --cross-validate \
  --model yolov8s-seg.pt --seed 42 --device 0 --batch-size 4 \
  --cv-output data/derived/cv-all-images-v1

# A different prepared dataset and explicit repeat count/fraction.
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all-czi-v2.json --cross-validate \
  --cv-runs 10 --cv-val-fraction 0.25 --seed 42 \
  --device 0 --batch-size 4 --cv-output data/derived/cv-czi-v1
```

The output directory must not exist. `--cv-output` controls all analysis
artifacts; ordinary `--run-dir`/`--output` are not per-run destinations in
analysis mode. The default root is `data/derived/cross_validation/<output-stem>/`,
following this machine's existing data-disk symlink when installed.

Outputs: `plan.json` records seeds, acquisition membership and original holdout
hashes; each `run-XX/` contains dataset JSON/contract, a fresh `checkpoint.pt`,
`training/checkpoint/convergence.json`, training protocol metrics, and independent
`evaluation/{val,test}/checkpoint/protocol_metrics.json`. `completed_runs.json`
retains finished runs if a later one fails; reruns require a new output directory.
`report.md`, `summary.json`, `metrics.csv`, `metrics_error_bars.png` and
`per_class_error_bars.png` summarize aggregate and per-class results.

Error bars are **sample standard deviation between runs**, not confidence
intervals. Repeated splits overlap, and test measurements reuse the same fields;
ten runs are not ten independent biological samples. Missing-class AP is null,
not invented as zero. Do not choose a model or tune hyperparameters on test AP.

## Separate dropout-sweep analysis

`--dropout-sweep` is independent of CV and cannot be combined with
`--cross-validate` or `--evaluate`. It keeps the supplied training, validation
and test splits fixed, requires a validated version-2 dataset with nonempty
labels in all three splits, and initializes every run from the same stock
`yolov8[nslmx]-seg.pt` base. All convergence controls above apply.

The standard Ultralytics [`dropout` option is classification-only](https://docs.ultralytics.com/modes/train/).
This analysis instead inserts **channel-wise `torch.nn.Dropout2d` after the
YOLOv8 backbone's final SPPF output activation**. It is active only in training;
validation/test/inference use ordinary deterministic evaluation, not Monte Carlo
dropout. The checkpoint contains standard PyTorch modules, so existing
annotation/evaluation commands can load it without custom hooks or classes.
This tests one documented dropout location, not every possible architecture.

```bash
# Default rates 0, 0.1, 0.2, 0.3, 0.5; three repeats each = 15 fresh runs.
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all-images-v2.json --dropout-sweep \
  --model yolov8s-seg.pt --seed 42 --device 0 --batch-size 4 \
  --sweep-output data/derived/dropout-all-images-v1

# Explicit rates and five paired repeats each = 20 fresh runs.
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all-images-v2.json --dropout-sweep \
  --dropout-rates 0 0.1 0.25 0.5 --dropout-runs 5 \
  --model yolov8s-seg.pt --seed 42 --device 0 --batch-size 4 \
  --sweep-output data/derived/dropout-all-images-v2
```

Rates must be unique probabilities in `[0,1)` and include zero. At least two
rates and two repeats are required. Repeat `i` uses seed `42+i` at **every**
rate, yielding paired comparisons on unchanged acquisitions. Output roots must
be new; default is `data/derived/dropout_analysis/<output-stem>/`. A sweep writes
`plan.json`, per-rate/per-run checkpoints, loss histories and val/test protocol
metrics, plus `report.md`, `summary.json`, `metrics.csv` and
`dropout_effect.png`. Tables include mean ± sample SD and mean ± SD of the
paired difference from zero dropout; per-rate reports include per-class metrics.
**Select a rate using validation only.** Comparing many test scores does not
make the test set a tuning set or establish a generalization improvement.

After choosing a rate, train a fresh final iteration using, for example,
`--feature-dropout 0.2` with the ordinary training command and a new output name.
To evaluate or annotate its checkpoint, use the existing CLI unchanged.

For an opt-in end-to-end CPU **software** check (two short CV runs and four
dropout runs on synthetic images, no biological review), see
[the testing guide](tests/README.md#training-analysis-checks).

## Notebook architecture and augmentation preset

`segment_retrain(1).ipynb` trained `yolov8s-seg.pt` at 1024 px for 1,000
epochs with batch size 20, eight workers, `nbs=32`, 180° rotations, 0.5
vertical/horizontal flips, and **no** mosaic, mixup, or copy-paste. Use
`--notebook-protocol` to select those settings; explicit CLI values override
the protocol defaults for a shorter test run. The current seeded, loss-plateau
optimizer/stopping policy above supersedes the historical fixed-epoch schedule;
this is no longer an exact reproduction of its optimizer trajectory.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images.json --annotated-only \
  --notebook-protocol --device 0 \
  --output models/all_images_yolov8s_notebook.pt
```

The notebook has an exploratory offline augmentation section, but its final
training YAML points to `datasets/train` rather than the `augmented/` folder.
The CLI therefore reproduces the final model-training settings rather than
silently duplicating offline samples. The canonical TIFF preparation also
matches the notebook's upper-left crop; CZI uses its separately documented
center-crop and orientation-alignment logic.

### Test zoom/crop augmentation

`--zoom-augmentation` increases the online geometric transform from the
notebook's `scale=0.1, translate=0.1` to `scale=0.5, translate=0.2`.
Ultralytics applies this transform jointly to each image and segmentation mask;
an enlarged view is consequently cropped by the 1024px canvas. Use it as a
controlled experiment, not as a replacement for independently annotated CZI
fields.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images_czi_holdout.json --annotated-only \
  --notebook-protocol --zoom-augmentation --device 0 \
  --output models/all_images_czi_holdout_yolov8s_zoom.pt
```

The prepared definitions are selected at build time, not train time. To change
the images or split, rebuild a new dataset-info file using
`prepare_yolo_data.py --file-format raw`; then pass that new file to
`train_yolo.py --dataset`.

### Current quarantined CZI holdout run

The current mixed dataset quarantines the visually uncertain 40× fields
`p1-1g7-08` and `p1-3c12-15`. Rebuild it reproducibly (using `p1-1g2-09` for
validation and `p1-1e3-13` for test), then train at a GPU-compatible batch
size. Do not use the earlier checkpoints trained before this quarantine.

```bash
python prepare_yolo_data.py --resume --input-dir data/raw --file-format raw \
  --source-format all --magnification all \
  --exclude-samples p1-1g7-08,p1-3c12-15 \
  --val-samples p1-1g2-09 --test-samples p1-1e3-13 \
  --val-split 0 --test-split 0 --random-seed 42 \
  --output-dir data/yolo_datasets/all_images_czi_holdout \
  --dataset-info data/dataset_info/all_images_czi_holdout.json

CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
python train_yolo.py --dataset data/dataset_info/all_images_czi_holdout.json \
  --annotated-only --notebook-protocol --batch-size 4 --device 0 \
  --output models/all_images_czi_holdout_yolov8s_quarantined_b4_retry.pt
```

## Manually correct prepared YOLO labels

Before training a new dataset iteration, inspect individual PNGs in the local
browser editor. Select one of the fixed seven phenotype classes, draw a
three-or-more-point polygon to add an instance, or choose **Remove by click**
to delete an existing polygon. The tool edits only the selected prepared label
file; raw CZI/TIFF/HDF5 sources remain immutable.

```bash
python manual_yolo_annotation_app.py \
  --dataset data/dataset_info/cycle-v1-review.json --split train
# Open http://127.0.0.1:8767 in a local browser.
```

The initial edit preserves `<image>.txt.manual-backup`; every save is logged
to `manual_annotation_edits.jsonl` under the prepared dataset. Re-render the
overlay stack before training, and use a fresh dataset/model version after
material corrections. Full raw-to-reviewed-dataset instructions are in
[ANNOTATION_LIFECYCLE.md](ANNOTATION_LIFECYCLE.md).

## Visualize ground-truth contours

Use `create_label_overlays.py` to render the exact prepared YOLO polygons used
for training and evaluation. It writes one PNG per image, retains the
`train`/`val`/`test` split layout, and creates a `manifest.csv` plus a
class-colour `legend.txt`. Images with no paired annotation are retained and
marked `no annotation`.

```bash
python create_label_overlays.py \
  --dataset data/dataset_info/all_images_czi_holdout.json \
  --output-dir validation/all_images_czi_holdout_label_overlays
```

Alternatively, point directly at a prepared YOLO YAML:

```bash
python create_label_overlays.py \
  --data data/yolo_datasets/all_images_czi_holdout/dataset.yaml \
  --output-dir validation/all_images_czi_holdout_label_overlays
```

## Train or evaluate only annotated images

`--annotated-only` creates a filtered YOLO YAML under the prepared dataset and
keeps only image files whose corresponding YOLO label is non-empty. Labels
come from paired HDF5 masks or explicitly approved YOLO reviews compiled with
`annotation_cycle.py compile`. Unlabelled pool images remain inference-only.
Datasets marked `purpose: candidate_review` are rejected for training/evaluation.
Use [ANNOTATION_LIFECYCLE.md](ANNOTATION_LIFECYCLE.md) for the tested review loop.

```bash
# Fine-tune a pretrained model using annotated images only.
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_czi.json --annotated-only \
  --model yolov8n-seg.pt --device 0 --epochs 100 --output models/czi.pt

# Evaluate a pretrained or trained model on only annotated held-out images.
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images.json --annotated-only --evaluate \
  --eval-split test --model models/all_images_yolov8n_seg.pt --device 0
```

Use `--eval-split train`, `val`, or `test` to select the desired split. An
evaluation split with no annotated images is rejected by YOLO rather than
silently reporting a score.

## Seven-class phenotype requirement

The current builder follows the seven-class HDF5 encoding in
`segment_retrain(1).ipynb`: `f`, `h`, `lmcf`, `lmsgfp`, `lsgfp`, `dip`, and
`d2`. Rebuild every dataset-info JSON after updating the preparation code;
older datasets and checkpoints use a collapsed three-class label map and are
not valid phenotype-classification models.

## Inference from a saved dataset

`batch_predict.py` also accepts the same dataset-info JSON. It reads the
prepared PNG images referenced by that object, rather than rediscovering raw
files or reopening CZI files. Each detected instance in the combined CSV is
tagged with `dataset_name`, `dataset_split`, `sample_id`, `magnification`,
`source_format`, `source_paths`, and `annotation_path`.

```bash
# Run a model over every prepared image, including inference-only images.
CUDA_VISIBLE_DEVICES=0 python batch_predict.py \
  --dataset data/dataset_info/all_images.json --split all \
  --model models/all_images_yolov8n_seg.pt \
  --output_csv predictions/all_images.csv

# Restrict prediction to annotated held-out images for an audit or comparison.
CUDA_VISIBLE_DEVICES=0 python batch_predict.py \
  --dataset data/dataset_info/all_czi.json --split test --annotated-only \
  --model models/all_czi_yolov8n_seg.pt \
  --output_csv predictions/czi_annotated_test.csv
```

Use either `--dataset` or the legacy `--input_dir` plus `--format` interface,
not both. Dataset-object inference requires a prepared `yolo_data` path in
the JSON; build it first with `prepare_yolo_data.py --file-format raw`.

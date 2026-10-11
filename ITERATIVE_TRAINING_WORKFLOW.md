# Iterative annotation and fresh-model training workflow

For the tested complete prediction → browser approval → reviewed-dataset
compilation → fresh retraining loop, follow
[ANNOTATION_LIFECYCLE.md](ANNOTATION_LIFECYCLE.md). Use
`annotation_cycle.py export` for editable candidate polygons and
`annotation_cycle.py compile` to preserve approved corrections. Rebuilding
from raw HDF5 alone will discard those corrections.

Compiled datasets are immutable in the browser: export an isolated review copy
before editing. Version-2 contracts enforce image/label hashes, explicit
holdout/pool roles and review lineage. Current unfinished biological work and
the approved-budget pilot are in [ANNOTATION_CYCLE_GAPS.md](ANNOTATION_CYCLE_GAPS.md)
and [ANNOTATION_PILOT.md](ANNOTATION_PILOT.md).

Use this workflow when adding microscopy fields, correcting their annotations,
and training the next yeast-fusion segmentation model. It preserves the
notebook-compatible seven-class contract and keeps raw scientific files
immutable.

**Always train a fresh model iteration after changing the dataset.** Do not
continue training from a previous experiment checkpoint: it entangles the old
and new dataset versions, invalidates clean comparisons, and makes correction
effects impossible to audit. Start from the standard pretrained
`yolov8s-seg.pt` with a new `--output` name for every dataset version.

## 1. Add raw images and annotations

Follow the mandatory names and layout in [DATASET_NAMING.md](DATASET_NAMING.md).
Put new raw images and HDF5 masks under the matching magnification folder;
never edit raw CZI, TIFF, or HDF5 files in place.

```text
data/raw/<40x|other>/images/czi/<sample-id>.czi
data/raw/<40x|other>/images/tiff/<sample-id>__bf.tiff
data/raw/<40x|other>/annotations/<sample-id>__mask.h5
```

Every HDF5 mask must correspond to the exact image sample ID and have the
same orientation as the converted PNG. Preserve all seven classes: `f`, `h`,
`lmcf`, `lmsgfp`, `lsgfp`, `dip`, and `d2`.

## 2. Annotate new images

Use the current model only to create candidate annotations for human review;
model predictions are not biological ground truth. Export prepared PNG
predictions with `annotation_cycle.py export`, then inspect/correct and approve
them in the browser. Use HDF5 only when a separate laboratory tool requires it.

```bash
python annotate_images.py \
  --model models/all_images_czi_holdout_yolov8s_quarantined_b4_retry.pt \
  --input path/to/new_images --format auto \
  --output predictions/new_images.csv --imgsz 1024 --crop 1024
```

For prepared data, use `batch_predict.py --dataset ...` to retain source and
HDF5 provenance. Predictions are useful as an annotation aid only: compile
explicitly approved YOLO labels before including a new field in supervision.

## 3. Review and manually correct ambiguous mask orientation

Materialize candidate PNGs and use the localhost-only browser tool:

```bash
mamba activate yeast_fusion_segmenter
python manual_orientation_review_app.py \
  --raw-root data/raw --overrides data/manual_orientation_overrides.json
```

Open `http://127.0.0.1:8765`. For each ambiguous annotated image, choose the
overlay (`orig`, `flip_ud`, `flip_lr`, or `flip_udlr`) whose contours align
with the image. The tool writes an exact
`<magnification>/<sample-id> -> transform` choice to the override JSON.

It does not modify raw files. If another tool needs corrected HDF5 files,
export derived copies with identical HDF5 hierarchy and storage protocol. This
is a fallback for HDF5-dependent tools, **not** the primary training export.

```bash
python export_corrected_hdf5_masks.py \
  --raw-root data/raw --overrides data/manual_orientation_overrides.json \
  --output-root data/derived/manual_orientation_hdf5
```

## 4. Export directly to a versioned YOLO dataset

Create a new output directory and dataset-info JSON for every iteration. Do
not overwrite an experiment that has already been trained or evaluated.
This is the primary export path: it applies approved browser overrides while
writing model-ready PNG images and YOLO segmentation polygon labels directly.
No corrected HDF5 copy is required for YOLO training.

```bash
python prepare_yolo_data.py --input-dir data/raw --file-format raw \
  --source-format all --magnification all \
  --orientation-overrides data/manual_orientation_overrides.json \
  --val-split 0.15 --test-split 0.15 --random-seed 42 \
  --output-dir data/yolo_datasets/iteration_002 \
  --dataset-info data/dataset_info/iteration_002.json
```

Use explicit `--val-samples` and `--test-samples` when a field must be held
out by acquisition or experiment. Use `--exclude-samples` to quarantine a
record with unresolved quality/orientation issues; do not delete the source.

Render the ground-truth contours and inspect all changed or newly annotated
fields before training:

```bash
python create_label_overlays.py \
  --dataset data/dataset_info/iteration_002.json \
  --output-dir validation/iteration_002_label_overlays
```

Confirm that every expected class has training instances and that validation
and test fields are independent of training acquisitions.

Only export derived HDF5 masks with `export_corrected_hdf5_masks.py` when a
separate downstream application explicitly requires HDF5.

### Correct individual prepared YOLO labels in the browser

For hand correction of individual instances, use the prepared-label browser
editor on a new, untrained iteration. It directly edits only the derived YOLO
polygon labels, keeping raw CZI/TIFF/HDF5 files immutable.

```bash
python manual_yolo_annotation_app.py \
  --dataset data/dataset_info/iteration_002.json --split train
# Open http://127.0.0.1:8767 in a local browser.
```

Choose an image and one of the fixed seven classes, draw a polygon with at
least three clicks, and finish it to add an instance. Use **Remove by click**
to delete an instance, then save. The editor creates a one-time
`.txt.manual-backup` and appends a UTC audit record to
`manual_annotation_edits.jsonl` in the dataset directory. Refresh the contour
overlays after editing and retain the audit log with this dataset version.

## 5. Train a fresh network iteration

Start from the pretrained base model, never from the prior iteration's best
checkpoint. Give the new model a distinct output name. `--annotated-only`
ensures only non-empty imported HDF5 or approved compiled YOLO labels are used
for supervision. Candidate review datasets are rejected.

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
python train_yolo.py --dataset data/dataset_info/iteration_002.json \
  --annotated-only --notebook-protocol --batch-size 4 --device 0 \
  --output models/iteration_002_yolov8s.pt
```

`--notebook-protocol` starts from `yolov8s-seg.pt` and applies the final
notebook augmentation schedule. An explicit batch size may be reduced for
available GPU memory; document that deviation. Do not use
`--model models/iteration_001_*.pt` to create `iteration_002`.

## 6. Evaluate and archive the iteration

Evaluate the copied best checkpoint on annotated test images only:

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/iteration_002.json --annotated-only --evaluate \
  --eval-split test --model models/iteration_002_yolov8s.pt \
  --batch-size 4 --device 0
```

Record the exact dataset-info JSON, orientation override JSON, exclusions,
split IDs, training command, requested/completed epochs, checkpoint path, and
box/mask metrics. State the number of independent test fields alongside every
metric; augmented images and multiple frames from the same acquisition are not
independent biological replicates.

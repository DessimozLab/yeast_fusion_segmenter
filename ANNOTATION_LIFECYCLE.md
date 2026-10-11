# Training, annotation, manual QC and retraining

This is the supported loop: build a versioned dataset, train a fresh model,
export its polygons into an isolated review dataset, correct and approve them,
compile a new training dataset, evaluate a fresh model, and annotate again.
All commands use the same dataset-info JSON contract.

Current readiness and unresolved issues are in
[ANNOTATION_CYCLE_GAPS.md](ANNOTATION_CYCLE_GAPS.md). The real human-reviewed
pilot and matched 100-epoch protocol are in [ANNOTATION_PILOT.md](ANNOTATION_PILOT.md);
its biological rubric still needs laboratory approval.

### Version-2 contracts and legacy migration

New builds write `schema_version: 2`, a prepared image/label hash inventory,
raw source hashes, annotation origins, explicit train/val/test/pool roles, and
an adjacent `dataset-contract.json`. Physical `test` can contain an unlabelled
pool, but that pool is never biological test ground truth. Training and
evaluation validate these contracts; direct candidate YAML is also rejected.

Legacy manifests support historical training with a warning, but must be
explicitly migrated before candidate export or review compilation:

```bash
python annotation_cycle.py migrate --dataset data/dataset_info/legacy.json \
  --acknowledge-legacy
python annotation_cycle.py qc --dataset data/dataset_info/legacy.json
```

Migration snapshots current bytes, retains a `.legacy-backup`, and does not
certify biological correctness. It cannot refresh a version-2 dataset's hashes
after edits. If geometry is suspect, rebuild a fresh version rather than migrate
it as if conversion were verified. Fresh CZI conversion now correctly preserves
`source_format: czi`, matching cached CZI center-crop geometry.

Raw CZI/TIFF/HDF5 files stay immutable. Naming must respect
[DATASET_NAMING.md](DATASET_NAMING.md) exactly or discovery and pairing fail.
The seven IDs are fixed: `0=f, 1=h, 2=lmcf, 3=lmsgfp, 4=lsgfp, 5=dip, 6=d2`.
Direct YOLO polygons are the primary reviewed annotation format. HDF5 is
needed only for importing existing masks or a separate HDF5-dependent tool.

## 1. Prepare the images and existing annotations

Activate `mamba activate yeast_fusion_segmenter` for Fiji/CZI conversion.
Use the canonical raw folders described in DATASET_NAMING.md. The shared
dataset loader retains exact raw source paths and HDF5 pairing, and materializes
RGB PNGs in BF/GFP/RFP order. Preparation crops/pads each image and its mask
jointly and applies the documented orientation checks.

For the current mixed cohort, keep the two quarantined fields excluded and
freeze the acquisition-level validation and test IDs:

```bash
python build_dataset.py --name cycle-v1 --raw-root data/raw \
  --exclude-samples p1-1g7-08,p1-3c12-15 \
  --val-samples p1-1g2-09 --test-samples p1-1e3-13 \
  --val-split 0 --test-split 0

python annotation_cycle.py qc --dataset data/dataset_info/cycle-v1.json \
  --output validation/cycle-v1-qc.json

python create_label_overlays.py --dataset data/dataset_info/cycle-v1.json \
  --output-dir validation/cycle-v1-label-overlays
```

For a different cohort, fractions such as `--val-split 0.15 --test-split 0.15`
allocate whole annotated acquisitions, keeping TIFF stack frames together.
Unannotated images remain an inference pool in `test`; they are not labelled
test examples. Freeze splits before any training or augmentation.

Inspect the overlays before training. If mask orientation is ambiguous, use
`manual_orientation_review_app.py` and rebuild the new version with
`--orientation-overrides data/manual_orientation_overrides.json`.
QC validates class IDs, polygon coordinates, acquisition leakage, missing
training phenotypes and suspicious subpixel polygons. A missing training class
means that class cannot be learned even though it is present in the YAML.

## 2. Train and measure an initial model

Always start a changed dataset iteration from the standard pretrained base,
with a distinct output name. Never resume the prior iteration's checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/cycle-v1.json --annotated-only \
  --notebook-protocol --batch-size 4 --device 0 \
  --output models/cycle-v1.pt

CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/cycle-v1.json --annotated-only --evaluate \
  --eval-split test --model models/cycle-v1.pt \
  --img-size 1024 --batch-size 4 --device 0
```

The notebook protocol defaults to YOLOv8s-seg, 1,000 epochs, 1024px, `nbs=32`,
180° rotation and horizontal/vertical flips, with mosaic/mixup/copy-paste
disabled. Batch 4 is an explicit memory-related deviation. Record the effective
Ultralytics version and optimizer: `optimizer=auto` can override requested
learning rate and momentum. Training transforms do not create independent
biological fields or missing phenotypes.

## 3. Export model candidates for browser correction

CSV measurements are not editable segmentation labels. Export normalized YOLO
polygons with `annotation_cycle.py export`, which preserves PNG geometry,
raw/HDF5 provenance, checkpoint hash and inference settings:

```bash
CUDA_VISIBLE_DEVICES=0 python annotation_cycle.py export \
  --dataset data/dataset_info/cycle-v1.json --model models/cycle-v1.pt \
  --split all --confidence 0.25 --imgsz 1024 --device 0 \
  --name cycle-v1-review \
  --output-dir data/yolo_datasets/cycle-v1-review \
  --dataset-info data/dataset_info/cycle-v1-review.json
```

Each image has a PNG and label file, even when the model detects nothing.
The review dataset has `purpose: candidate_review`; the training/evaluation
CLI rejects it. A prediction is an annotation aid, not verified ground truth.
Export requires a seven-class segmentation checkpoint; historical collapsed
models and COCO checkpoints are rejected.

## 4. Check, correct and explicitly approve images in the browser

```bash
python manual_yolo_annotation_app.py \
  --dataset data/dataset_info/cycle-v1-review.json --split train
# Open http://127.0.0.1:8767.
```

Select an image and phenotype class. Draw at least three polygon vertices,
then finish. Remove a wrong polygon by clicking inside it; use **Change class
by click** to reclassify one. Use the zoom control and scroll to inspect small
cells and dense fields at native resolution. A malformed/zero-area polygon is
rejected. Unsaved navigation prompts before discarding work.

Only isolated candidate-review images with train/pool roles are editable.
Compiled datasets, legacy datasets without a contract, and labelled holdouts
are read-only. **Defer unresolved image** records reviewer/notes and excludes
that field from compilation. Any save marks prior approval changed; explicit
reapproval is required. Do not run multiple server processes on one review tree.
Within one server, stale browser-tab saves return HTTP 409: reload and reconcile
your edits rather than overwrite another reviewer's work.

**Save labels** records a draft. **Save and approve reviewed image** requires
a reviewer name and confirms review of the entire field. Check:

- image/mask orientation, crop alignment and fluorescence-channel interpretation;
- all eligible target objects, including cells missed by the model;
- false positives, merged/split instances and boundaries;
- the phenotype of every instance using an agreed laboratory rubric;
- truncated edge objects and ambiguous phenotypes using one consistent policy.

Define the eligible target population first. Partially labelling that population
teaches YOLO to treat its unlabelled members as background. Do not approve an
image merely because some displayed predictions look plausible. Seek a second
reviewer for ambiguous phenotypes; quarantine unresolved fields.

To label new, unannotated pool images, restart the editor with `--split test`
and select only pool acquisitions. Never promote the reserved biological
validation/test fields. The compiler checks this automatically.

Approval saves reviewer, UTC timestamp, image SHA-256 and label SHA-256 in
`review_approvals.json`. Any later browser save removes approval; any external
image/label change fails the compiler's hash check. A one-time
`*.txt.manual-backup` preserves the initial label; edits are logged in
`manual_annotation_edits.jsonl`. Saving clears YOLO caches.

Browser API clients must GET `/api/labels/<png>` to obtain `labels`,
`label_sha256`, `image_sha256`, `editable` and review state. POST those expected
hashes with `labels` when saving, or with `reviewer`/`notes` when approving or
deferring. A list-only POST is intentionally rejected; revision checks cannot
be bypassed accidentally.

## 5. Compile a new training dataset

```bash
python annotation_cycle.py compile \
  --dataset data/dataset_info/cycle-v1.json \
  --reviewed-dataset data/dataset_info/cycle-v1-review.json \
  --name cycle-v2 \
  --output-dir data/yolo_datasets/cycle-v2 \
  --dataset-info data/dataset_info/cycle-v2.json

python annotation_cycle.py qc --dataset data/dataset_info/cycle-v2.json \
  --output validation/cycle-v2-qc.json

python create_label_overlays.py --dataset data/dataset_info/cycle-v2.json \
  --output-dir validation/cycle-v2-label-overlays
```

The compiler imports only explicitly approved, hash-matched, non-empty reviews.
It verifies exact image bytes and raw-record provenance; it cannot pair a label
with a similar-looking image. Approved corrections replace that training
image's label. Approved pool images move into training. The compiler rejects
any acquisition reserved for validation or annotated testing, including sibling
frames, and retains their labels and images unchanged.

Existing unreviewed labels in the base training dataset remain unchanged;
unapproved candidate predictions are never imported. Reviewed-empty masks are
not promoted as biological negatives in this workflow.

The result records a `review_history`, the approved per-image provenance and a
prepared image/label hash inventory. It is passed to the same training,
evaluation, overlay and batch annotation CLIs. Never rebuild this compiled
version directly from raw HDF5: doing so discards manual corrections.

If a later raw build adds more fields, use the same frozen named holdouts and
exclusions, then first compile the prior approved dataset into that new base to
carry forward its reviewed labels. For example:

```bash
python annotation_cycle.py compile \
  --dataset data/dataset_info/new-fields-base.json \
  --reviewed-dataset data/dataset_info/cycle-v2.json \
  --name new-fields-with-prior-reviews \
  --output-dir data/yolo_datasets/new-fields-with-prior-reviews \
  --dataset-info data/dataset_info/new-fields-with-prior-reviews.json
```

Then export candidates, review the new fields, and compile the next iteration.
Keep raw identities, preparation geometry and the fixed holdouts consistent.

## 6. Retrain, evaluate and annotate again

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/cycle-v2.json --annotated-only \
  --notebook-protocol --batch-size 4 --device 0 \
  --output models/cycle-v2.pt

CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/cycle-v2.json --annotated-only --evaluate \
  --eval-split test --model models/cycle-v2.pt \
  --img-size 1024 --batch-size 4 --device 0

CUDA_VISIBLE_DEVICES=0 python annotation_cycle.py export \
  --dataset data/dataset_info/cycle-v2.json --model models/cycle-v2.pt \
  --split all --confidence 0.25 --imgsz 1024 --device 0 \
  --name cycle-v2-review \
  --output-dir data/yolo_datasets/cycle-v2-review \
  --dataset-info data/dataset_info/cycle-v2-review.json
```

Compare both checkpoints on the same untouched labelled test fields, with the
same input size and evaluation settings. Report mask precision, recall,
mAP50/mAP50-95, class coverage, acquisition counts and review effort.
A low-confidence candidate count is not an accuracy metric. Dense fields may
hit YOLO's default 300-detection cap; inspect such fields explicitly.

For measurements and provenance-preserving CSVs:

```bash
CUDA_VISIBLE_DEVICES=0 python batch_predict.py \
  --dataset data/dataset_info/cycle-v2.json --split all \
  --model models/cycle-v2.pt --output_csv predictions/cycle-v2.csv
```

The adjacent `cycle-v2.manifest.csv` records every processed image, including
zero-detection cases. CSV fluorescence statistics use the polygon interior and
BF/GFP/RFP channels. Historical `*_std` columns still contain variance for
compatibility; use their square roots for standard deviation.

Archive the raw manifest, both dataset-info JSONs, review history, reviewer
approvals, model hash, exact commands, effective training configuration,
overlay stack and test metrics together. See
[ANNOTATION_LOOP_TEST_REPORT.md](ANNOTATION_LOOP_TEST_REPORT.md) for the
completed real-image/CUDA/browser mechanical test and its limitations.

# Annotation lifecycle: raw fields to a fresh training iteration

This is the supported end-to-end workflow for turning a folder of microscopy
images and masks into a reviewed, versioned YOLO training set. It keeps raw
scientific files immutable, preserves the notebook-compatible seven-class
phenotype contract, and makes human corrections reproducible.

**File names are part of the dataset API.** Read
[DATASET_NAMING.md](DATASET_NAMING.md) before beginning. A malformed or
ambiguous file name must be fixed before preparation; the loader deliberately
fails instead of guessing which image and HDF5 mask belong together.

## 1. Place raw images and masks in the required layout

Create a unique lowercase `<sample-id>` using letters, digits, and single
hyphens. Put 40× acquisitions only under `40x`; all other magnifications go
under `other`.

```text
data/raw/
  40x/
    images/czi/<sample-id>.czi
    images/tiff/<sample-id>__bf.tiff
    images/tiff/<sample-id>__gfp.tiff       # optional
    images/tiff/<sample-id>__rfp.tiff       # optional
    annotations/<sample-id>__mask.h5        # optional; .hdf5 also accepted
  other/
    images/czi/...
    images/tiff/...
    annotations/...
```

The HDF5 mask belongs to the sample with the exact same ID in the same
magnification folder. An image without a mask is valid for inference, but is
not a supervised example. Do not rename, flip, crop, or overwrite raw CZI,
TIFF, or HDF5 files.

## 2. Inspect the discovered records

CZI conversion needs Fiji/ImageJ, so activate the supported environment for
any command that handles CZI files.

```bash
mamba activate yeast_fusion_segmenter
python - <<'PY'
from image_dataset import MicroscopyImageDataset
dataset = MicroscopyImageDataset('data/raw')
dataset.write_manifest('data/raw/manifest.json')
print(f'{len(dataset.records)} image records discovered')
PY
```

Review `data/raw/manifest.json` and correct missing or wrong
`annotation_path` values by fixing the raw filename/layout, never by manual
pairing. TIFF is converted from BF/GFP/RFP in that RGB order; CZI and TIFF
PNG conversion and crop/pad rules are implemented by `MicroscopyImageDataset`
and `prepare_yolo_data.py`.

## 3. Build a new, versioned prepared YOLO dataset

Choose a new directory for every iteration. The command jointly converts each
image and mask into a 1024-pixel prepared PNG and YOLO segmentation labels,
then saves the durable dataset-info JSON used by training and batch inference.

```bash
python build_dataset.py --name iteration-003 --raw-root data/raw \
  --source-format all --magnification all \
  --val-split 0.15 --test-split 0.15 --random-seed 42 \
  --output-dir data/yolo_datasets/iteration_003 \
  --dataset-info data/dataset_info/iteration_003.json
```

For CZI masks, the builder performs the notebook's boundary-alignment check.
If visual review finds an ambiguous orientation, record an exact override with
`manual_orientation_review_app.py`, then rebuild this *new* dataset with
`--orientation-overrides data/manual_orientation_overrides.json`. The result
has the standard layout:

```text
data/yolo_datasets/iteration_003/
  train/images/*.png   train/labels/*.txt
  val/images/*.png     val/labels/*.txt
  test/images/*.png    test/labels/*.txt
  dataset.yaml
```

## 4. Review individual labels in the browser

Use the local editor on an iteration that has not yet been trained. It edits
only derived YOLO `*.txt` segmentation labels; it never changes raw images or
HDF5 masks.

```bash
python manual_yolo_annotation_app.py \
  --dataset data/dataset_info/iteration_003.json --split train
# Open http://127.0.0.1:8767 in a local browser.
```

Select an image and a class (`0: f`, `1: h`, `2: lmcf`, `3: lmsgfp`,
`4: lsgfp`, `5: dip`, `6: d2`). Select **Draw polygon**, click at least three
vertices around the instance, then select **Finish polygon**. To remove an
incorrect instance, select **Remove by click** and click inside its polygon.
Choose **Save labels** to commit that image. Class IDs and their order are
fixed; do not add, rename, merge, or reinterpret classes.

The first save for an edited label creates a nearby
`<image>.txt.manual-backup`. Every save appends image, split, label path,
instance count, and UTC timestamp to
`data/yolo_datasets/iteration_003/manual_annotation_edits.jsonl`. Keep these
files with the dataset version. The editor is localhost-only by default; do
not expose it on an untrusted network.

Repeat for `val` or `test` only when correcting existing ground truth. Avoid
using a test set as a source of model-selection feedback. If a correction is
substantial, preserve the original dataset directory and create a new
iteration before making it, so comparisons remain interpretable.

## 5. Produce an annotated stack for visual quality control

After browser changes, render the exact polygons that YOLO will consume. This
is the required check that image orientation, classes, and hand-drawn contours
are correct.

```bash
python create_label_overlays.py \
  --dataset data/dataset_info/iteration_003.json \
  --output-dir validation/iteration_003_label_overlays
```

Open the PNGs under the `train`, `val`, and `test` subfolders. The generated
`manifest.csv` and `legend.txt` identify images and class colours. Correct
problems in the browser, save, and regenerate this overlay stack until it is
right. Do not treat an unannotated image or an empty mask as negative
biological ground truth.

## 6. Augment the supervised input by adding *reviewed* images

There are two distinct meanings of augmentation:

1. **New manually checked fields:** add their raw CZI/TIFF and paired HDF5
   masks under `data/raw`, inspect the manifest, and rebuild a *new* dataset
   version that contains the prior and new fields. Use browser edits and the
   overlay check above on that new version. This is how the biological training
   set grows.
2. **Online training transforms:** YOLO applies rotations, flips, translation,
   scale, shear, HSV jitter, and mosaic according to `train_yolo.py`. These
   transforms are applied jointly to image and polygon, but they are not new
   biological samples and cannot replace diverse, independently acquired,
   manually checked fields.

Do not copy an edited label into a different raw image or alter a raw HDF5
file to mimic a browser edit. Keep original and revised prepared dataset
directories separate, and retain each dataset-info JSON, override JSON,
manual-edit log, and overlay stack.

## 7. Train a fresh model and evaluate it on labelled held-out data

Train every materially changed dataset from the standard pretrained base
checkpoint with a new output name. Do not continue training a previous
iteration's checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
python train_yolo.py --dataset data/dataset_info/iteration_003.json \
  --annotated-only --notebook-protocol --batch-size 4 --device 0 \
  --output models/iteration_003_yolov8s.pt

CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/iteration_003.json --annotated-only --evaluate \
  --eval-split test --model models/iteration_003_yolov8s.pt \
  --batch-size 4 --device 0
```

Record the build command, split IDs, source exclusions, raw manifest,
orientation overrides, manual-edit log, overlay directory, training command,
checkpoint, and metrics. Interpret held-out results in light of the number of
independent fields and per-class examples; augmentation does not remedy a
missing phenotype or insufficient biological replication.

# Annotation loop verification — 2026-10-11

[Documentation index](README.MD#documentation) ·
[Reproduce tests](tests/README.md) ·
[Supported lifecycle](ANNOTATION_LIFECYCLE.md) ·
[Human-reviewed pilot](ANNOTATION_PILOT.md) ·
[Remaining gaps](ANNOTATION_CYCLE_GAPS.md)

Artifact paths and machine-local storage links are explained in
[STORAGE_LAYOUT.md](STORAGE_LAYOUT.md). All guides are reachable through the
README [documentation index](README.MD#documentation).

## Superseding verification: corrected CZI geometry and hardened review cycle

The current complete mechanical run is archived at
`data/derived/annotation_loop_verified_20261011` on `/mnt/data1`.
`verification.json` records passing raw-source and holdout checks and actual
commands. **The historical 39-instance run described below used incorrect
CZI crop geometry and is not a scientific baseline.**

The discrepancy is now explained: freshly materialized CZI records were
incorrectly constructed with `source_format="tiff"`. This selected upper-left
cropping on fresh builds, while cached CZI records retained `"czi"` and selected
center cropping. The fix preserves the original source format. Regression
tests cover fresh/cached behavior. The corrected held-out PNG is byte-identical
to the existing quarantined center-cropped PNG:
`162dbd6619d0cae352b20a4e97610a18eb7c15eebc20ed392508762c1342f873`.
Its 54-contour label is also byte-identical:
`719957c6e09fcfdb5d46135193bf4b53e4eb42ba42202bd23186126e7af09591`.
Both copies of the source HDF5 hash to
`527b9ba46385b345ce6fc5e4f148f062b77a60d23c02e719b0f776ad30e90e39`.
No source annotation change caused the discrepancy.

Corrected preparation has 5 labelled training fields/162 contours,
1 validation field/84 contours, 1 labelled test field/54 contours, and
1 unannotated inference-pool field. Training contains `f`, `h`, `lmsgfp`,
`lsgfp`, `d2`; `lmcf` and `dip` are absent from this fixture's training split.
All seven IDs remain fixed. This is not a statement about full-cohort coverage.

The repeatable runner exercised real fresh Fiji conversion, two independent
YOLOv8s CUDA 0 runs (1 epoch, 320px, batch 2, workers 0, seed 42, notebook
augmentation), candidate export, Chromium canvas draw/reclassify/remove,
rapid image switching with a delayed response, approval of restored original
HDF5 polygons, compiled-dataset validation, labelled test evaluation,
reannotation and contour stacks. Synthetic UI polygons were removed before
retraining. The reviewer remains explicitly automated, not a biological expert.

| Corrected one-epoch model | Box mAP50 | Box mAP50–95 | Mask mAP50 | Mask mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.003759 | 0.000752 | 0.007158 | 0.000716 |
| v2, restored original labels | 0.003759 | 0.000752 | 0.007158 | 0.000716 |

Identical results are expected from this mechanical test: no real biological
correction was introduced. They do not demonstrate useful segmentation or
review-driven improvement. Full evaluation reports preserve precision/recall,
class support, checkpoint/dataset hashes, versions and effective settings.

Final per-class reports are in `evaluation-report-final/model-v1` and
`evaluation-report-final/model-v2`. The guarded comparison is
`mechanical-comparison-final.md` with an adjacent JSON result. These summarize
the restored-label smoke test, not the pending human-reviewed scientific pilot.

The consolidated `python tests/run_tests.py` suite passes **74 tests**, including
revision conflicts, frozen holdouts, candidate YAML rejection, stale approvals,
failed-copy cleanup, successive correction lineage, exact T<n> selection and
matched-comparison guards. Function-style conversion tests are now discoverable.

Reproduce in the project mamba environment with CUDA 0 and an installed
Playwright CLI (or its wrapper):

```bash
python tests/run_tests.py
python tests/run_annotation_cycle_smoke.py \
  --output-dir /mnt/data1/yeast_fusion_segmenter_datasets/data/derived/loop-new-version \
  --browser-cli /path/to/playwright-cli \
  --browser-config /path/to/optional-local-chromium-config.json
```

Omit `--browser-config` when Playwright's default browser is already installed.
Outputs must be fresh; failed runs retain logs for diagnosis. The run is not
resumed by replacing dataset/model artifacts. Screenshot evidence is
`output/playwright/annotation-loop-annotation_loop_verified_20261011.png`.

The real pilot has a separate **unapproved** review dataset at
`data/dataset_info/annotation-pilot-human-review.json`, with candidates from
the mature seven-class checkpoint at 1024px/confidence 0.25. Scientific
100-epoch training is waiting for [rubric approval](ANNOTATION_REVIEW_RUBRIC.md)
and whole-field human review. See [ANNOTATION_PILOT.md](ANNOTATION_PILOT.md)
and [ANNOTATION_CYCLE_GAPS.md](ANNOTATION_CYCLE_GAPS.md).

## Historical smoke run — superseded geometry

The software loop was exercised with real TIFF/CZI data, actual Fiji
conversion, two fresh CUDA 0 training runs, a real Chromium browser, dataset
compilation, evaluation and reannotation. This is a mechanical integration
test, not evidence that manual correction improves biological accuracy.

## Data and protocol

The isolated fixture copies these exact raw records via
`MicroscopyImageDataset`; production raw files and production datasets are
not changed:

- Four annotated TIFF acquisitions: `mk3-f012` through `mk3-f015`.
- Three annotated CZI acquisitions: `p1-3b9-04` (train), `p1-1g2-09`
  (validation), `p1-1e3-13` (test).
- One unannotated CZI field: `filtro-h-5-03` (inference pool in test).

Preparation produces 5 training PNGs, 1 validation PNG and 2 test PNGs.
Only one test PNG is labelled. The fresh preparation produces **39** test
instances; older repository dataset versions had different counts and must
not be substituted for this test's labels when comparing metrics.

The fixture has 153 labelled training instances across `f`, `h`, `lmsgfp`,
`lsgfp`, and `d2`; `lmcf` and `dip` are absent from this small fixture, though
all seven class IDs remain present in the YAML. This statement does not imply
those phenotypes are absent from the full repository.

Environment: project `yeast_fusion_segmenter` Python 3.9.23, PyTorch
2.8.0+cu128, Ultralytics 8.4.7, NVIDIA RTX PRO 4000 Blackwell on CUDA 0.
Both fresh runs start from `yolov8s-seg.pt`, use `--notebook-protocol`, and
explicitly reduce the budget to **1 epoch, 320px, batch 2, workers 0**.
Online rotation/flip augmentation follows the notebook protocol, with mosaic,
mixup and copy-paste disabled. These budget overrides exercise the pipeline;
they do not reproduce the notebook's converged 1000-epoch model.

## Tested steps

| Step | Result |
| --- | --- |
| Exact raw image/HDF5 pairing | Passed through the canonical loader |
| TIFF channel conversion | Passed with required BF and optional fluorescence channels |
| CZI conversion | Passed through ImageJ/Fiji; no cached production PNG substitution |
| Paired crop/pad and CZI alignment | Passed through canonical preparation |
| Fractional acquisition split | Regression passed: sibling TIFF frames stay together |
| Initial fresh training | Passed on CUDA 0; seven-class best checkpoint written |
| Labelled holdout evaluation | Passed; unannotated CZI pool excluded |
| Model-to-YOLO polygon export | Passed for five training images and later all eight images |
| Browser class selection and drawing | Passed: added and saved a class-6 triangle through actual canvas clicks |
| Browser reclassification | Passed: changed that triangle to class 1 through the UI |
| Browser removal | Passed: clicked inside the triangle and saved an empty label |
| Browser image approval | Passed: reviewer and image/label SHA-256 recorded |
| Later saves invalidate approval | Passed against the final HTTP API |
| Stale YOLO cache invalidation | Passed against the final HTTP API |
| Browser zoom and reload status | Passed: 200% zoom and approval visible after reload |
| Review compilation | Passed: accepted one restored training label into a new version |
| Frozen holdout integrity | Passed: validation/test PNGs and labels byte-identical before/after compilation |
| Carry reviewed labels into later versions | Passed through two further compilation steps |
| Fresh retraining | Passed from the standard base checkpoint, not the initial experiment checkpoint |
| Reannotation | Passed: eight PNG/YOLO polygon pairs exported with the retrained model |
| Mature-model candidate export | Passed at 1024px/confidence 0.25 using the existing quarantined model |
| CSV batch annotation | Passed: 30 accepted detections on the labelled CZI holdout with the mature model |
| CSV channel/interior statistics | Regression passed: BF/GFP/RFP order and polygon interior measurements |
| Zero-detection CSV export | Passed with confidence 1: header-only CSV and an image manifest reporting zero |

The triangle was a synthetic UI test and was removed. Before compilation and
retraining, its image's original HDF5-derived labels were restored. The
approval reviewer was explicitly named
`automated-smoke-test-original-hdf5`. **This does not represent a new human
biological approval**, and no unannotated production image received synthetic
ground truth. Promotion of a manually approved pool image, refusal of a
heldout acquisition, and stale/unapproved-label rejection were exercised in
isolated regression fixtures.

There are 19 passing unittest cases across the review, loader and builder
tests, plus 7 explicitly executed function-style annotation/orientation tests.
The function-style tests must be invoked explicitly if pytest is unavailable;
`python -m unittest` alone does not discover them.

## Performance and QC findings

Both one-epoch models have box and mask mAP50/mAP50-95 of **0** on validation
and the fixed CZI test field. This is expected for this training budget and
provides no accuracy-improvement claim. Zero mAP does not mean zero detected
objects: the retrained model still emitted some high-confidence false/poor
matches. The explicit zero-output test used confidence 1.

Very low confidence (`0.001`) was used only to exercise candidate polygon
export from the short-run checkpoints. It generated many unreliable polygons
and hit the default 300-detection cap on some fields. Use a biologically
validated checkpoint at ordinary confidence for actual review.

The existing mature model exported 59 candidates on `p1-1e3-13` at confidence
0.25, of which the CSV workflow accepted 30 at confidence 0.5. These are
proposal counts, not precision/recall measurements.

QC found a subpixel source-derived contour in `other_mk3-f014.png`, polygon
index 34, with area approximately `1.8e-7` pixels. It is reported for human
review rather than silently removed. Browser serialization now uses 12-digit
precision so editing cannot round such contours into zero-area polygons.

The train CZI contour overlay visually contains numerous cells without labels.
Before scientific retraining, define the eligible target population with the
laboratory and review the **entire field**, including objects missed by the
model. If those unlabelled cells belong to a target class, incomplete masks
teach incorrect background. Software checks alone cannot resolve eligibility
or the phenotype of an ambiguous cell.

## Commands and artifacts

The integration run originally used `/tmp/yeast-annotation-loop-20261011`.
Artifacts were archived onto the data disk and their text metadata paths
rebased. Logs retain the actual historical commands/settings. A temporary
alias still resolves the original paths during this session.

Permanent artifact root:
`data/derived/annotation_loop_smoke_20261011`, also linked at
`validation/annotation_loop_smoke_20261011`.

- `v1.json`, `v2.json`: initial and compiled dataset contracts.
- `model-v1.pt`, `model-v2.pt`: fresh one-epoch checkpoints; run subfolders
  contain `args.yaml`, `results.csv` and training plots.
- `review.json`, `reannotation.json`, `mature-model-preview.json`: candidate
  dataset objects with editable normalized YOLO polygon labels.
- `qc-v1.json`, `qc-v2.json`: class/split coverage and QC details.
- `overlays-v1/`, `overlays-v2/`, `overlays-reannotation/`,
  `overlays-mature-model/`: annotated PNG stacks.
- `review/review_approvals.json`, `review/manual_annotation_edits.jsonl`:
  browser test evidence and hash approvals.
- `eval-v1.log`, `eval-v2.log`, `train-v2.log`: actual CUDA execution results.
- `batch-predictions-final.csv` and its `.manifest.csv`: mature-model CSVs.
- `batch-zero-final.csv` and its `.manifest.csv`: explicit zero-output test.
- `output/playwright/annotation-loop-review.png`: actual browser screenshot.

Representative commands used (the script names and flags are the production
interfaces; the paths and budgets below are isolated test settings):

```bash
PYTHONPATH=. python tests/setup_annotation_cycle_smoke.py /tmp/yeast-annotation-loop-20261011

python build_dataset.py --name loop-v1 \
  --raw-root /tmp/yeast-annotation-loop-20261011/raw \
  --output-dir /tmp/yeast-annotation-loop-20261011/v1 \
  --dataset-info /tmp/yeast-annotation-loop-20261011/v1.json \
  --val-samples p1-1g2-09 --test-samples p1-1e3-13 \
  --val-split 0 --test-split 0

CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset /tmp/yeast-annotation-loop-20261011/v1.json --annotated-only \
  --model yolov8s-seg.pt --notebook-protocol --epochs 1 --batch-size 2 \
  --img-size 320 --workers 0 --device 0 \
  --output /tmp/yeast-annotation-loop-20261011/model-v1.pt

python annotation_cycle.py export \
  --dataset /tmp/yeast-annotation-loop-20261011/v1.json \
  --model /tmp/yeast-annotation-loop-20261011/model-v1.pt \
  --split train --confidence 0.001 --imgsz 320 --device 0 \
  --name loop-review --output-dir /tmp/yeast-annotation-loop-20261011/review \
  --dataset-info /tmp/yeast-annotation-loop-20261011/review.json

python manual_yolo_annotation_app.py \
  --dataset /tmp/yeast-annotation-loop-20261011/review.json --split train
# Actual Chromium canvas add/change-class/remove/save/approve test here.

python annotation_cycle.py compile \
  --dataset /tmp/yeast-annotation-loop-20261011/v1.json \
  --reviewed-dataset /tmp/yeast-annotation-loop-20261011/review.json \
  --name loop-v2 --output-dir /tmp/yeast-annotation-loop-20261011/v2 \
  --dataset-info /tmp/yeast-annotation-loop-20261011/v2.json
```

Retraining repeats the initial command with `v2.json` and `model-v2.pt`,
still starting from `yolov8s-seg.pt`. Both models were evaluated with
`train_yolo.py --annotated-only --evaluate --eval-split test --img-size 320`.
Reannotation repeats export with `v2.json`, `model-v2.pt`, `--split all` and a
new output dataset. Follow [ANNOTATION_LIFECYCLE.md](ANNOTATION_LIFECYCLE.md)
for ordinary scientific-run settings, real human review and repeatable next
iterations.

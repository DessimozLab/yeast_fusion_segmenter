# Seven-class model performance

This report records the reproducible seven-class dataset rebuild and baseline
training on 2026-09-22. CZI preparation includes the notebook's per-file
mask-orientation alignment check. The class mapping exactly follows
`segment_retrain(1).ipynb`: `f`, `h`, `lmcf`, `lmsgfp`, `lsgfp`, `dip`, and
`d2`. Earlier three-class checkpoints collapsed lysis phenotypes and are not
valid for phenotype classification.

## Notebook-compatible TIFF model

The earlier 50-epoch `yolov8n` baselines below are retained as failure
diagnostics, but they are **not** comparable to the final notebook training
run. The notebook uses upper-left TIFF crops (not centre crops),
`yolov8s-seg.pt`, 1,000 requested epochs with early stopping, batch 20, and
the notebook augmentation/hyperparameter schedule. `all_tiff_notebook` is a
fresh build using that exact TIFF geometry and the same seven-class encoding.

| Dataset | Source selection | Images (train / val / test) | Annotated images (train / val / test) | Instances (train / val / test) |
| --- | --- | ---: | ---: | ---: |
| `all_tiff_notebook` | all TIFF image triplets at every magnification | 87 / 2 / 2 | 21 / 2 / 2 | 731 / 120 / 87 |

The remaining 66 TIFF images are retained as unannotated images for inference;
`--annotated-only` excludes them from training and evaluation. The test split
contains `f`, `h`, `lmcf`, and `dip` labels only, so absent phenotype rows are
not evidence of performance on the other three classes.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_tiff_notebook.json --annotated-only \
  --notebook-protocol --device 0 \
  --output models/all_tiff_yolov8s_notebook_1000e.pt
```

The run early-stopped after 352 epochs (best epoch 252). Its best validation
result was mask mAP50 0.602 and mask mAP50-95 0.457 on two labelled images.
On the separate two-image TIFF test split, the frozen best checkpoint obtained:

| Test images / instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| 2 / 87 | 0.646 | 0.555 | 0.646 | 0.465 |

| Test class | Instances | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: |
| `f` | 20 | 0.857 | 0.634 |
| `h` | 56 | 0.850 | 0.616 |
| `lmcf` | 10 | 0.879 | 0.609 |
| `dip` | 1 | 0.000 | 0.000 |

This establishes that the TIFF data and annotation pipeline are valid under
the notebook protocol. It is not a robust biological-performance estimate:
two held-out fields and one diploid instance are far too few for model
selection or phenotype-specific claims.

### CZI-only notebook-protocol rerun

The CZI conversion and mask-orientation path was rebuilt separately and run
with the same notebook protocol. It contained 21 CZI files: five annotated
training fields, one annotated validation field, one annotated test field,
and 14 unannotated inference-only test files. `dip` is absent from the five
training fields.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_czi_notebook.json --annotated-only \
  --notebook-protocol --device 0 \
  --output models/all_czi_yolov8s_notebook_1000e.pt
```

Early stopping selected epoch 45 of 145 completed epochs. Its one-field
validation result was mask mAP50 0.0137 and mask mAP50-95 0.00549. The
separate 54-instance CZI test field gave mask mAP50 0.00132 and mask
mAP50-95 0.000132.

This is a negative training result, not evidence that CZI loading is broken:
the ImageJ conversion was pixel-checked against the prepared images and the
one required HDF5 vertical flip was selected again during the rebuild. Five
annotated CZI fields do not provide enough acquisition, morphology, or
phenotype diversity for this seven-class model to generalize. Do not use this
checkpoint for annotation; add independently annotated CZI fields and use an
experiment-level split before retraining.

### Mixed TIFF+CZI training on the same CZI holdout

To test transfer from the successful TIFF domain without test leakage,
`all_images_czi_holdout` keeps the same CZI test field (`p1-1e3-13`) and CZI
validation field (`p1-3c12-15`) as the CZI-only run. Its training set contains
the remaining five annotated CZI fields plus all 25 annotated TIFF fields
(30 annotated fields total). The test folder also retains 14 unannotated CZI
images for inference, but `--annotated-only` evaluates only the 54-instance
target CZI field.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images_czi_holdout.json --annotated-only \
  --notebook-protocol --device 0 \
  --output models/all_images_czi_holdout_yolov8s_notebook_1000e.pt
```

The mixed run early-stopped after 142 epochs (best epoch 42). Its CZI
validation mask mAP50 was 0.00568. On the unchanged CZI test field it reached
mask mAP50 0.00156 and mask mAP50-95 0.000312, versus 0.00132 and 0.000132
for CZI-only training. This negligible change is not evidence of usable
cross-domain transfer: TIFF training examples improve TIFF evaluation but do
not substitute for independently annotated CZI examples.

## Datasets

| Dataset | Source selection | PNGs (train / val / test) | Annotated PNGs (train / val / test) | Training classes |
| --- | --- | ---: | ---: | --- |
| `40x_only` | 40× CZI | 5 / 1 / 1 | 5 / 1 / 1 | `f`, `h`, `lmcf`, `lmsgfp`, `lsgfp`, `d2` |
| `all_czi` | all CZI | 5 / 1 / 15 | 5 / 1 / 1 | `f`, `h`, `lmcf`, `lmsgfp`, `lsgfp`, `d2` |
| `all_images` | all CZI and TIFF | 92 / 3 / 17 | 26 / 3 / 3 | all seven classes |

An annotated image has a non-empty YOLO label. Empty HDF5 frames and images
without an HDF5 annotation are retained for inference but excluded by
`--annotated-only`. The mixed test folder includes 14 unannotated CZI images
in addition to three annotated test images. The 40× and CZI-only training
sets are the same five annotated 40× images; `dip` is absent from those two
datasets but is present in the `all_images` training labels. Consequently,
`all_images` is the only maintained dataset/model that trains all seven
phenotype classes.

## Training protocol

Each model fine-tuned pretrained `yolov8n-seg.pt` on CUDA device 0 at 1024px,
batch size 4, two workers, for 50 epochs. The CLI used its default
augmentations and hyperparameters, `--annotated-only`, and selected the best
validation checkpoint. The checkpoints were then evaluated with the same CLI
on the annotated test split.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images.json --annotated-only \
  --model yolov8n-seg.pt --epochs 50 --batch-size 4 --img-size 1024 \
  --device 0 --workers 2 --output models/all_images_yolov8n_seg_orientationfix_50e.pt
```

| Model | Dataset | Validation images / instances | Validation mask mAP50-95 | Checkpoint |
| --- | --- | ---: | ---: | --- |
| `40x_only_yolov8n_seg_orientationfix_50e` | `40x_only` | 1 / 12 | 0.0000 | [checkpoint](models/40x_only_yolov8n_seg_orientationfix_50e.pt) |
| `all_czi_yolov8n_seg_orientationfix_50e` | `all_czi` | 1 / 12 | 0.0000 | [checkpoint](models/all_czi_yolov8n_seg_orientationfix_50e.pt) |
| `all_images_yolov8n_seg_orientationfix_50e` | `all_images` | 3 / 115 | 0.00829 | [checkpoint](models/all_images_yolov8n_seg_orientationfix_50e.pt) |

## Annotated held-out test evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images.json --annotated-only --evaluate \
  --eval-split test --model models/all_images_yolov8n_seg_orientationfix_50e.pt \
  --img-size 1024 --batch-size 4 --device 0 --workers 2
```

| Model | Test dataset | Test images / instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `40x_only_yolov8n_seg_orientationfix_50e` | `40x_only` | 1 / 54 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `all_czi_yolov8n_seg_orientationfix_50e` | `all_czi` | 1 / 54 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `all_images_yolov8n_seg_orientationfix_50e` | `all_images` | 3 / 117 | 0.0267 | 0.0176 | 0.0361 | 0.0152 |

## Interpretation

These are smoke-test baselines, not reliable biological-performance estimates.
The 40× and CZI-only evaluations have one image each; the mixed test set has
three. The mixed model detects some `h` instances (mask mAP50 0.143 for that
class), but no model is suitable for scientific use. Add independently
annotated images for every phenotype, freeze a larger test set, and rerun this
protocol before selecting a model.

The directly comparable corrected-model annotations on all seven 40× images
are recorded in [MODEL_COMPARISON_40X.md](MODEL_COMPARISON_40X.md). All three
models produce zero accepted instances at the annotation threshold, so that
comparison is a failure diagnostic rather than a biological agreement study.

Strong augmentation makes a small set useful for a pipeline test, but it does
not replace independent biological/acquisition diversity. The original U-Net
microscopy challenge used 35 images with augmentation; in contrast, broader
benchmarks such as [LIVECell](https://www.nature.com/articles/s41592-021-01249-6)
use 5,239 independently acquired images. A practical next milestone here is
hundreds of independent image fields overall, ample instances of every
phenotype, and an experiment-level held-out test set; augmented images must
never be counted as independent samples.

# Seven-class model performance

This report records the reproducible seven-class dataset rebuild and baseline
training on 2026-09-22. CZI preparation includes the notebook's per-file
mask-orientation alignment check. The class mapping exactly follows
`segment_retrain(1).ipynb`: `f`, `h`, `lmcf`, `lmsgfp`, `lsgfp`, `dip`, and
`d2`. Earlier three-class checkpoints collapsed lysis phenotypes and are not
valid for phenotype classification.

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

Strong augmentation makes a small set useful for a pipeline test, but it does
not replace independent biological/acquisition diversity. The original U-Net
microscopy challenge used 35 images with augmentation; in contrast, broader
benchmarks such as [LIVECell](https://www.nature.com/articles/s41592-021-01249-6)
use 5,239 independently acquired images. A practical next milestone here is
hundreds of independent image fields overall, ample instances of every
phenotype, and an experiment-level held-out test set; augmented images must
never be counted as independent samples.

# Model performance and cross-dataset evaluation

This report records the baseline training and cross-evaluation performed on 2026-09-22. It is intentionally explicit about evaluation limitations: the current annotations are too small for a reliable biological-performance claim.

> **Superseded for phenotype classification.** These checkpoints and metrics
> used an earlier three-class conversion that collapsed the notebook's lysis
> phenotypes. They are retained only as historical segmentation baselines.
> Rebuild the datasets with the current seven-class preparation logic and
> retrain before using a model for phenotype classification.

## Datasets

| Dataset | Source selection | Raw records / PNG frames | Train / val / test PNGs | Labelled test PNGs |
| --- | --- | ---: | ---: | ---: |
| `40x_only` | 40× CZI | 7 / 7 | 5 / 1 / 1 | 1 |
| `all_czi` | all unique CZI | 21 / 21 | 5 / 1 / 15 | 1 |
| `all_images` | all CZI and TIFF | 46 / 128 | 80 / 17 / 31 | 17 |

`all_czi` has 14 unannotated CZI frames. They are retained in its test folder for inference, but they are **not ground truth**. Likewise, 14 of the 31 `all_images` test frames are unannotated. The CZI and 40× training sets contain the same five annotated 40× images after splitting; consequently they are not independent model comparisons.

## Training protocol

All models fine-tuned the pretrained `yolov8n-seg.pt` segmentation architecture using CUDA GPU 0, image size 512, batch size 4, two data-loader workers, and Ultralytics automatic optimizer selection. Each run saves the best checkpoint to `models/` and artifacts under `runs/segment/yolo_training/models/`.

| Model | Dataset | Epochs | Checkpoint | Final train box / seg / class loss | Validation mask mAP50 |
| --- | --- | ---: | --- | --- | ---: |
| `40x_only` | `40x_only` | 10 | [checkpoint](models/40x_only_yolov8n_seg.pt) | 6.231 / 7.695 / 43.427 | 0.000 |
| `all_images` | `all_images` | 5 | [checkpoint](models/all_images_yolov8n_seg.pt) | 6.507 / 7.653 / 44.492 | 0.000 |
| `all_czi` | `all_czi` | 10 | [checkpoint](models/all_czi_yolov8n_seg.pt) | 6.231 / 7.695 / 43.427 | 0.000 |

These are short baseline runs, not converged training jobs. The zero validation mAP means none should be used as a production segmentation model. The immediate next step is to increase independently annotated train/validation/test samples before increasing epochs.

## Cross annotation and held-out test evaluation

Each model annotated every image in the CZI-only and all-images test folders, then was evaluated by Ultralytics against the corresponding YOLO labels at 512 px. Prediction overlays are saved under `runs/segment/evaluation/predictions/`; metric artifacts are under `runs/segment/evaluation/metrics/`. The machine-readable results are [cross_40x_only.json](evaluation/cross_40x_only.json), [cross_all_images.json](evaluation/cross_all_images.json), and [cross_all_czi.json](evaluation/cross_all_czi.json).

| Model | Test dataset | Test images (labelled) | Box mAP50 | Mask mAP50 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `40x_only` | `all_czi` | 15 (1) | 0.000 | 0.000 | 0.000 | 0.000 |
| `40x_only` | `all_images` | 31 (17) | 0.000 | 0.000 | 0.000 | 0.000 |
| `all_images` | `all_czi` | 15 (1) | 0.000 | 0.000 | 0.000 | 0.000 |
| `all_images` | `all_images` | 31 (17) | 0.000 | 0.000 | 0.000 | 0.000 |
| `all_czi` | `all_czi` | 15 (1) | 0.000 | 0.000 | 0.000 | 0.000 |
| `all_czi` | `all_images` | 31 (17) | 0.000 | 0.000 | 0.000 | 0.000 |

## Interpretation

The cross-annotation procedure ran successfully, but the measured results are uniformly zero. This is consistent with the tiny labelled CZI holdout, the short baseline schedules, and a large class loss after training. Do not rank these models on these values. In particular, unannotated frames are represented as empty YOLO labels during validation and can penalize otherwise plausible predictions.

For a meaningful comparison, create a frozen labelled test set with enough independently acquired images from both CZI and TIFF modalities; keep unannotated images out of metric evaluation; then train longer runs with model selection based on mask mAP50-95.

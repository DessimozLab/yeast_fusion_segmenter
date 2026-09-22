# Training from a prepared dataset

Use the `--dataset` argument to train from a saved dataset definition. It is
the preferred interface because it records the exact raw records, HDF5 masks,
magnification/source-format filters, split seed, and YOLO YAML used for the
run.

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

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/all_images.json \
  --model yolov8n-seg.pt \
  --epochs 100 --batch-size 8 --img-size 1024 --device 0 \
  --workers 4 --output models/all_images_yolov8n_seg.pt
```

- `--model`: pretrained segmentation checkpoint to fine-tune.
- `--epochs`, `--batch-size`, `--img-size`: training budget and input size.
- `--device 0`: use CUDA GPU 0; use `--device cpu` only when CUDA is absent.
- `--output`: durable copy of the run’s best checkpoint. Training logs and
  artifacts remain in `runs/segment/`.
- `--hyp`: optional YAML override for augmentation and optimizer settings.

## Reproduce the final notebook training protocol

`segment_retrain(1).ipynb` trained `yolov8s-seg.pt` at 1024 px for 1,000
epochs with batch size 20, eight workers, `nbs=32`, 180° rotations, 0.5
vertical/horizontal flips, and **no** mosaic, mixup, or copy-paste. Use
`--notebook-protocol` to select those settings; explicit CLI values override
the protocol defaults for a shorter test run.

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
keeps only image files whose corresponding YOLO label is non-empty. Those
labels are generated from the HDF5 path paired in the dataset object, so this
excludes inference-only images with no annotation.

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

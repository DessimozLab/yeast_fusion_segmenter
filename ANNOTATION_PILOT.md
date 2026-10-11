# Human-reviewed annotation pilot

**Status: awaiting approved biological rubric and genuine human field review.**
Do not use the automatically approved smoke-test review dataset for this pilot.
The selected pilot is an isolated eight-acquisition cohort: four annotated
TIFF fields (`mk3-f012`–`mk3-f015`), three annotated CZI fields, and one
unannotated CZI field. It is a bounded demonstration, not the full mixed cohort
or a generalization benchmark.

The corrected, immutable baseline is
`data/derived/annotation_loop_verified_20261011/v1.json`. It contains five
labelled training fields, validation `p1-1g2-09`, labelled test `p1-1e3-13`,
and pool field `filtro-h-5-03` in the physical test directory. New preparation
uses the correct CZI center crop. The same baseline can support mechanical
tests and the pilot, but use separate scientific models and review outputs.

## 1. Approve rules, export a separate human review copy

Approve [ANNOTATION_REVIEW_RUBRIC.md](ANNOTATION_REVIEW_RUBRIC.md) first.
The existing seven-class model below supplies **proposals only**. It is not
one of the matched fresh baseline/retrained pilot models.

```bash
mamba activate yeast_fusion_segmenter
CUDA_VISIBLE_DEVICES=0 python annotation_cycle.py export \
  --dataset data/derived/annotation_loop_verified_20261011/v1.json \
  --model models/all_images_czi_holdout_yolov8s_quarantined_b4_retry.pt \
  --split all --confidence 0.25 --imgsz 1024 --device 0 \
  --name annotation-pilot-human-review \
  --output-dir data/yolo_datasets/annotation-pilot-human-review \
  --dataset-info data/dataset_info/annotation-pilot-human-review.json

python manual_yolo_annotation_app.py \
  --dataset data/dataset_info/annotation-pilot-human-review.json --split train
# Open http://127.0.0.1:8767; review mk3-f012, mk3-f014 and p1-3b9-04.
# Stop that server, then open the pool split:
python manual_yolo_annotation_app.py \
  --dataset data/dataset_info/annotation-pilot-human-review.json --split test
# Review filtro-h-5-03. Labelled test p1-1e3-13 is read-only.
```

Inspect whole fields, correct missing/false/merged instances and phenotypes,
then save and approve with your real reviewer identity. Use **Defer unresolved
image** and explanatory notes when uncertain. Do not substitute model output
or an automated reviewer for biological review. If any pilot field is deferred,
resolve it or explicitly document a reduced pilot before interpreting results.

## 2. Compile and check

```bash
python annotation_cycle.py compile \
  --dataset data/derived/annotation_loop_verified_20261011/v1.json \
  --reviewed-dataset data/dataset_info/annotation-pilot-human-review.json \
  --name annotation-pilot-v2 \
  --output-dir data/yolo_datasets/annotation-pilot-v2 \
  --dataset-info data/dataset_info/annotation-pilot-v2.json
python annotation_cycle.py qc --dataset data/dataset_info/annotation-pilot-v2.json \
  --output validation/annotation-pilot-v2-qc.json
python create_label_overlays.py --dataset data/dataset_info/annotation-pilot-v2.json \
  --output-dir data/derived/annotation-pilot-v2-overlays
```

The original HDF5 labels on other training fields are retained with legacy
provenance. The compiler promotes only approved pool labels, preserves every
held-out acquisition, and carries review/teacher lineage into the new version.

## 3. Matched fresh training and one frozen-test comparison

Both runs start independently from `yolov8s-seg.pt`, with 100 epochs,
1024px, batch 4, seed 42 and notebook augmentation (mosaic/mixup/copy-paste
off). Select best checkpoints by validation, never test performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/derived/annotation_loop_verified_20261011/v1.json --annotated-only \
  --model yolov8s-seg.pt --notebook-protocol --epochs 100 --img-size 1024 \
  --batch-size 4 --workers 2 --seed 42 --device 0 \
  --run-dir data/derived/annotation-pilot/training \
  --output models/annotation-pilot-baseline.pt

CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/annotation-pilot-v2.json --annotated-only \
  --model yolov8s-seg.pt --notebook-protocol --epochs 100 --img-size 1024 \
  --batch-size 4 --workers 2 --seed 42 --device 0 \
  --run-dir data/derived/annotation-pilot/training \
  --output models/annotation-pilot-reviewed.pt

CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/derived/annotation_loop_verified_20261011/v1.json --evaluate \
  --model models/annotation-pilot-baseline.pt --eval-split test \
  --img-size 1024 --batch-size 4 --workers 2 --seed 42 --device 0 \
  --run-dir data/derived/annotation-pilot/evaluation

CUDA_VISIBLE_DEVICES=0 python train_yolo.py \
  --dataset data/dataset_info/annotation-pilot-v2.json --evaluate \
  --model models/annotation-pilot-reviewed.pt --eval-split test \
  --img-size 1024 --batch-size 4 --workers 2 --seed 42 --device 0 \
  --run-dir data/derived/annotation-pilot/evaluation
```

Evaluation filters unlabelled pool images automatically. Each run writes
`protocol_metrics.json` with aggregate/per-class metrics, dataset/checkpoint
hashes, split inventory, versions and effective settings. Record any optimizer
auto-selection; requested learning rate alone is not the effective protocol.

For each model, repeat `annotation_cycle.py export` on the compiled dataset
with `--split all --confidence 0.25 --imgsz 1024`, a fresh output name and
its respective checkpoint, then use `create_label_overlays.py` on those
candidate manifests. Record side-by-side overlays and proposal counts, but
do not mistake proposal counts for precision or recall.

## Pilot report acceptance

Generate a guarded Markdown/JSON comparison after both evaluations:

```bash
python compare_annotation_models.py \
  --baseline-dataset data/derived/annotation_loop_verified_20261011/v1.json \
  --reviewed-dataset data/dataset_info/annotation-pilot-v2.json \
  --baseline-report data/derived/annotation-pilot/evaluation/annotation-pilot-baseline/protocol_metrics.json \
  --reviewed-report data/derived/annotation-pilot/evaluation/annotation-pilot-reviewed/protocol_metrics.json \
  --baseline-training-report data/derived/annotation-pilot/training/annotation-pilot-baseline/protocol_metrics.json \
  --reviewed-training-report data/derived/annotation-pilot/training/annotation-pilot-reviewed/protocol_metrics.json \
  --output validation/annotation-pilot-comparison.md
```

It rejects changed holdouts, mismatched dataset/checkpoint reports, non-base
initialization and different budgets/augmentation/evaluation settings. Use the
actual run directory printed by the CLI if its name received a numeric suffix.

Report reviewer/rubric version, dataset inventories and class support, actual
commands and effective budgets, best-validation checkpoint selection, box/mask
precision/recall/mAP50/mAP50–95, and per-class support. Verify baseline/v2
validation and labelled-test image/label hashes are identical. Include the
complete review ledger, deferred fields, overlays and lineage.

The fixture has no `lmcf` or `dip` training examples. A correct seven-class
YAML cannot teach absent phenotypes. One CZI test field cannot support broad
generalization; zero or worse improvement is a valid result to report. Expand
to independent experiments and adequate phenotype support for scientific claims.

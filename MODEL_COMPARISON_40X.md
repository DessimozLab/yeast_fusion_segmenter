# Corrected-model comparison on the 40× dataset

This comparison uses the seven prepared 40× CZI PNGs after the notebook's
mask-orientation alignment repair. It compares the corrected 50-epoch
checkpoints, not the earlier `*_7class_50e.pt` checkpoints prepared before
that repair.

| Model | Training data | 40× images | Accepted instances at confidence > 0.5 | Maximum raw confidence | Mean per-image maximum confidence |
| --- | --- | ---: | ---: | ---: | ---: |
| `40x_only_yolov8n_seg_orientationfix_50e` | five annotated 40× images | 7 | 0 | 0.0034 | 0.0030 |
| `all_czi_yolov8n_seg_orientationfix_50e` | same five annotated 40× images | 7 | 0 | 0.0034 | 0.0030 |
| `all_images_yolov8n_seg_orientationfix_50e` | 26 annotated CZI/TIFF images | 7 | 0 | 0.0120 | 0.0073 |

No model produced a candidate at confidence 0.25 either, so there are no
instance masks, phenotype assignments, or spatial overlaps to compare. The
40× and CZI-only results are exactly identical because their annotated
training data, split, initialization, and deterministic training settings are
identical. The mixed model is marginally less uncertain, but still far below a
usable annotation threshold.

## Command used

```bash
CUDA_VISIBLE_DEVICES=0 python batch_predict.py \
  --dataset data/dataset_info/40x_only.json --split all \
  --model models/all_images_yolov8n_seg_orientationfix_50e.pt \
  --output_csv predictions/model_comparison/all_images_model_on_40x.csv
```

`batch_predict.py` accepts instances only above confidence 0.5. The raw-score
audit additionally evaluated all boxes at `conf=0.001`; it found no score
reaching 0.25. Thus lowering the reporting threshold would only create
unvalidated, extremely low-confidence candidates, not a meaningful agreement
comparison.

The correct conclusion is not that the models agree biologically. They agree
only that none has learned a confident detector from the available data. Use
the corrected CZI conversion going forward, but add independently annotated
images for every phenotype before using any of these checkpoints for
annotation.

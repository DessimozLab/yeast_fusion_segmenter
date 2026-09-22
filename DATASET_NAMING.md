# Raw image dataset convention

**File names are part of the dataset API. They must follow this convention exactly; otherwise dataset discovery, conversion, training, and annotation will fail.** Do not rely on alphabetical file ordering to pair images, channels, or masks.

## Layout

```text
data/raw/
  40x/
    images/czi/<sample-id>.czi
    images/tiff/<sample-id>__bf.tiff
    images/tiff/<sample-id>__gfp.tiff          # optional
    images/tiff/<sample-id>__rfp.tiff          # optional
    annotations/<sample-id>__mask.h5           # optional
  other/
    images/czi/...
    images/tiff/...
    annotations/...
data/derived/png/<40x|other>/<sample-id>[__f0000].png
```

`40x` contains only acquisitions collected at 40×. Every other magnification belongs in `other`. Raw images are immutable; generated PNGs go in `data/derived/png`, never alongside raw files.

## File names

- `<sample-id>` uses lowercase letters, digits, and single hyphens only (`fusion-a-001`, not `Fusion A 001`).
- CZI acquisitions are exactly `<sample-id>.czi`.
- TIFF channel sets use exactly `<sample-id>__bf.tiff`, with optional matching `__gfp.tiff` and `__rfp.tiff`. A TIFF sample requires `__bf`.
- An HDF5 annotation is exactly `<sample-id>__mask.h5` (or `.hdf5`) and is stored in the matching magnification’s `annotations/` folder.
- Extensions are lower case. A sample id must be unique within a magnification, even across CZI and TIFF sources.

## Loading and conversion

Use `MicroscopyImageDataset` rather than globbing files:

```python
from image_dataset import MicroscopyImageDataset

dataset = MicroscopyImageDataset("data/raw")
dataset.write_manifest()                  # metadata includes annotation_path
png_records = dataset.materialize_pngs()  # TIFF and CZI -> model-ready PNG
```

CZI conversion uses ImageJ/Fiji through `pyimagej`; run it in the `yeast_fusion_segmenter` mamba environment. TIFF stacks are normalized per channel, combined as BF/GFP/RFP RGB, and converted frame-by-frame. The resulting `ImageRecord` keeps the paired `annotation_path` so training code can locate the correct HDF5 mask without another filename search.

## Migrating the existing source collections

Review the move plan first, then apply it:

```bash
python organize_raw_images.py
python organize_raw_images.py --apply
```

The migration handles the authoritative `40x_final`, `new_images`, `zoomed`, and `Images_mk3` source folders. It intentionally leaves generated datasets and duplicate working copies untouched.

Train directly from canonical data with `python prepare_yolo_data.py --input-dir data/raw --file-format raw --output-dir datasets/new_run`. Annotate canonical data with `python annotate_images.py --input data/raw --format raw ...`. Both commands invoke `MicroscopyImageDataset` first, so PNG conversion and HDF5 metadata pairing are never duplicated in their callers.

## Annotation classes

HDF5 pixels encode the seven notebook phenotypes in 1000-wide bins. Dataset
preparation preserves these IDs in YOLO labels: `0=f`, `1=h`, `2=lmcf`,
`3=lmsgfp`, `4=lsgfp`, `5=dip`, and `6=d2`. A pixel value in class `i` obeys
`i*1000 <= value < (i+1)*1000`; zero is background. **Do not collapse or
renumber these classes.** It breaks the trained model's phenotype semantics.

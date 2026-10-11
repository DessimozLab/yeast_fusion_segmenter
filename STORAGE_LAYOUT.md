# Dataset storage and local symlinks

[Documentation index](README.MD#documentation) ·
[Dataset naming](DATASET_NAMING.md) ·
[Preparation](DATASET_PREPARATION.md) ·
[Verification artifacts](ANNOTATION_LOOP_TEST_REPORT.md)

## Current machine

Heavy datasets live under `/mnt/data1/yeast_fusion_segmenter_datasets`.
Original project paths remain symlinks, so existing scripts can still use
`data/raw`, `data/derived`, `data/yolo_datasets` and the legacy collection paths.
Do not replace these links with empty folders or move individual image/mask
files independently.

The relocated roots are `Images_mk3`, `augmented`, `data/derived`, `data/raw`,
`data/yolo_datasets`, `datasets`, `datasets_corrected`,
`datasets_czi_fix_preview`, `images_384_CNN`, `images_CNN`, `images_CNN_clean`,
`new40x`, `new_images`, `new_images_png`, `yeaz`, `zoomed`, and `zoomed_png`.
Their names do not supersede the canonical raw
[naming contract](DATASET_NAMING.md); legacy collections are historical inputs.

## Other users and checkouts

The absolute links and heavy generated datasets are **not distributed through
Git**: another machine need not have `/mnt/data1`. Use local directories or
your own storage links at the same project paths, populate the canonical raw
layout, then rebuild fresh datasets using the [builder](DATASET_BUILDER_CLI.md).
Committed experiment/dataset-info JSON files are provenance snapshots; their
absolute paths may belong to the original machine. Do not expect those snapshots
to train without their referenced files.

Choose artifact paths on the data disk **before** conversion/training.
Version-2 manifests hash raw files, prepared images and labels; training reports
also hash the entire manifest. Mechanically rewriting paths inside an existing
experiment changes that manifest identity and can invalidate comparisons.
The legacy archive helper therefore refuses version-2 datasets.

Before relocating an existing collection, confirm the exact source/destination,
back up metadata and check free space. Move a complete collection without
overwriting an existing destination, verify file inventories/checksums, and only
then create its project-path symlink. Never delete raw files to regain space.
See [preparation](DATASET_PREPARATION.md) for immutable-source requirements.

## Git status on this machine

Some historical small files beneath the relocated roots were tracked before
the move. Git does not traverse directory symlinks as tracked trees, so these
can appear as deleted despite remaining accessible through the project path.
**Do not commit those apparent deletions or replace the tracked trees with
machine-specific absolute links.** They are a local storage-layout consequence,
not intentional removal of source metadata. Stage code, docs, tests, notebooks
and intended dataset-info snapshots explicitly rather than using `git add -A`.

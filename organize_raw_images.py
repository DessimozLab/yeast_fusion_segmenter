#!/usr/bin/env python3
"""Move legacy source files into the canonical raw-data layout.

Run with ``--apply`` only after reviewing the printed plan.  It intentionally
does not touch generated datasets, model outputs, or duplicate working copies.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


def slug(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def move(source: Path, destination: Path, apply: bool) -> None:
    print(f"{source} -> {destination}")
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        shutil.move(str(source), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize legacy raw microscopy images")
    parser.add_argument("--apply", action="store_true", help="perform the moves (default is dry run)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raw = root / "data" / "raw"

    # Authoritative CZI folders: 40x_final and new_images.  ``new40x`` is a
    # duplicate preview set and is deliberately preserved for manual review.
    czi_sources = [(root / "40x_final", "40x"), (root / "new_images", "other"), (root / "zoomed", "other")]
    seen: set[Path] = set()
    for folder, magnification in czi_sources:
        for source in folder.rglob("*.czi"):
            sample = slug("-".join(source.relative_to(folder).with_suffix("").parts))
            destination = raw / magnification / "images" / "czi" / f"{sample}.czi"
            move(source, destination, args.apply)
            seen.add(source.resolve())
            mask = source.with_name(source.stem + "_mask.h5")
            if mask.exists():
                move(mask, raw / magnification / "annotations" / f"{sample}__mask.h5", args.apply)

    # Images_mk3 is the canonical TIFF collection.  Its old F<number> name is
    # made unambiguous while retaining the original acquisition number.
    tiff_folder = root / "Images_mk3"
    for source in tiff_folder.glob("F*_*im.[Tt][Ii][Ff]"):
        match = re.match(r"F(\d+)_(GFP|RFP)?_?im\.[Tt][Ii][Ff]$", source.name)
        if not match:
            continue
        number, channel = match.groups()
        channel = (channel or "BF").lower()
        sample = f"mk3-f{int(number):03d}"
        move(source, raw / "other" / "images" / "tiff" / f"{sample}__{channel}.tiff", args.apply)
        if channel == "bf":
            mask = source.with_name(f"F{number}_mask.h5")
            if mask.exists():
                move(mask, raw / "other" / "annotations" / f"{sample}__mask.h5", args.apply)


if __name__ == "__main__":
    main()

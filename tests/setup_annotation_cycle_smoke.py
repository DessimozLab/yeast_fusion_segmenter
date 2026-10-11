"""Make an isolated real-image raw fixture without modifying canonical sources."""
from pathlib import Path
import shutil
import sys

from image_dataset import MicroscopyImageDataset


def main():
    target = Path(sys.argv[1]).resolve()
    target.mkdir(parents=True, exist_ok=False)
    selected = {'mk3-f012', 'mk3-f013', 'mk3-f014', 'mk3-f015',
                'p1-1e3-13', 'p1-1g2-09', 'p1-3b9-04', 'filtro-h-5-03'}
    dataset = MicroscopyImageDataset('data/raw')
    for record in dataset.records:
        if record.sample_id not in selected:
            continue
        for source in record.sources.values():
            dest = target / 'raw' / source.relative_to(dataset.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        if record.annotation_path:
            dest = target / 'raw' / record.annotation_path.relative_to(dataset.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.annotation_path, dest)
    print(target)


if __name__ == '__main__':
    main()

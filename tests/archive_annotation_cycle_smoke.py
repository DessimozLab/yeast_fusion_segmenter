"""Archive a completed isolated test and rebase its text metadata paths."""
from pathlib import Path
import json
import shutil
import sys


def main():
    source, destination = map(lambda p: Path(p).resolve(), sys.argv[1:])
    if not source.is_dir() or destination.exists():
        raise ValueError('Expected an existing source and a new destination')
    for path in source.rglob('*.json'):
        payload=json.loads(path.read_text())
        if isinstance(payload,dict) and payload.get('schema_version')==2 and 'yolo_data' in payload:
            raise ValueError('Version-2 datasets cannot be mechanically rebased; write new runs directly onto the data disk')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    # Bulk mechanical path rebase only; never touch microscopy bytes or weights.
    for path in destination.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.jsonl','.yaml','.txt','.csv'):
            text=path.read_text()
            changed=text.replace(str(source),str(destination))
            if changed!=text:
                path.write_text(changed)
    source.symlink_to(destination,target_is_directory=True)
    print(destination)


if __name__ == '__main__':
    main()

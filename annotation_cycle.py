#!/usr/bin/env python3
"""Export candidate YOLO polygons and compile approved reviews into a fresh dataset."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import os
import tempfile
from contextlib import contextmanager

import yaml

from image_dataset import MicroscopyImageDataset

CLASSES = ('f', 'h', 'lmcf', 'lmsgfp', 'lsgfp', 'dip', 'd2')
IMAGE_NAME = re.compile(r'^(40x|other)_([a-z0-9]+(?:-[a-z0-9]+)*)(?:__f\d{4})?\.png$')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path, value):
    """Replace one file atomically; interrupted writes never expose partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def read_registry(root):
    path = Path(root) / 'review_approvals.json'
    return json.loads(path.read_text()) if path.exists() else {}


def identity(name):
    match = IMAGE_NAME.fullmatch(name)
    if not match:
        raise ValueError(f'Invalid prepared PNG name: {name}')
    return match.group(1), match.group(2)


def load_prepared(path):
    info = MicroscopyImageDataset.load_dataset_info(path)
    config_path = Path(info['yolo_data'])
    config = yaml.safe_load(config_path.read_text())
    names = config['names']
    names = [names.get(i, names.get(str(i))) for i in range(len(names))] if isinstance(names, dict) else names
    if tuple(names) != CLASSES:
        raise ValueError('Dataset must preserve the fixed seven-class map')
    root = Path(config.get('path', config_path.parent))
    if not root.is_absolute():
        root = config_path.parent / root
    return info, root.resolve(), config


def image_entries(info, root):
    records = {(r['magnification'], r['sample_id']): r for r in info['records']}
    if len(records) != len(info['records']):
        raise ValueError('Duplicate raw acquisition identities')
    result = {}
    for split in ('train', 'val', 'test'):
        for image in sorted((root / split / 'images').glob('*.png')):
            key = identity(image.name)
            if key not in records or image.name in result:
                raise ValueError(f'Missing provenance or duplicate image: {image}')
            result[image.name] = (split, image, records[key])
    return result


def prepare_output(output, info_path):
    output, info_path = Path(output).resolve(), Path(info_path).resolve()
    if output.exists() or info_path.exists():
        raise ValueError('Choose a new output directory and dataset-info path for every iteration')
    return output, info_path


def write_info(info, config, output, info_path, name, published_root=None, published_info=None, persist=True):
    public_root = Path(published_root or output).resolve()
    public_info = Path(published_info or info_path).resolve()
    config = dict(config, path=str(public_root), train='train/images', val='val/images', test='test/images',
                  purpose=info.get('purpose','training'), dataset_info=str(public_info))
    info = dict(info, schema_version=2, purpose=info.get('purpose', 'training'),
                name=name, yolo_data=str(public_root / 'dataset.yaml'))
    previous = {row['image']: row for row in info.get('prepared_images', [])}
    approvals = read_registry(output)
    inventory = []
    for image_name, (split, image, record) in image_entries(info, output).items():
        prior = previous.get(image_name, {})
        label = output / split / 'labels' / (image.stem + '.txt')
        approval = approvals.get(f'{split}/{image_name}', {})
        role = prior.get('role')
        if split == 'train':
            role = 'train'
        elif not role or role == 'train':
            role = split if split == 'val' or record.get('annotation_path') or (info['purpose']!='candidate_review' and label.read_text().strip()) else 'pool'
        origin = 'human_reviewed' if approval else ('model_candidate' if info['purpose'] == 'candidate_review'
                 else prior.get('annotation_origin', 'legacy_hdf5' if record.get('annotation_path') else 'unlabelled'))
        inventory.append({'image': image_name, 'split': split, 'role': role,
                          'image_sha256': digest(image), 'label_sha256': digest(label),
                          'raw_identity': list(identity(image_name)), 'annotation_origin': origin,
                          'review_status': 'approved' if approval else 'pending' if origin == 'model_candidate' else 'legacy' if origin == 'legacy_hdf5' else 'unreviewed',
                          'approval': approval or None,
                          'candidate_source': info.get('candidate_source') if info['purpose']=='candidate_review' else prior.get('candidate_source')})
    info['prepared_images'] = inventory
    info['actual_split_counts'] = dict(Counter(row['split'] for row in inventory))
    info['actual_role_counts'] = dict(Counter(row['role'] for row in inventory))
    info['dataset_info'] = str(public_info)
    if 'raw_sha256' not in info:
        paths = {p for record in info['records'] for p in list(record['sources'].values()) + [record.get('annotation_path')] if p}
        info['raw_sha256'] = {str(Path(p)): digest(p) for p in sorted(paths) if Path(p).is_file()}
    if persist:
        atomic_text(output / 'dataset.yaml', yaml.safe_dump(config, sort_keys=False))
        atomic_text(info_path, json.dumps(info, indent=2) + '\n')
        atomic_text(output / 'dataset-contract.json', json.dumps(info, indent=2) + '\n')
    return info


def validate_contract(info, root, allow_review=False, require_version=False, check_marker=True):
    """Check frozen inventory and approvals; review drafts may change label bytes."""
    if info.get('schema_version') != 2:
        marker=root/'dataset-contract.json'
        if marker.exists() and json.loads(marker.read_text()).get('schema_version')==2:
            raise ValueError('Legacy manifest points to a version-2 dataset; use its canonical dataset-info JSON')
        if require_version:
            raise ValueError('Legacy manifest: run annotation_cycle.py migrate --acknowledge-legacy first')
        return ['Legacy manifest has no enforced inventory; migrate before review compilation.']
    entries = image_entries(info, root)
    for source, expected in info.get('raw_sha256', {}).items():
        if not Path(source).is_file() or digest(source) != expected:
            raise ValueError(f'Raw source changed or unavailable: {source}')
    inventory = info.get('prepared_images', [])
    indexed = {row['image']: row for row in inventory}
    if len(indexed) != len(inventory) or set(indexed) != set(entries):
        raise ValueError('Prepared inventory differs from dataset files')
    if info.get('purpose') == 'candidate_review' and not allow_review:
        raise ValueError('Candidate predictions are not ground truth. Approve and compile first.')
    approvals = read_registry(root)
    identities = {}
    for name, (split, image, record) in entries.items():
        row = indexed[name]
        if not {'split','raw_identity','image_sha256','label_sha256','role','annotation_origin','review_status'}.issubset(row):
            raise ValueError(f'Incomplete version-2 inventory: {name}')
        label = root / split / 'labels' / (image.stem + '.txt')
        if row['split'] != split or row['raw_identity'] != list(identity(name)):
            raise ValueError(f'Inventory provenance/split changed: {name}')
        if digest(image) != row['image_sha256']:
            raise ValueError(f'Prepared image changed: {name}')
        if not allow_review and digest(label) != row['label_sha256']:
            raise ValueError(f'Label inventory changed: {name}')
        if row['role'] not in ('train', 'val', 'test', 'pool') or (row['role'] == 'pool' and split != 'test') or (row['role'] != 'pool' and row['role'] != split):
            raise ValueError(f'Invalid split role: {name}')
        identities.setdefault(identity(name), set()).add(split)
        if not allow_review and row['annotation_origin'] == 'human_reviewed':
            state = approvals.get(f'{split}/{name}', {})
            if (state.get('status', 'approved') != 'approved' or not state.get('reviewer', '').strip()
                    or state.get('label_sha256') != digest(label) or state.get('image_sha256') != digest(image)
                    or state != row.get('approval')):
                raise ValueError(f'Stale approval: {name}')
        if not allow_review and row['annotation_origin'] == 'model_candidate':
            raise ValueError(f'Unapproved model candidate: {name}')
        if not allow_review and label.read_text().strip() and not record.get('annotation_path') and row['annotation_origin'] != 'human_reviewed':
            raise ValueError(f'Unverified label without HDF5 provenance: {name}')
    if any(len(splits) > 1 for splits in identities.values()):
        raise ValueError('Acquisition leakage, including empty sibling frames')
    if not allow_review:
        valid_keys = {f'{split}/{name}' for name, (split, _, _) in entries.items()}
        if set(approvals) - valid_keys:
            raise ValueError('Dangling review registry entries')
    marker = root / 'dataset-contract.json'
    if check_marker and marker.exists() and json.loads(marker.read_text()) != info:
        raise ValueError('Dataset-info differs from adjacent dataset contract')
    return []


@contextmanager
def staged_output(output, info_path):
    """Publish only validated, complete datasets; remove only our owned staging tree."""
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=str(output.parent)))
    published = False
    try:
        yield stage
        if output.exists() or info_path.exists():
            raise ValueError('Output appeared during compilation; refusing replacement')
        stage.rename(output)
        published = True
        try:
            # Dataset artifacts may be on /mnt/data1 while JSON lives on NVMe;
            # os.replace across devices is not atomic and raises EXDEV.
            atomic_text(info_path, (output / '.publication.json').read_text())
        except Exception:
            output.rename(stage)
            published = False
            raise
        if info_path != output / '.publication.json':
            try:
                (output / '.publication.json').unlink()
            except OSError:
                pass  # Published contract is complete; leftover staging metadata is harmless.
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def migrate_dataset(args):
    if not args.acknowledge_legacy:
        raise ValueError('Migration records current bytes, not biological approval; pass --acknowledge-legacy')
    info, root, config = load_prepared(args.dataset)
    if info.get('schema_version') == 2:
        validate_contract(info, root)
        raise ValueError('Dataset is already version 2; do not refresh hashes to conceal changes')
    if info.get('purpose') == 'candidate_review':
        raise ValueError('Re-export legacy candidate reviews; do not migrate model predictions to ground truth')
    info['migration'] = {'created_utc': now(), 'source_manifest_sha256': digest(args.dataset),
                         'biological_approval': False}
    preview=write_info(info,config,root,Path(args.dataset),info['name'],persist=False)
    validate_contract(preview,root,check_marker=False)
    backup=Path(str(args.dataset)+'.legacy-backup')
    if backup.exists():
        raise ValueError('Legacy backup exists; inspect prior migration instead of overwriting it')
    atomic_text(backup,Path(args.dataset).read_text())
    write_info(info, config, root, Path(args.dataset), info['name'])
    validate_contract(*load_prepared(args.dataset)[:2])
    print('Migrated current bytes. Existing HDF5 labels remain legacy, not human-approved.')


def export_candidates(args):
    from ultralytics import YOLO
    info, root, config = load_prepared(args.dataset)
    validate_contract(info, root, require_version=True)
    output, info_path = prepare_output(args.output_dir, args.dataset_info)
    entries = image_entries(info, root)
    selected = {n: item for n, item in entries.items() if args.split == 'all' or item[0] == args.split}
    if not selected:
        raise ValueError('No images selected')
    model = YOLO(args.model)
    if model.task != 'segment' or tuple(model.names[i] for i in range(len(model.names))) != CLASSES:
        raise ValueError('Use a seven-class segmentation checkpoint; historical/COCO models are invalid')
    info = dict(info, purpose='candidate_review', candidate_source={
        'dataset_info': str(Path(args.dataset).resolve()), 'model': str(Path(args.model).resolve()),
        'model_sha256': digest(args.model), 'confidence': args.confidence,
        'imgsz': args.imgsz, 'created_utc': now(),
    })
    # A subset inherits raw metadata but not the full base's prepared inventory.
    with staged_output(output, info_path) as stage:
        for split in ('train', 'val', 'test'):
            (stage / split / 'images').mkdir(parents=True)
            (stage / split / 'labels').mkdir()
        for name, (split, image, _) in selected.items():
            shutil.copy2(image, stage / split / 'images' / name)
            result = model.predict(str(image), imgsz=args.imgsz, conf=args.confidence,
                                   device=args.device, verbose=False, retina_masks=True)[0]
            lines = []
            if result.masks is not None:
                for class_id, polygon in zip(result.boxes.cls.tolist(), result.masks.xyn):
                    if len(polygon) >= 3:
                        lines.append(f'{int(class_id)} ' + ' '.join(f'{min(1., max(0., float(v))):.12f}' for point in polygon for v in point))
            (stage / split / 'labels' / (image.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        # Prediction exports have no inherited human approval registry.
        write_info(info, config, stage, stage / '.publication.json', args.name,
                   published_root=output, published_info=info_path)
        validate_contract(json.loads((stage / '.publication.json').read_text()), stage, allow_review=True, require_version=True)
    print(f'Exported {len(selected)} candidate images. Review them before supervision: {info_path}')


def compile_reviews(args):
    from manual_yolo_annotation_app import parse_label, validate_labels
    base, root, config = load_prepared(args.dataset)
    reviewed, review_root, _ = load_prepared(args.reviewed_dataset)
    validate_contract(base, root, require_version=True)
    validate_contract(reviewed, review_root, allow_review=reviewed.get('purpose') == 'candidate_review', require_version=True)
    if root==review_root:
        raise ValueError('Review a separate dataset so the base and heldouts remain immutable')
    output, info_path = prepare_output(args.output_dir, args.dataset_info)
    for source_root in (root,review_root):
        if source_root in output.parents or output in source_root.parents:
            raise ValueError('Output must be separate from base/review dataset trees')
    base_entries = image_entries(base, root)
    review_entries = image_entries(reviewed, review_root)
    approvals_path = review_root / 'review_approvals.json'
    approvals = json.loads(approvals_path.read_text()) if approvals_path.exists() else {}
    # Freeze every labelled val/test acquisition, including sibling stack frames.
    protected = {tuple(row['raw_identity']) for row in base['prepared_images'] if row['role'] in ('val', 'test')}
    accepted = []
    for name, (split, image, record) in review_entries.items():
        key = f'{split}/{name}'
        approval = approvals.get(key)
        if not approval:
            continue
        if approval.get('status', 'approved') != 'approved':
            continue
        label = review_root / split / 'labels' / (image.stem + '.txt')
        if digest(image) != approval['image_sha256'] or digest(label) != approval['label_sha256']:
            raise ValueError(f'Stale approval: {key}. Re-review after editing.')
        if not approval.get('reviewer', '').strip():
            raise ValueError(f'Missing reviewer: {key}')
        segments = parse_label(label)
        validate_labels(segments, len(CLASSES))
        if not segments:
            raise ValueError(f'Empty reviewed mask is not eligible for supervision: {key}')
        if identity(name) in protected:
            raise ValueError(f'Held-out acquisition cannot enter training: {key}')
        if name not in base_entries:
            raise ValueError(f'Build a base including new raw records first: {name}')
        if digest(base_entries[name][1]) != digest(image):
            raise ValueError(f'Prepared image changed; rebuild and re-review: {name}')
        # Exact raw record equality prevents borrowed labels from another source.
        if base_entries[name][2] != record:
            raise ValueError(f'Source provenance differs: {name}')
        accepted.append((name, split, image, label, approval))
    if not accepted:
        raise ValueError('No approved, non-empty reviews. Save and approve in the browser first.')
    with staged_output(output, info_path) as stage:
        _compile_output(args, base, reviewed, root, review_root, config, base_entries,
                        accepted, stage, output, info_path)
    print(f'Compiled {len(accepted)} approved labels; heldouts preserved: {info_path}')


def _compile_output(args, base, reviewed, root, review_root, config, base_entries,
                    accepted, output, public_root, info_path):
    # Copy images and labels only, avoiding stale YOLO caches and filtered lists.
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (output / split / kind).mkdir(parents=True)
            for source in sorted((root / split / kind).glob('*.png' if kind == 'images' else '*.txt')):
                shutil.copy2(source, output / split / kind / source.name)
    for name, _, image, label, _ in accepted:
        previous_split = base_entries[name][0]
        if previous_split != 'train':
            (output / previous_split / 'images' / name).unlink()
            (output / previous_split / 'labels' / label.name).unlink()
        shutil.copy2(image, output / 'train' / 'images' / name)
        shutil.copy2(label, output / 'train' / 'labels' / label.name)
    accepted_names={name for name,_,_,_,_ in accepted}
    baseline_rows={row['image']:row for row in base['prepared_images']}
    for name,(split,image,_) in image_entries(base,output).items():
        expected=baseline_rows[name]
        if digest(image)!=expected['image_sha256']:
            raise ValueError(f'Base image changed during compilation: {name}')
        label=output/split/'labels'/(image.stem+'.txt')
        if name not in accepted_names and digest(label)!=expected['label_sha256']:
            raise ValueError(f'Base label changed during compilation: {name}')
    # Keep approvals in the final split namespace so a compiled dataset can
    # carry its checked labels forward when a later raw build adds new fields.
    prior_approvals=root/'review_approvals.json'
    carried=json.loads(prior_approvals.read_text()) if prior_approvals.exists() else {}
    for name, _, _, _, approval in accepted:
        carried[f'train/{name}']=approval
    (output/'review_approvals.json').write_text(json.dumps(carried,indent=2)+'\n')
    with (output/'manual_annotation_edits.jsonl').open('w') as audit:
        for audit_source in (root/'manual_annotation_edits.jsonl',review_root/'manual_annotation_edits.jsonl'):
            if audit_source.exists():
                audit.write(audit_source.read_text())
    history = list(base.get('review_history', []))
    history.append({'created_utc': now(), 'base_dataset': str(Path(args.dataset).resolve()),
                    'review_dataset': str(Path(args.reviewed_dataset).resolve()),
                    'accepted': [{'image': n, 'source_split': s, **a} for n, s, _, _, a in accepted]})
    # Retain teacher lineage per accepted image, including carry-forward cycles.
    source_rows = {row['image']: row for row in reviewed['prepared_images']}
    rows = {row['image']: row for row in base['prepared_images']}
    for name, _, _, _, _ in accepted:
        rows[name] = dict(source_rows[name], role='train', split='train')
    info = dict(base, purpose='training', review_history=history, prepared_images=list(rows.values()))
    info['inherited_review_history'] = list(base.get('inherited_review_history', [])) + reviewed.get('review_history', [])
    result = write_info(info, config, output, output / '.publication.json', args.name,
                        published_root=public_root, published_info=info_path)
    validate_contract(result, output, require_version=True)


def quality_report(args):
    from manual_yolo_annotation_app import parse_label, validate_labels
    from PIL import Image
    info, root, _ = load_prepared(args.dataset)
    report = {'name': info['name'], 'purpose': info.get('purpose', 'training'), 'splits': {}, 'subpixel_polygons': [], 'errors': [], 'warnings': []}
    try:
        report['warnings'].extend(validate_contract(info, root, allow_review=info.get('purpose') == 'candidate_review'))
    except ValueError as error:
        report['errors'].append(str(error))
    identities = {}
    for name, (split, image, _) in image_entries(info, root).items():
        label = root / split / 'labels' / (image.stem + '.txt')
        if not label.exists():
            raise ValueError(f'Missing label: {label}')
        try:
            segments = parse_label(label)
            validate_labels(segments, len(CLASSES))
            with Image.open(image) as source:
                width,height=source.size
                source.verify()
        except (ValueError,OSError) as error:
            report['errors'].append(f'{name}: {error}')
            continue
        for index, segment in enumerate(segments):
            points=segment['points']; x0,y0=points[0]
            area=abs(sum((a[0]-x0)*(b[1]-y0)-(b[0]-x0)*(a[1]-y0) for a,b in zip(points,points[1:]+points[:1])))*width*height/2
            if area<1:
                report['subpixel_polygons'].append({'image':name,'polygon_index':index,'area_pixels':area})
        row = report['splits'].setdefault(split, {'images': 0, 'labelled_images': 0, 'instances': Counter()})
        row['images'] += 1
        row['labelled_images'] += bool(segments)
        row['instances'].update(CLASSES[s['class_id']] for s in segments)
        identities.setdefault(identity(name), set()).add(split)
    report['acquisition_leakage'] = [f'{m}/{s}' for (m, s), splits in identities.items() if len(splits) > 1]
    report['missing_training_classes'] = [c for c in CLASSES if not report['splits'].get('train', {}).get('instances', {}).get(c)]
    if report['acquisition_leakage']:
        report['errors'].append('Acquisition leakage, including empty sibling frames')
    if report['missing_training_classes']:
        report['warnings'].append('Missing training phenotypes: ' + ', '.join(report['missing_training_classes']))
    if report['subpixel_polygons']:
        report['warnings'].append('Subpixel source contours require review; none were removed automatically.')
    registry = read_registry(root)
    report['review_progress'] = dict(Counter(state.get('status', 'approved') for state in registry.values()))
    report['status'] = 'fail' if report['errors'] else 'warn' if report['warnings'] else 'pass'
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['errors']:
        raise ValueError('; '.join(report['errors']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Export model polygons into an isolated review dataset')
    compile_parser = commands.add_parser('compile', help='Compile explicitly approved labels into a fresh training dataset')
    qc = commands.add_parser('qc', help='Validate polygons, class coverage, and acquisition leakage')
    migrate = commands.add_parser('migrate', help='Snapshot a legacy manifest without claiming biological approval')
    migrate.add_argument('--acknowledge-legacy', action='store_true')
    for p in (export, compile_parser, qc, migrate):
        p.add_argument('--dataset', required=True)
    for p in (export, compile_parser):
        p.add_argument('--name', required=True)
        p.add_argument('--output-dir', required=True)
        p.add_argument('--dataset-info', required=True)
    export.add_argument('--model', required=True)
    export.add_argument('--split', choices=('all', 'train', 'val', 'test'), default='all')
    export.add_argument('--confidence', type=float, default=0.25)
    export.add_argument('--imgsz', type=int, default=1024)
    export.add_argument('--device', default='0')
    compile_parser.add_argument('--reviewed-dataset', required=True)
    qc.add_argument('--output')
    args = parser.parse_args()
    {'export': export_candidates, 'compile': compile_reviews, 'qc': quality_report, 'migrate': migrate_dataset}[args.command](args)


if __name__ == '__main__':
    main()

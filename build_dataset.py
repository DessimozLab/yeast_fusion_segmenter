#!/usr/bin/env python3
"""Build a versioned, provenance-preserving YOLO dataset from ``data/raw``.

The output dataset-info JSON is the single input for training, evaluation, and
dataset-based batch annotation.  This wrapper intentionally supports only the
canonical raw CZI/TIFF/HDF5 layout; it delegates conversion and label creation
to ``prepare_yolo_data.py`` so there is exactly one implementation of the
seven-class conversion rules.
"""
import argparse
import re
import sys
from pathlib import Path


_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]*$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', required=True,
                        help='Versioned dataset name, e.g. iteration-003')
    parser.add_argument('--raw-root', default='data/raw',
                        help='Canonical raw-image root')
    parser.add_argument('--output-dir', default=None,
                        help='Prepared YOLO output (default: data/yolo_datasets/<name>)')
    parser.add_argument('--dataset-info', default=None,
                        help='Dataset-info JSON (default: data/dataset_info/<name>.json)')
    parser.add_argument('--magnification', choices=('all', '40x'), default='all')
    parser.add_argument('--source-format', choices=('all', 'czi', 'tiff'), default='all')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--test-split', type=float, default=0.1)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--val-samples', default='',
                        help='Comma-separated annotated sample IDs reserved for validation')
    parser.add_argument('--test-samples', default='',
                        help='Comma-separated annotated sample IDs reserved for test')
    parser.add_argument('--exclude-samples', default='',
                        help='Comma-separated source sample IDs to quarantine')
    parser.add_argument('--orientation-overrides', default=None,
                        help='Human-approved CZI orientation JSON')
    parser.add_argument('--crop-size', type=int, default=1024)
    parser.add_argument('--resume', action='store_true',
                        help='Resume a partial build; never re-split a finalized dataset')
    parser.add_argument('--replace', action='store_true',
                        help='Allow replacement of an existing prepared output dataset')
    return parser.parse_args(argv)


def build_prepare_argv(args):
    """Return the canonical builder invocation for a parsed wrapper command."""
    output_dir = Path(args.output_dir or Path('data/yolo_datasets') / args.name)
    info_path = Path(args.dataset_info or Path('data/dataset_info') / f'{args.name}.json')
    command = [
        'prepare_yolo_data.py', '--input-dir', str(args.raw_root), '--file-format', 'raw',
        '--output-dir', str(output_dir), '--dataset-info', str(info_path),
        '--magnification', args.magnification, '--source-format', args.source_format,
        '--val-split', str(args.val_split), '--test-split', str(args.test_split),
        '--random-seed', str(args.random_seed), '--val-samples', args.val_samples,
        '--test-samples', args.test_samples, '--exclude-samples', args.exclude_samples,
        '--crop-size', str(args.crop_size),
    ]
    if args.orientation_overrides:
        command.extend(('--orientation-overrides', args.orientation_overrides))
    if args.resume:
        command.append('--resume')
    return command, output_dir, info_path


def validate(args, output_dir, info_path):
    if not _NAME_RE.fullmatch(args.name):
        raise ValueError('--name must contain lowercase letters, digits, hyphens, or underscores')
    if not Path(args.raw_root).is_dir():
        raise ValueError(f'Raw root does not exist: {args.raw_root}')
    if args.val_split < 0 or args.test_split < 0 or args.val_split + args.test_split >= 1:
        raise ValueError('--val-split and --test-split must be non-negative and sum to less than 1')
    if args.resume and args.replace:
        raise ValueError('Use either --resume or --replace, not both')
    if output_dir.exists() and not (args.resume or args.replace):
        raise ValueError(
            f'Output dataset already exists: {output_dir}. Choose a new --name, '
            'use --resume for a partial build, or explicitly pass --replace.'
        )
    if info_path.exists() and not (args.resume or args.replace):
        raise ValueError(
            f'Dataset-info file already exists: {info_path}. Choose a new --name or pass --replace.'
        )


def main(argv=None):
    args = parse_args(argv)
    command, output_dir, info_path = build_prepare_argv(args)
    try:
        validate(args, output_dir, info_path)
    except ValueError as error:
        raise SystemExit(f'error: {error}') from error

    # Reuse the sole raw-data implementation. Changing sys.argv keeps all
    # preparation paths (including ImageJ CZI conversion) identical.
    from prepare_yolo_data import main as prepare_main
    old_argv = sys.argv
    try:
        sys.argv = command
        prepare_main()
    finally:
        sys.argv = old_argv

    if not info_path.exists():
        raise SystemExit(f'error: builder completed without dataset-info JSON: {info_path}')
    print(f'Dataset info: {info_path.resolve()}')
    print('Use this file with train_yolo.py --dataset, batch_predict.py --dataset, '
          'and manual_yolo_annotation_app.py --dataset.')


if __name__ == '__main__':
    main()

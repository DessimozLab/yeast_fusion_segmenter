"""Unit tests for the canonical dataset-builder command wrapper."""
import tempfile
import unittest
from pathlib import Path

from build_dataset import build_prepare_argv, parse_args, validate


class TestBuildDatasetCli(unittest.TestCase):
    def test_builds_raw_only_preparation_command_and_info_path(self):
        args = parse_args(['--name', 'iteration-003', '--source-format', 'czi',
                           '--val-split', '0.2', '--test-samples', 'heldout-a'])
        command, output_dir, info_path = build_prepare_argv(args)
        self.assertEqual(command[:6], ['prepare_yolo_data.py', '--input-dir', 'data/raw',
                                       '--file-format', 'raw', '--output-dir'])
        self.assertIn('--dataset-info', command)
        self.assertIn('data/dataset_info/iteration-003.json', command)
        self.assertEqual(output_dir, Path('data/yolo_datasets/iteration-003'))
        self.assertEqual(info_path, Path('data/dataset_info/iteration-003.json'))

    def test_existing_output_requires_explicit_intent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / 'raw'; root.mkdir()
            output = Path(tempdir) / 'output'; output.mkdir()
            args = parse_args(['--name', 'new-dataset', '--raw-root', str(root),
                               '--output-dir', str(output)])
            _, output_dir, info_path = build_prepare_argv(args)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                validate(args, output_dir, info_path)


if __name__ == '__main__':
    unittest.main()

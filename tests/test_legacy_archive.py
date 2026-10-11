"""The legacy path-rebase helper must not invalidate current hashed datasets."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.archive_annotation_cycle_smoke import main


class TestLegacyArchive(unittest.TestCase):
    def test_version_two_refused_before_any_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            (source/'dataset.json').write_text(json.dumps({'schema_version':2,'yolo_data':'dataset.yaml'}))
            destination=root/'destination'
            with patch('sys.argv',['archive',str(source),str(destination)]):
                with self.assertRaisesRegex(ValueError,'Version-2 datasets'):
                    main()
            self.assertTrue(source.is_dir())
            self.assertFalse(source.is_symlink())
            self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()

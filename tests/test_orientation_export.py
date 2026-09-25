from pathlib import Path
import tempfile

import h5py
import numpy as np

from export_corrected_hdf5_masks import export_mask


def test_export_preserves_hdf5_layout_and_flips_pixels():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'source.h5'
        destination = Path(directory) / 'derived' / 'source.h5'
        values = np.arange(12, dtype=np.uint16).reshape(3, 4)
        with h5py.File(source, 'w') as handle:
            handle.attrs['protocol'] = 'annotation-v1'
            group = handle.create_group('FOV0')
            dataset = group.create_dataset('T0', data=values, compression='gzip')
            dataset.attrs['frame'] = 0

        export_mask(source, destination, 'flip_ud')

        with h5py.File(destination, 'r') as handle:
            assert handle.attrs['protocol'] == 'annotation-v1'
            assert handle['FOV0/T0'].dtype == np.uint16
            assert handle['FOV0/T0'].compression == 'gzip'
            assert handle['FOV0/T0'].attrs['frame'] == 0
            assert np.array_equal(handle['FOV0/T0'][...], np.flipud(values))

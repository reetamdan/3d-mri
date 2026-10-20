import pytest
from src.data.loader import load_mri
from src.data.validation import VolumeValidationError


def test_valid_file(nifti, synthetic_volume):
    array, meta = load_mri(nifti)
    assert array.shape == synthetic_volume.shape
    assert meta['voxel_spacing'] == [2., 1., 3.]
    assert meta['orientation'] == 'RAS'


def test_corrupt_file(tmp_path):
    path = tmp_path / 'broken.nii.gz'
    path.write_bytes(b'not a nifti')
    with pytest.raises(VolumeValidationError) as error:
        load_mri(path)
    assert error.value.code == 'unreadable'


def test_missing_file(tmp_path):
    with pytest.raises(VolumeValidationError) as error:
        load_mri(tmp_path / 'missing.nii')
    assert error.value.code == 'missing'


def test_size_limit(nifti):
    with pytest.raises(VolumeValidationError, match='max_voxels'):
        load_mri(nifti, max_voxels=5)

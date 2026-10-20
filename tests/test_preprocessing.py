import numpy as np
import pytest
from src.data.loader import load_mri
from src.data.preprocessing import normalize, center_crop_pad, preprocess_volume


def test_minmax(synthetic_volume):
    x = normalize(synthetic_volume)
    assert x.min() == pytest.approx(-1)
    assert x.max() == pytest.approx(1)
    assert x.dtype == np.float32


def test_constant_normalization():
    assert np.all(normalize(np.ones((3, 3, 3))) == 0)


def test_zscore(synthetic_volume):
    x = normalize(synthetic_volume, 'zscore')
    assert x.mean() == pytest.approx(0, abs=1e-6)
    assert x.std() == pytest.approx(1)


def test_crop_pad_world_coordinates(synthetic_volume):
    out, transform = center_crop_pad(synthetic_volume, (8, 18, 10))
    assert out.shape == (8, 18, 10)
    np.testing.assert_array_equal(out[:, 2:16], synthetic_volume[2:10])
    np.testing.assert_array_equal(transform[:3, 3], [2, -2, 0])


def test_preprocess_shape_and_determinism(nifti, config):
    volume, meta = load_mri(nifti)
    a, info = preprocess_volume(volume, meta, config['preprocessing'])
    b, _ = preprocess_volume(volume, meta, config['preprocessing'])
    assert a.shape == (16, 16, 8)
    assert info['orientation'] == 'RAS'
    np.testing.assert_array_equal(a, b)
    assert a.min() == pytest.approx(-1)
    assert a.max() == pytest.approx(1)


def test_orientation_flip(synthetic_volume):
    meta = {'affine': np.diag([-1., 1., 1., 1.]).tolist(), 'orientation': 'LAS', 'spatial_unit': 'mm'}
    p = dict(target_shape=list(synthetic_volume.shape), crop_size=None, target_spacing=None, normalization='minmax')
    out, info = preprocess_volume(synthetic_volume, meta, p)
    np.testing.assert_allclose(out, normalize(synthetic_volume[::-1]))
    assert info['orientation'] == 'RAS'


def test_unknown_units_rejected(nifti, config):
    volume, meta = load_mri(nifti)
    meta['spatial_unit'] = 'unknown'
    with pytest.raises(ValueError, match='units'):
        preprocess_volume(volume, meta, config['preprocessing'])


def test_resampling_limit(nifti, config):
    volume, meta = load_mri(nifti)
    config['preprocessing']['target_spacing'] = [.001]*3
    with pytest.raises(ValueError, match='max_voxels'):
        preprocess_volume(volume, meta, config['preprocessing'])


def test_resize_preserves_world_endpoints(nifti, config):
    volume, meta = load_mri(nifti)
    p = {**config['preprocessing'], 'crop_size': None, 'target_spacing': None}
    out, info = preprocess_volume(volume, meta, p)
    source_affine = np.array(meta['affine'])
    target_affine = np.array(info['affine'])
    np.testing.assert_allclose(target_affine @ [0, 0, 0, 1], source_affine @ [0, 0, 0, 1])
    np.testing.assert_allclose(target_affine @ [*(np.array(out.shape)-1), 1],
                               source_affine @ [*(np.array(volume.shape)-1), 1])


def test_comparison_matches_physical_grid(nifti, config):
    from src.visualization.mri_viewer import aligned_original
    volume, meta = load_mri(nifti)
    # Keep comparison inside the field of view: repeated interpolation across a
    # padded boundary need not equal direct interpolation of the source.
    p = {**config['preprocessing'], 'crop_size': [18, 12, 10]}
    out, info = preprocess_volume(volume, meta, p)
    aligned = aligned_original(volume, meta['affine'], info, meta['spatial_unit'])
    # The fixture is a spatially linear ramp, so independent interpolations agree.
    np.testing.assert_allclose(normalize(aligned), out, atol=1e-5)

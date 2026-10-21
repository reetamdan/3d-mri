"""Seeded geometry invariants over varying shapes, orientations and units."""
import nibabel as nib
import numpy as np
import pytest
from src.data.preprocessing import center_crop_pad, normalize, preprocess_volume


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('mode', ['constant', 'edge', 'reflect'])
def test_crop_pad_preserves_intensities_and_world_coordinates(seed, mode):
    rng = np.random.default_rng(seed)
    shape = tuple(rng.integers(3, 12, size=3))
    target = tuple(rng.integers(3, 15, size=3))
    volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    affine = np.eye(4)
    affine[:3, :3] = np.diag(rng.uniform(.3, 4., size=3)*rng.choice([-1., 1.], size=3))
    affine[:3, 3] = rng.uniform(-50, 50, size=3)
    result, transform = center_crop_pad(volume, target, mode)
    output_affine = affine@transform
    assert result.shape == target
    checked = 0
    for index in np.ndindex(target):
        point = np.array([*index, 1.])
        source = transform@point
        if all(0 <= source[i] < shape[i] for i in range(3)):
            assert result[index] == volume[tuple(source[:3].astype(int))]
            np.testing.assert_allclose(output_affine@point, affine@source, atol=1e-12)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize('seed', range(6))
def test_canonicalization_matches_nibabel_under_axis_permutations_and_flips(seed):
    rng = np.random.default_rng(seed)
    volume = rng.normal(size=(7, 9, 11)).astype(np.float32)
    affine = np.eye(4)
    affine[:3, :3] = np.eye(3)[:, rng.permutation(3)]@np.diag(rng.choice([-1., 1.], 3)*rng.uniform(.5, 3., 3))
    affine[:3, 3] = rng.uniform(-40, 40, 3)
    expected = nib.as_closest_canonical(nib.Nifti1Image(volume, affine))
    metadata = dict(affine=affine.tolist(), orientation=''.join(nib.aff2axcodes(affine)), spatial_unit='mm')
    config = dict(target_spacing=None, crop_size=None, target_shape=list(expected.shape),
                  normalization='minmax', normalization_range=[-1, 1], max_voxels=67108864)
    result, geometry = preprocess_volume(volume, metadata, config)
    np.testing.assert_array_equal(result, normalize(np.asarray(expected.dataobj)))
    np.testing.assert_allclose(geometry['affine'], expected.affine)
    assert geometry['orientation'] == 'RAS'

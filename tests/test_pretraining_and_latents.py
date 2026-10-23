"""Unlabeled external-volume contracts, tissue alignment and train-only scaling."""
import numpy as np
import pytest
import torch

from src.data.knee_pretraining import prepare_knee_pretraining
from src.utils.logging import write_csv
from src.training.latent_statistics import transform_latents, cache_representation
from src.training.common import MRIDataset, model_config
from src.training.experiments import train_vq
from src.models.medical_features import MedicalFeatureAdapter
from src.training.anatomy import anatomy_consistency


def test_pretraining_import_preserves_volumes_groups_and_no_invented_labels(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'source'; source.mkdir()
    rows = []
    for i, split in enumerate(('train', 'validation')):
        with h5py.File(source/f'{i}.h5', 'w') as file:
            file.create_dataset('reconstruction_rss', data=np.arange(5*8*9).reshape(5, 8, 9)+i)
            file.attrs['acquisition'] = 'CORPD_FBK'
        rows.append(dict(file_path=f'{i}.h5', patient_id=f'p{i}', split=split, acquisition='CORPD_FBK'))
    manifest = tmp_path/'manifest.csv'; write_csv(manifest, rows, list(rows[0]))
    output = tmp_path/'processed'
    result = prepare_knee_pretraining(source, manifest, output, dataset='fastmri', acquisition='CORPD_FBK',
                                     image_key='reconstruction_rss', shape=(8, 8, 8))
    assert result['valid_scans'] == 2
    data = MRIDataset(output)
    assert data[0].shape == (1, 8, 8, 8) and data.rows[0]['labels'] == ''
    rows[1]['patient_id'] = 'p0'; write_csv(manifest, rows, list(rows[0]))
    with pytest.raises(ValueError, match='patient leakage'):
        prepare_knee_pretraining(source, manifest, tmp_path/'bad', dataset='fastmri', acquisition='CORPD_FBK',
                                 image_key='reconstruction_rss', shape=(8, 8, 8))


def test_skm_tissue_channels_align_and_wrong_acquisition_is_not_mixed(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'source'; source.mkdir()
    with h5py.File(source/'case.h5', 'w') as file:
        file.create_dataset('echo1', data=np.arange(8**3).reshape(8, 8, 8))
        tissue = np.zeros((8, 8, 8, 6), dtype=np.uint8); tissue[2:6, 2:6, 2:6, 4] = 1
        file.create_dataset('seg', data=tissue)
    manifest = tmp_path/'manifest.csv'
    write_csv(manifest, [dict(file_path='case.h5', patient_id='subject', split='train', acquisition='qDESS')],
              ['file_path', 'patient_id', 'split', 'acquisition'])
    output = tmp_path/'processed'
    result = prepare_knee_pretraining(source, manifest, output, dataset='skm-tea', acquisition='qDESS', image_key='echo1', shape=(8, 8, 8))
    assert result['valid_scans'] == 1
    row = MRIDataset(output).rows[0]
    masks = np.load(output/row['mask_file']); assert masks.shape == (6, 8, 8, 8)
    np.testing.assert_array_equal(masks[4], tissue[..., 4])
    empty = prepare_knee_pretraining(source, manifest, tmp_path/'filtered', dataset='skm-tea', acquisition='other', image_key='echo1', shape=(8, 8, 8))
    assert empty['valid_scans'] == 0


def test_channel_scaling_inverse_and_legacy_equivalence():
    x = torch.randn(2, 3, 4, 4, 4)
    legacy = dict(mean=2., std=.5)
    torch.testing.assert_close(transform_latents(x, legacy), (x-2)/.5, rtol=0, atol=0)
    manifest = dict(channel_mean=[1., 2., 3.], channel_std=[.5, 1., 2.])
    torch.testing.assert_close(transform_latents(transform_latents(x, manifest), manifest, inverse=True), x)
    with pytest.raises(ValueError):
        transform_latents(x, dict(channel_mean=[0], channel_std=[1]))


def test_generic_latent_cache_is_train_only_and_integrity_checked(labeled_tensors, tmp_path):
    c = model_config(preset='smoke-models.yaml'); c.update(shape=[16]*3, epochs=1, steps_per_epoch=1, gan_weight=0)
    checkpoint = train_vq(labeled_tensors, tmp_path/'vq', c)
    manifest = cache_representation(labeled_tensors, checkpoint, tmp_path/'cache', 'cpu')
    train = torch.stack([torch.load(tmp_path/'cache'/r['file'], weights_only=True) for r in manifest['rows'] if r['split'] == 'train'])
    expected = train.double().movedim(1, 0).flatten(1)
    np.testing.assert_allclose(manifest['channel_mean'], expected.mean(1).numpy())
    np.testing.assert_allclose(manifest['channel_std'], expected.std(1, unbiased=False).numpy(), atol=1e-8)
    assert manifest['normalization_fit'] == 'train_only_per_channel'


def test_frozen_feature_adapter_keeps_input_gradients_and_no_weight_gradients():
    backbone = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool3d(1))
    model = MedicalFeatureAdapter(backbone)
    x = torch.randn(1, 1, 8, 16, 16, requires_grad=True)
    model(x).square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_supplied_anatomy_masks_have_correct_overlap_and_boundary_units():
    first = np.zeros((8, 8, 8), dtype=np.uint8); first[2:6, 2:6, 2:6] = 1
    identical = anatomy_consistency(first, first)
    assert identical['classes'][0]['dice'] == 1 and identical['classes'][0]['surface_distance_95'] == 0
    shifted = np.roll(first, 1, axis=0)
    measured = anatomy_consistency(shifted, first, spacing=(2, 1, 1))
    assert measured['classes'][0]['dice'] == .75
    assert measured['classes'][0]['surface_distance_95'] == 2
    assert measured['boundary_units'] == 'mm'
    empty = anatomy_consistency(np.zeros_like(first), np.zeros_like(first))
    assert empty['classes'][0]['dice'] is None


def test_pretraining_exposure_is_audited_on_same_cohort_validation(labeled_tensors):
    from src.training.protocol import evaluation_context, cohort_records
    from src.training.common import fingerprint
    records = cohort_records(labeled_tensors)
    checkpoint = dict(dataset_fingerprint=fingerprint(labeled_tensors),
                      representation_pretraining_records=[r for r in records if r['split'] == 'validation'])
    with pytest.raises(ValueError, match='overlaps'):
        evaluation_context(labeled_tensors, checkpoint, 'validation')


def test_feature_neighbor_and_patch_descriptors_follow_array_grid():
    from scripts.evaluate_representation_features import cosine_distances, patch_descriptors
    vectors = np.eye(3, dtype=np.float32)
    distance = cosine_distances(vectors, vectors)
    np.testing.assert_allclose(np.diag(distance), 0)
    assert distance[0, 1] == 1
    array = np.zeros((1, 16, 16, 8), dtype=np.float32)
    array[:, 8:, 8:, 4:] = 1
    patches = patch_descriptors(array)
    assert patches.shape == (8, 256)
    assert patches[-1].mean() == 1 and patches[0].mean() == 0

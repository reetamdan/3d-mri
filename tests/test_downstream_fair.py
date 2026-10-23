"""Operating-point, tied-AP and budget/gate contracts on artificial fixtures."""
import numpy as np
import pytest
import torch

from src.training.downstream_fair import (average_precision, select_thresholds, downstream_metrics,
                                          conventional_augmentation, compare_downstream)
from src.training.common import TrainingDataset, fingerprint
from src.utils.integrity import sha256_file
from src.utils.logging import write_json


def test_average_precision_groups_ties_and_thresholds_are_development_only():
    assert average_precision([1, 0, 1], [.8, .8, .3]) == pytest.approx(.5*.5+.5*2/3)
    assert average_precision([1, 0, 1, 0], [.5]*4) == .5
    assert average_precision([0, 0], [.1, .2]) is None
    truth = np.array([[0]*3, [1]*3, [0]*3, [1]*3])
    probabilities = truth*.2+.1
    thresholds = select_thresholds(truth, probabilities)
    measured = downstream_metrics(truth, probabilities, thresholds)
    assert all(m['balanced_accuracy'] == 1 and m['sensitivity'] == 1 and m['auprc'] == 1 for m in measured.values())
    uniform = np.full((4, 3), .7, dtype=np.float32)
    high_thresholds = select_thresholds(truth, uniform)
    assert all(m['sensitivity'] == 0 for m in downstream_metrics(truth, uniform, high_thresholds).values())


def test_torchio_augmentation_is_seeded_finite_and_shape_preserving():
    pytest.importorskip('torchio')
    x = torch.linspace(-1, 1, 8).expand(1, 16, 16, 8).clone()
    torch.manual_seed(42); first = conventional_augmentation(x)
    torch.manual_seed(42); second = conventional_augmentation(x)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == x.shape and torch.isfinite(first).all() and first.min() >= -1 and first.max() <= 1


def test_fair_classifier_initialization_budget_and_development_scope(labeled_tensors, tmp_path):
    pytest.importorskip('torchio')
    result = compare_downstream(labeled_tensors, tmp_path/'comparison', arms=('real_only', 'torchio', 'resample'),
        shape=(16, 16, 16), steps=2, batch_size=2, seeds=(42,), device='cpu', smoke_test=True, bootstrap_samples=100)
    assert result['steps_per_arm'] == 2 and len(result['results']) == 3
    assert len({r['initial_parameter_sha256'] for r in result['results']}) == 1
    assert result['independent_test'] is False and 'development-only' in result['evaluation_role']
    assert all(len(r['thresholds']) == 3 for r in result['results'])
    with pytest.raises(ValueError, match='no silent resize'):
        compare_downstream(labeled_tensors, tmp_path/'bad-shape', shape=(8, 8, 8), steps=1)


def test_unbalanced_synthetic_pool_and_unpassed_generation_gate_are_rejected(labeled_tensors, tmp_path):
    generated = tmp_path/'generated'; generated.mkdir()
    data = TrainingDataset(labeled_tensors, 'train', (16, 16, 16))
    path = generated/'sample.npy'; np.save(path, data[0][0].numpy())
    write_json(generated/'manifest.json', dict(dataset_fingerprint=fingerprint(labeled_tensors),
        generator_train_ids=data.ids, shape=[16]*3, smoke_test=True,
        rows=[dict(file=path.name, sha256=sha256_file(path), conditioning_labels=[0, 0, 0])]))
    with pytest.raises(ValueError, match='every observed joint-label'):
        compare_downstream(labeled_tensors, tmp_path/'imbalanced', arms=('synthetic',), generated=generated,
            shape=[16]*3, steps=1, smoke_test=True)
    with pytest.raises(ValueError, match='passed generation quality gate'):
        compare_downstream(labeled_tensors, tmp_path/'not-qualified', arms=('synthetic',), generated=generated,
            shape=[16]*3, steps=1, smoke_test=False)

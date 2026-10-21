import numpy as np
import pytest
import torch
from src.training.evaluation import multiscale_ssim
from src.training.diagnostics import foreground_metrics
from src.training.downstream import classification_metrics, classification_intervals


def test_ssim_gaussian_single_scale_matches_pytorch_reference():
    reference = pytest.importorskip('pytorch_msssim')
    torch.set_num_threads(2)
    a = np.random.default_rng(9).uniform(0, 2, (16, 16, 16)).astype(np.float32)
    b = .9*a+.05
    expected = reference.ssim(torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None],
                              data_range=2., win_size=3, win_sigma=1.5, nonnegative_ssim=True)
    actual, levels = multiscale_ssim(a, b, max_levels=1)
    assert levels == 1 and actual == pytest.approx(float(expected), abs=2e-5)


def test_five_scale_metric_matches_trusted_reference_for_matching_protocol():
    reference = pytest.importorskip('pytorch_msssim')
    a = np.random.default_rng(14).uniform(0, 2, (48, 48, 48)).astype(np.float32)
    b = .85*a+.1
    weights = np.array([.0448, .2856, .3001, .2363, .1333]); weights /= weights.sum()
    expected = reference.ms_ssim(torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None],
                                 data_range=2., win_size=3, win_sigma=1.5, weights=weights.tolist())
    actual, levels = multiscale_ssim(a, b)
    assert levels == 5 and actual == pytest.approx(float(expected), abs=2e-5)


def test_foreground_metrics_exclude_background_without_claiming_a_mask():
    a = np.full((9, 9, 9), -1., dtype=np.float32)
    a[2:7, 2:7, 2:7] = .5
    b = a.copy(); b[2:7, 2:7, 2:7] = .25
    metrics = foreground_metrics(a, b)
    assert metrics['foreground_mae'] == .25
    assert metrics['foreground_mse'] == .25**2
    assert 'not an anatomical segmentation' in metrics['foreground_method']
    assert foreground_metrics(np.full_like(a, -1), np.full_like(a, -1))['foreground_mse'] is None


def test_classification_thresholds_bootstrap_and_undefined_auc():
    labels = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]])
    probabilities = np.array([[.1]*3, [.9]*3, [.2]*3, [.8]*3])
    metrics = classification_metrics(labels, probabilities)
    assert metrics['acl']['auc'] == metrics['acl']['sensitivity'] == metrics['acl']['specificity'] == 1.
    a = classification_intervals(labels, probabilities, groups=['a', 'b', 'a', 'b'], resamples=100)
    b = classification_intervals(labels, probabilities, groups=['a', 'b', 'a', 'b'], resamples=100)
    assert a == b and a['acl']['auc']['clusters'] == 2
    assert a['acl']['auc']['lower'] == a['acl']['auc']['upper'] == 1.
    assert classification_metrics(np.zeros((2, 3)), np.zeros((2, 3)))['acl']['auc'] is None

"""Independent audit numerics, including tied scores and paired intervals."""
import numpy as np
import pytest

from scripts.verify_outputs import independent_metrics, independent_classification, paired_interval
from src.metrics import reconstruction_metrics
from src.training.downstream import classification_metrics


def test_independent_metrics_match_closed_form_and_production():
    a = np.arange(125, dtype=np.float64).reshape(5, 5, 5)/125
    b = a+.1
    measured = independent_metrics(a, b)
    assert measured['mse'] == pytest.approx(.01)
    assert measured['mae'] == pytest.approx(.1)
    assert measured['psnr'] == pytest.approx(10*np.log10(400))
    for key, value in reconstruction_metrics(a, b).items():
        assert measured[key] == pytest.approx(value)
    assert independent_metrics(a, a)['psnr'] is None


def test_independent_pairwise_auc_handles_ties_and_missing_classes():
    truth = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]])
    probabilities = np.array([[.1, .9, .8], [.9, .1, .8], [.5, .5, .8], [.5, .5, .8]])
    measured = independent_classification(truth, probabilities)
    assert measured['abnormal']['auc'] == pytest.approx(.875)
    assert measured['meniscus']['auc'] is None
    assert measured == classification_metrics(truth, probabilities)


def test_paired_interval_preserves_pairing_and_repeats():
    result = paired_interval([1, 1, 1])
    assert result['mean'] == result['lower'] == result['upper'] == 1
    assert paired_interval([-1, 0, 1]) == paired_interval([-1, 0, 1])
    with pytest.raises(ValueError):
        paired_interval([float('nan')])

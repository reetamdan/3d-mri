import numpy as np
import pytest
from src.data.validation import validate_volume, VolumeValidationError


@pytest.mark.parametrize('array,code', [
    (np.full((3, 3, 3), np.nan), 'nonfinite'),
    (np.full((3, 3, 3), np.inf), 'nonfinite'),
    (np.empty((0, 3, 3)), 'empty'), (np.ones((3, 3)), 'dimensions'),
    (np.zeros((3, 3, 3)), 'zero'), (np.ones((3, 3, 3)), 'constant'),
])
def test_rejected_arrays(array, code):
    with pytest.raises(VolumeValidationError) as error:
        validate_volume(array)
    assert error.value.code == code

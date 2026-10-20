"""Reject malformed scans before spatial operations or export."""
import numpy as np


class VolumeValidationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def validate_volume(volume: np.ndarray) -> None:
    if volume.ndim != 3:
        raise VolumeValidationError('dimensions', 'Expected a 3D volume; 4D series must be separated explicitly')
    if volume.size == 0 or any(s == 0 for s in volume.shape):
        raise VolumeValidationError('empty', 'Volume is empty')
    if not np.isfinite(volume).all():
        raise VolumeValidationError('nonfinite', 'Volume contains NaN or infinite intensities')
    if not np.any(volume):
        raise VolumeValidationError('zero', 'Volume contains only zero intensities')
    if float(volume.min()) == float(volume.max()):
        raise VolumeValidationError('constant', 'Volume has no intensity variation')

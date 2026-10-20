from pathlib import Path
import numpy as np
import nibabel as nib
from .validation import validate_volume, VolumeValidationError


def load_mri(path: str | Path, max_voxels: int = 67108864) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if not path.is_file():
        raise VolumeValidationError('missing', 'File does not exist')
    if not path.name.lower().endswith(('.nii', '.nii.gz')):
        raise VolumeValidationError('format', 'Only NIfTI .nii and .nii.gz are supported')
    try:
        img = nib.load(str(path))
        if len(img.shape) != 3:
            raise VolumeValidationError('dimensions', 'Expected exactly three dimensions')
        if np.prod(img.shape, dtype=np.int64) > max_voxels:
            raise VolumeValidationError('size_limit', 'Volume exceeds configured max_voxels')
        spacing = np.asarray(img.header.get_zooms()[:3])
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise VolumeValidationError('spacing', 'Voxel spacing is invalid')
        affine = img.affine
        if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise VolumeValidationError('affine', 'Spatial affine is invalid or singular')
        volume = img.get_fdata(dtype=np.float32)
    except VolumeValidationError:
        raise
    except Exception as exc:
        raise VolumeValidationError('unreadable', 'File cannot be decoded as NIfTI') from exc
    validate_volume(volume)
    return volume, {
        'file_path': str(path.resolve()), 'volume_shape': list(volume.shape),
        'voxel_spacing': spacing.tolist(), 'orientation': ''.join(nib.aff2axcodes(affine)),
        'data_type': str(img.get_data_dtype()), 'affine': affine.tolist(),
        'spatial_unit': img.header.get_xyzt_units()[0],
        'minimum_intensity': float(volume.min()), 'maximum_intensity': float(volume.max()),
        'mean_intensity': float(volume.mean(dtype=np.float64)),
        'standard_deviation': float(volume.std(dtype=np.float64)),
    }

"""Spatial operations preserve an affine mapping voxel indices to world coordinates.

Canonical array axes are R, A, S (also denoted H, W, D for model input).
Resizing preserves first/last voxel centers; it changes effective spacing.
"""
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from .validation import validate_volume


def normalize(volume: np.ndarray, method: str = 'minmax', value_range=(-1., 1.)) -> np.ndarray:
    x = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(x).all() or not x.size:
        raise ValueError('Normalization requires a nonempty finite array')
    if method == 'minmax':
        lo, hi = value_range
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Normalization range must be finite and increasing')
        low, high = float(x.min()), float(x.max())
        result = np.full_like(x, (lo + hi) / 2) if low == high else (x.astype(np.float64) - low) / (high - low) * (hi - lo) + lo
    elif method == 'zscore':
        std = x.std(dtype=np.float64)
        result = np.zeros_like(x) if std == 0 else (x.astype(np.float64) - x.mean(dtype=np.float64)) / std
    else:
        raise ValueError('Unknown normalization method')
    return np.asarray(result, dtype=np.float32)


def center_crop_pad(volume: np.ndarray, target: tuple, mode='constant') -> tuple[np.ndarray, np.ndarray]:
    target = tuple(int(v) for v in target)
    starts = np.maximum((np.array(volume.shape) - target) // 2, 0)
    slices = tuple(slice(int(s), int(s + t)) for s, t in zip(starts, target))
    cropped = volume[slices]
    missing = np.maximum(np.array(target) - cropped.shape, 0)
    before = missing // 2
    padded = np.pad(cropped, tuple((int(b), int(m-b)) for b, m in zip(before, missing)), mode=mode)
    transform = np.eye(4)
    transform[:3, 3] = starts - before
    return padded, transform


def preprocess_volume(volume: np.ndarray, metadata: dict, config: dict) -> tuple[np.ndarray, dict]:
    validate_volume(volume)
    p = config
    img = nib.as_closest_canonical(nib.Nifti1Image(volume, np.array(metadata['affine'])))
    stages = [{'operation': 'canonical_RAS', 'shape': list(img.shape)}]
    if p.get('target_spacing'):
        unit = metadata.get('spatial_unit', 'unknown')
        if unit not in ('mm', 'meter', 'micron'):
            raise ValueError('Spatial units are unknown; disable spacing resampling or correct source units')
        factor = {'mm': 1., 'meter': 1000., 'micron': .001}[unit]
        affine = img.affine.copy()
        affine[:3, :] *= factor
        img = nib.Nifti1Image(np.asarray(img.dataobj), affine)
        from nibabel.spaces import vox2out_vox
        out_shape, _ = vox2out_vox((img.shape, img.affine), p['target_spacing'])
        if np.prod(out_shape, dtype=np.int64) > p.get('max_voxels', 67108864):
            raise ValueError('Resampling exceeds max_voxels; choose a coarser spacing')
        img = resample_to_output(img, voxel_sizes=p['target_spacing'], order=1)
        stages.append({'operation': 'resample_mm', 'shape': list(img.shape)})
    if p.get('crop_size'):
        data, transform = center_crop_pad(np.asarray(img.dataobj), p['crop_size'], p.get('padding_mode', 'constant'))
        img = nib.Nifti1Image(data, img.affine @ transform)
        stages.append({'operation': 'center_crop_pad', 'shape': list(img.shape)})
    target = tuple(int(v) for v in p['target_shape'])
    if target != img.shape:
        transform = np.eye(4)
        for axis, (old, new) in enumerate(zip(img.shape, target)):
            if old > 1 and new > 1:
                transform[axis, axis] = (old - 1) / (new - 1)
            else:
                transform[axis, axis] = old / new
                transform[axis, 3] = (old - 1) / 2 - transform[axis, axis] * (new - 1) / 2
        img = resample_from_to(img, (target, img.affine @ transform), order=1, mode='nearest')
        stages.append({'operation': 'resize', 'shape': list(img.shape)})
    spatial = np.asarray(img.dataobj, dtype=np.float32)
    validate_volume(spatial)
    result = normalize(spatial, p['normalization'], p.get('normalization_range', [-1, 1]))
    validate_volume(result)
    return result, {
        'original_orientation': metadata['orientation'],
        'orientation': ''.join(nib.aff2axcodes(img.affine)), 'affine': img.affine.tolist(),
        'volume_shape': list(result.shape), 'voxel_spacing': nib.affines.voxel_sizes(img.affine).tolist(),
        'spatial_unit': 'mm' if p.get('target_spacing') else metadata.get('spatial_unit', 'unknown'),
        'minimum_intensity': float(result.min()), 'maximum_intensity': float(result.max()),
        'stages': stages, 'normalization': p['normalization'],
    }

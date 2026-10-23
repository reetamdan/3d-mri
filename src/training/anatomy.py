"""Paired supplied-mask proxies; no anatomy labels are inferred from exam labels."""
from typing import Any
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def anatomy_consistency(prediction: np.ndarray, reference: np.ndarray, spacing=None) -> dict[str, Any]:
    a, b = np.asarray(prediction), np.asarray(reference)
    if a.shape != b.shape or a.ndim not in (3, 4) or not a.size or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Anatomy proxies require matching aligned binary [C,]3D masks')
    if spacing is not None and (len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0):
        raise ValueError('Physical boundary distances need verified positive spacing')
    if a.ndim == 3:
        a, b = a[None], b[None]
    rows = []
    for first, second in zip(a.astype(bool), b.astype(bool)):
        intersection, union = np.sum(first & second), np.sum(first | second)
        size_a, size_b = int(first.sum()), int(second.sum())
        distances = None
        if size_a and size_b:
            surface_a, surface_b = first & ~binary_erosion(first), second & ~binary_erosion(second)
            distances = np.concatenate([distance_transform_edt(~surface_b, sampling=spacing)[surface_a],
                                        distance_transform_edt(~surface_a, sampling=spacing)[surface_b]])
        rows.append(dict(dice=float(2*intersection/(size_a+size_b)) if size_a+size_b else None,
                         iou=float(intersection/union) if union else None,
                         predicted_voxels=size_a, reference_voxels=size_b,
                         volume_ratio=size_a/size_b if size_b else None,
                         predicted_components=int(label(first)[1]), reference_components=int(label(second)[1]),
                         surface_distance_mean=float(distances.mean()) if distances is not None else None,
                         surface_distance_95=float(np.quantile(distances, .95)) if distances is not None else None))
    return dict(classes=rows, boundary_units='mm' if spacing is not None else 'array voxels',
                protocol='Aligned supplied masks/frozen-model predictions; empty classes excluded, not perfect Dice',
                interpretation='Anatomy consistency proxy, not clinical/pathology validity')

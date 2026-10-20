"""Content checks and confined paths for dataset and model artifacts."""
import hashlib
from pathlib import Path


def confined_path(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Artifact path escapes its directory')
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def generated_volume(root, row, shape):
    import numpy as np
    path = confined_path(root, row['file'])
    if not row.get('sha256') or sha256_file(path) != row['sha256']:
        raise ValueError('Generated tensor checksum missing or mismatched; regenerate samples')
    volume = np.load(path, allow_pickle=False)
    if volume.shape != (1, *shape) or not np.issubdtype(volume.dtype, np.floating):
        raise ValueError('Generated tensor shape or dtype is invalid')
    if not np.isfinite(volume).all() or volume.min() < -1.0001 or volume.max() > 1.0001:
        raise ValueError('Generated tensor must be finite and normalized to [-1,1]')
    return volume

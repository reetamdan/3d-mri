"""Metrics require spatially aligned volumes on the same intensity scale."""
import numpy as np
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity


def reconstruction_metrics(reference, candidate, data_range: float = 2.) -> dict:
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or min(a.shape) < 3:
        raise ValueError('Metrics require matching 3D volumes with each dimension >= 3')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(data_range) or data_range <= 0:
        raise ValueError('Metrics require finite inputs and positive data_range')
    mse = float(np.mean((a-b)**2))
    win = min(7, min(a.shape))
    win -= 1 - win % 2
    return {'mse': mse, 'mae': float(np.mean(abs(a-b))),
            'psnr': float('inf') if mse == 0 else float(10*np.log10(data_range**2/mse)),
            'ssim': float(structural_similarity(a, b, data_range=data_range, win_size=win))}


def ms_ssim(reference, candidate, data_range: float = 2.) -> float:
    """3D five-scale MS-SSIM using pytorch-msssim's standard implementation.

    Optional dependency; inputs must satisfy its five-scale spatial size constraints.
    Kept explicit rather than substituting an average of single-scale SSIM scores.
    """
    reconstruction_metrics(reference, candidate, data_range)
    try:
        from pytorch_msssim import ms_ssim as calculate
    except ImportError as exc:
        raise ImportError('Install pytorch-msssim for optional MS-SSIM support') from exc
    import torch
    a = torch.as_tensor(np.asarray(reference).copy(), dtype=torch.float32)[None, None]
    b = torch.as_tensor(np.asarray(candidate).copy(), dtype=torch.float32)[None, None]
    return float(calculate(a, b, data_range=data_range, size_average=True))


def downsample_reconstruction_baseline(volume, data_range=2.) -> tuple[np.ndarray, dict]:
    """Explicit interpolation baseline, not a trained model or a clinical metric."""
    from scipy.ndimage import map_coordinates
    a = np.asarray(volume, dtype=np.float32)
    low = zoom(a, .5, order=1)
    coords = np.meshgrid(*(np.linspace(0, n-1, t) for n, t in zip(low.shape, a.shape)), indexing='ij')
    restored = map_coordinates(low, coords, order=1, mode='nearest')
    return restored, reconstruction_metrics(a, restored, data_range)

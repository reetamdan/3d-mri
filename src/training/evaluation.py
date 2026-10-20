"""Unpaired generation diagnostics; never treat random real/generated pairs as reconstructions."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.utils.logging import write_json
from .common import TrainingDataset, fingerprint
from src.utils.integrity import generated_volume


def multiscale_ssim(a, b, data_range=2., window=3, max_levels=5):
    """3D MS-SSIM with explicitly adaptive scale count for thin volumes.

    Gaussian local statistics, contrast-structure at early scales, full SSIM at
    the final scale. Standard five weights are truncated and renormalized when
    spatial size cannot support five levels. Returns score and actual levels.
    """
    if window < 3 or window % 2 != 1 or not 1 <= max_levels <= 5:
        raise ValueError('Use an odd window >=3 and between one and five scales')
    a = torch.as_tensor(np.asarray(a).copy(), dtype=torch.float32)[None, None]
    b = torch.as_tensor(np.asarray(b).copy(), dtype=torch.float32)[None, None]
    if a.shape != b.shape or a.ndim != 5 or min(a.shape[2:]) < window:
        raise ValueError('MS-SSIM requires matching 3D volumes at least the window size')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or data_range <= 0:
        raise ValueError('Invalid MS-SSIM inputs')
    levels, smallest = 1, min(a.shape[2:])
    while levels < max_levels and smallest//2 >= window:
        levels += 1; smallest //= 2
    weights = torch.tensor([.0448, .2856, .3001, .2363, .1333][:levels])
    weights = weights/weights.sum()
    coordinates = torch.arange(window)-(window-1)/2
    g = torch.exp(-coordinates.square()/(2*1.5**2)); g /= g.sum()
    kernel = torch.einsum('i,j,k->ijk', g, g, g)[None, None]
    values = []
    for level in range(levels):
        mu_a, mu_b = F.conv3d(a, kernel), F.conv3d(b, kernel)
        va = (F.conv3d(a*a, kernel)-mu_a.square()).clamp_min(0)
        vb = (F.conv3d(b*b, kernel)-mu_b.square()).clamp_min(0)
        covariance = F.conv3d(a*b, kernel)-mu_a*mu_b
        cs = (2*covariance+(.03*data_range)**2)/(va+vb+(.03*data_range)**2)
        luminance = (2*mu_a*mu_b+(.01*data_range)**2)/(mu_a.square()+mu_b.square()+(.01*data_range)**2)
        values.append((luminance*cs if level == levels-1 else cs).mean().clamp(0, 1))
        if level < levels-1:
            a, b = F.avg_pool3d(a, 2), F.avg_pool3d(b, 2)
    return float(torch.prod(torch.stack(values)**weights)), levels


def continuity(volume):
    return [float(np.abs(np.diff(volume, axis=i)).mean()) for i in range(3)]


def evaluate_generation(generated_dir, real_root, output, max_samples=16):
    if max_samples < 2:
        raise ValueError('At least two samples must be requested')
    generated_dir = Path(generated_dir)
    manifest = json.loads((generated_dir/'manifest.json').read_text())
    if manifest['dataset_fingerprint'] != fingerprint(real_root):
        raise ValueError('Real dataset does not match generation provenance')
    generated = [generated_volume(generated_dir, r, manifest['shape'])[0] for r in manifest['rows'][:max_samples]]
    if len(generated) < 2:
        raise ValueError('At least two generated volumes are required for diversity analysis')
    real = TrainingDataset(real_root, 'train', manifest['shape'])
    selected = [i for i, row in enumerate(real.rows) if row['scan_id'] in manifest['generator_train_ids']][:max_samples]
    originals = [real[i][0][0].numpy() for i in selected]
    if not originals:
        raise ValueError('No matching generator training scans for comparison')
    def diversity(volumes):
        values = [multiscale_ssim(a, b) for a, b in itertools.combinations(volumes, 2)]
        return {'mean_ms_ssim': float(np.mean([v[0] for v in values])) if values else None,
                'pairs': len(values), 'levels': values[0][1] if values else None, 'window': 3}
    report = dict(smoke_test=manifest['smoke_test'], generated_count=len(generated), real_count=len(originals),
        generated_diversity=diversity(generated), real_training_diversity=diversity(originals),
        generated_adjacent_slice_mae=[continuity(v) for v in generated],
        real_adjacent_slice_mae=[continuity(v) for v in originals],
        nearest_training_mse=[min(float(np.mean((a-b)**2)) for b in originals) for a in generated],
        intensity_ranges=[[float(v.min()), float(v.max())] for v in generated],
        interpretation='Lower pairwise MS-SSIM can indicate diversity but also noise/artifacts. Adjacent-slice MAE is a descriptive check, not proof of anatomical consistency. Nearest-training MSE is only a memorization screen.',
        paired_ssim_psnr='Not reported for unrelated generated/real volumes; use VQ reconstruction evaluation for paired metrics',
        anatomical_review='Pending qualified visual review; no clinical/anatomical validity claim')
    write_json(output, report)
    return report


def segmentation_overlap(prediction, target):
    for mask in (prediction, target):
        values = np.asarray(mask)
        if not values.size or not np.isin(values, [0, 1]).all():
            raise ValueError('Segmentation masks must contain only binary values')
    a, b = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('Segmentation masks must share shape and alignment')
    intersection, total, union = np.sum(a & b), np.sum(a)+np.sum(b), np.sum(a | b)
    return {'dice': float(2*intersection/total) if total else 1., 'iou': float(intersection/union) if union else 1.}

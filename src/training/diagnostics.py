"""Reconstruction/tokenizer diagnostics with explicit non-anatomical scope."""
import numpy as np
import torch
from skimage.metrics import structural_similarity
from src.metrics import reconstruction_metrics


def foreground_metrics(reference, candidate, threshold=-.95):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    reconstruction_metrics(a, b)
    mask = a > threshold
    result = dict(foreground_fraction=float(mask.mean()), foreground_threshold=threshold,
                  foreground_method='Reference intensity threshold on normalized grid; not an anatomical segmentation')
    if not mask.any():
        return {**result, **{f'foreground_{k}': None for k in ('mse', 'mae', 'psnr', 'ssim')}}
    mse = float(np.square(a[mask]-b[mask]).mean())
    win = min(7, min(a.shape)); win -= 1-win % 2
    _, similarity = structural_similarity(a, b, data_range=2., win_size=win, full=True)
    border = win//2
    inner = tuple(slice(border, -border) for _ in range(3))
    valid = mask[inner]
    result.update(foreground_mse=mse, foreground_mae=float(np.abs(a[mask]-b[mask]).mean()),
                  foreground_psnr=float(10*np.log10(4/mse)) if mse else None,
                  foreground_ssim=float(similarity[inner][valid].mean()) if valid.any() else None)
    return result


def artifact_descriptors(volume):
    a = np.asarray(volume, dtype=np.float64)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError('Artifact diagnostics require a finite 3D array')
    return dict(gradient_energy=float(np.mean([np.square(np.diff(a, axis=i)).mean() for i in range(3)])),
                adjacent_slice_mae=[float(np.abs(np.diff(a, axis=i)).mean()) for i in range(3)],
                note='Texture/continuity descriptors are not anatomical-validity measures')


@torch.no_grad()
def tokenizer_diagnostics(model, dataset, device, max_samples=None):
    model.eval()
    counts = torch.zeros(model.quantizer.embedding.num_embeddings, dtype=torch.float64)
    errors, rows = [], []
    for i in range(len(dataset) if max_samples is None else min(len(dataset), max_samples)):
        x, _ = dataset[i]
        z = model.encoder(x[None].to(device))
        q, _, _ = model.quantizer(z)
        indices = model.quantizer.last_indices.detach().cpu()
        counts += torch.bincount(indices.flatten(), minlength=len(counts)).double()
        error = float((z-q).square().mean())
        errors.append(error)
        rows.append(dict(scan_id=dataset.rows[i]['scan_id'], quantization_mse=error))
    if not rows:
        raise ValueError('Tokenizer diagnostics require a nonempty cohort')
    probabilities = counts/counts.sum()
    return dict(samples=len(rows), counts=counts.tolist(), used_codes=int((counts > 0).sum()),
                dead_code_fraction=float((counts == 0).double().mean()),
                aggregate_perplexity=float(torch.exp(-(probabilities*(probabilities+1e-15).log()).sum())),
                quantization_mse=float(np.mean(errors)), rows=rows)

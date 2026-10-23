"""Differentiable full-3D structural objectives; weights calibrated on training data."""
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def gradient_loss(candidate: torch.Tensor, target: torch.Tensor, spacing=(1., 1., 1.)) -> torch.Tensor:
    if (candidate.shape != target.shape or candidate.ndim != 5 or min(candidate.shape[2:]) < 2
            or len(spacing) != 3 or any(not math.isfinite(s) or s <= 0 for s in spacing)):
        raise ValueError('Gradient loss needs matching 5D volumes and positive three-axis spacing')
    return torch.stack([F.l1_loss(torch.diff(candidate.float(), dim=axis+2)/step,
                                  torch.diff(target.float(), dim=axis+2)/step) for axis, step in enumerate(spacing)]).mean()


def ms_ssim_score(candidate: torch.Tensor, target: torch.Tensor, max_levels=4) -> torch.Tensor:
    """Adaptive Gaussian-window 3D MS-SSIM, matching the existing window-3 metric.

    Thin slice grids cannot support the usual five 11-voxel windows. Truncate and
    renormalize standard scale weights, recording this protocol in evaluations.
    """
    if candidate.shape != target.shape or candidate.ndim != 5 or min(candidate.shape[2:]) < 3:
        raise ValueError('MS-SSIM requires matching 5D inputs with spatial dimensions >=3')
    with torch.autocast(device_type=candidate.device.type, enabled=False):
        a, b = candidate.float(), target.float()
        levels, smallest = 1, min(a.shape[2:])
        while levels < max_levels and smallest//2 >= 3:
            levels += 1; smallest //= 2
        weights = a.new_tensor([.0448, .2856, .3001, .2363, .1333][:levels]); weights /= weights.sum()
        coordinates = torch.arange(3, device=a.device, dtype=a.dtype)-1
        g = torch.exp(-coordinates.square()/(2*1.5**2)); g /= g.sum()
        kernel = torch.einsum('i,j,k->ijk', g, g, g)[None, None].repeat(a.shape[1], 1, 1, 1, 1)
        scores = []
        for level in range(levels):
            conv = lambda x: F.conv3d(x, kernel, groups=a.shape[1])
            ma, mb = conv(a), conv(b)
            va, vb = (conv(a*a)-ma.square()).clamp_min(0), (conv(b*b)-mb.square()).clamp_min(0)
            cs = (2*(conv(a*b)-ma*mb)+.06**2)/(va+vb+.06**2)
            luminance = (2*ma*mb+.02**2)/(ma.square()+mb.square()+.02**2)
            scores.append((luminance*cs if level == levels-1 else cs).mean((1, 2, 3, 4)).clamp(1e-6, 1))
            if level < levels-1:
                a, b = F.avg_pool3d(a, 2), F.avg_pool3d(b, 2)
        return torch.prod(torch.stack(scores)**weights[:, None], dim=0).mean()


def calibrate_weights(candidate: torch.Tensor, target: torch.Tensor, fractions=(.25, .25)) -> dict[str, Any]:
    """Freeze output-gradient balancing coefficients using one training batch only."""
    with torch.enable_grad():
        x = candidate.detach().float().requires_grad_(True)
        terms = [F.l1_loss(x, target.float()), 1-ms_ssim_score(x, target), gradient_loss(x, target)]
        norms = [float(torch.autograd.grad(term, x, retain_graph=True)[0].norm()) for term in terms]
    if min(norms) <= 1e-12 or not all(math.isfinite(v) for v in norms):
        raise ValueError('Structural calibration gradients are degenerate; inspect the training-only warmup')
    return dict(ssim=fractions[0]*norms[0]/norms[1], gradient=fractions[1]*norms[0]/norms[2],
                calibration=dict(l1_gradient_norm=norms[0], ssim_gradient_norm=norms[1], gradient_gradient_norm=norms[2],
                                 target_fractions=list(fractions), fit_split='train'))


class ReconstructionObjective(nn.Module):
    def __init__(self, ssim=0., gradient=0., regularizer=1., spacing=(1., 1., 1.), feature_model=None, perceptual=0.):
        super().__init__()
        if any(not math.isfinite(v) or v < 0 for v in (ssim, gradient, regularizer, perceptual)):
            raise ValueError('Loss weights must be finite and nonnegative')
        if perceptual and feature_model is None:
            raise ValueError('Perceptual loss requires an explicit frozen feature model')
        self.ssim_weight, self.gradient_weight, self.regularizer_weight = ssim, gradient, regularizer
        self.spacing, self.perceptual_weight = spacing, perceptual
        self.feature_model = feature_model
        if feature_model is not None:
            feature_model.eval()
            for parameter in feature_model.parameters():
                parameter.requires_grad_(False)

    def forward(self, candidate: torch.Tensor, target: torch.Tensor, regularizer: torch.Tensor):
        l1 = F.l1_loss(candidate.float(), target.float())
        ssim = 1-ms_ssim_score(candidate, target) if self.ssim_weight else l1.new_zeros(())
        gradient = gradient_loss(candidate, target, self.spacing) if self.gradient_weight else l1.new_zeros(())
        perceptual = l1.new_zeros(())
        if self.feature_model is not None and self.perceptual_weight:
            self.feature_model.eval()
            first = self.feature_model(candidate.float()).flatten(1)
            with torch.no_grad():
                second = self.feature_model(target.float()).flatten(1)
            perceptual = F.mse_loss(F.normalize(first, dim=1), F.normalize(second, dim=1))
        loss = l1+self.ssim_weight*ssim+self.gradient_weight*gradient+self.regularizer_weight*regularizer+self.perceptual_weight*perceptual
        return loss, dict(l1=l1, ms_ssim_loss=ssim, gradient_loss=gradient, regularizer=regularizer, perceptual=perceptual)

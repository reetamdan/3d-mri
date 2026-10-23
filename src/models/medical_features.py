"""Explicit frozen medical feature adapter; descriptors are not anatomy/pathology scores."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.integrity import sha256_file


class MedicalFeatureAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Application layout is row,column,slice; the pretrained 3D backbone
        # receives depth,row,column. This is an array convention, not RAS inference.
        y = x.float().permute(0, 1, 4, 2, 3)
        y = F.interpolate(y, size=(32, 64, 64), mode='trilinear', align_corners=False)
        y = (y-y.mean((2, 3, 4), keepdim=True))/y.std((2, 3, 4), unbiased=False, keepdim=True).clamp_min(1e-6)
        return self.backbone(y).flatten(1)


def load_medicalnet(path: str | Path, expected_sha256: str) -> MedicalFeatureAdapter:
    from monai.networks.nets import resnet10
    if sha256_file(path) != expected_sha256:
        raise ValueError('MedicalNet weight checksum mismatch')
    backbone = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, feed_forward=False,
                        shortcut_type='B', bias_downsample=False)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    backbone.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
    return MedicalFeatureAdapter(backbone).eval()

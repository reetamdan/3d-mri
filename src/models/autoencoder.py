"""Compact official MONAI KL model with a small, explicit sampling-grid adapter."""
from math import prod
from typing import Any, cast

import torch
from torch import nn
from .vqvae import VQVAE3D


class KLAutoencoder3D(nn.Module):
    def __init__(self, channels=(8, 16, 32), latent_channels=8, norm_groups=4,
                 strides=((2, 2, 2), (2, 2, 2)), use_checkpoint=False, use_convtranspose=False):
        super().__init__()
        from monai.networks.nets import AutoencoderKL
        from monai.networks.nets.autoencoderkl import AEKLDownsample
        strides = tuple(tuple(int(v) for v in s) for s in strides)
        if (len(strides) != len(channels)-1 or any(len(s) != 3 or any(v not in (1, 2) for v in s) for s in strides)
                or latent_channels < 1):
            raise ValueError('Use one three-axis 1/2 sampling stride per KL resolution transition')
        self.config = dict(family='kl', channels=list(channels), latent_channels=latent_channels,
                           norm_groups=norm_groups, strides=[list(s) for s in strides], use_checkpoint=use_checkpoint,
                           use_convtranspose=use_convtranspose)
        self.compression = tuple(prod(s[i] for s in strides) for i in range(3))
        self.net = AutoencoderKL(spatial_dims=3, in_channels=1, out_channels=1, channels=tuple(channels),
                                 latent_channels=latent_channels, num_res_blocks=1, norm_num_groups=norm_groups,
                                 attention_levels=(False,)*len(channels), with_encoder_nonlocal_attn=False,
                                 with_decoder_nonlocal_attn=False, use_checkpoint=use_checkpoint,
                                 use_convtranspose=use_convtranspose)
        downs = [block for block in self.net.encoder.modules() if isinstance(block, AEKLDownsample)]
        ups = [block for block in self.net.decoder.modules() if isinstance(block, nn.Upsample)]
        deconvolution_blocks = [block for block in self.net.decoder.modules()
                               if isinstance(block, nn.Sequential) and 'deconv' in block._modules]
        if len(downs) != len(strides) or len(deconvolution_blocks if use_convtranspose else ups) != len(strides):
            raise RuntimeError('Unsupported MONAI sampling layout; revalidate the adapter before upgrading MONAI')
        for down, stride in zip(downs, strides):
            # MONAI's stride-2 3x3 convolution uses right-only padding. A stride-1
            # axis needs symmetric padding to retain its length (no lost slice).
            padding = tuple(v for s in reversed(stride) for v in ((0, 1) if s == 2 else (1, 1)))
            down.add_module('pad', nn.ConstantPad3d(cast(tuple[int, int, int, int, int, int], padding), 0.))
            convolution = next(layer for layer in down.conv.modules() if isinstance(layer, nn.Conv3d))
            convolution.stride = cast(tuple[int, int, int], stride)
        for up, stride in zip(ups, reversed(strides)):
            up.scale_factor = tuple(float(v) for v in stride)
        if use_convtranspose:
            for block, stride in zip(deconvolution_blocks, reversed(strides)):
                original = block._modules['deconv']
                if not isinstance(original, nn.ConvTranspose3d):
                    raise RuntimeError('Unsupported MONAI learned upsampling layout')
                if tuple(original.stride) != stride:
                    # Kernel==stride gives exactly gap-free phase expansion and
                    # preserves stride-1 axes without inventing/dropping slices.
                    block.add_module('deconv', nn.ConvTranspose3d(original.in_channels, original.out_channels,
                                      kernel_size=stride, stride=stride, bias=original.bias is not None))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5 or x.shape[1] != 1 or any(n % factor for n, factor in zip(x.shape[2:], self.compression)):
            raise ValueError('KL inputs must be [B,1,row,column,slice], divisible by the configured compression')
        return self.net.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.net.decode(z).tanh()

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encode(x)
        use_sample = self.training if sample is None else sample
        z = mu + torch.randn_like(sigma)*sigma if use_sample else mu
        return self.decode(z), mu, sigma


def build_autoencoder(spec: dict[str, Any]) -> nn.Module:
    options = dict(spec)
    family = options.pop('family', 'vq')
    if family == 'vq':
        return VQVAE3D(**options)
    if family == 'kl':
        return KLAutoencoder3D(**options)
    raise ValueError('Autoencoder family must be vq or kl')


def encode_latent(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, (KLAutoencoder3D, VQVAE3D)):
        return model.encode(x)[0]  # KL posterior mean or VQ quantized features.
    raise TypeError('Unsupported autoencoder')


def decode_latent(model: nn.Module, z: torch.Tensor, project=True) -> torch.Tensor:
    if isinstance(model, VQVAE3D):
        return model.decode(z, quantize=project)
    if isinstance(model, KLAutoencoder3D):
        return model.decode(z)
    raise TypeError('Unsupported autoencoder')


def autoencoder_step(model: nn.Module, x: torch.Tensor, reference_elements: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, VQVAE3D):
        reconstruction, regularizer, _ = model(x)
        return reconstruction, regularizer
    if isinstance(model, KLAutoencoder3D):
        reconstruction, mu, sigma = model(x)
        # The constrained study passes reference_elements=1 for dimension-
        # invariant mean KL. An explicit multiplier is available to callers;
        # this is not the MONAI tutorial's batch-mean spatially summed KL.
        variance = sigma.float().square().clamp_min(1e-12)
        kl = .5*(mu.float().square()+variance-variance.log()-1).mean()*reference_elements
        return reconstruction, kl
    raise TypeError('Unsupported autoencoder')

"""Compact 3D VQ-VAE with optional patch discriminator for VQ-GAN training."""
import torch
from torch import nn
from torch.nn import functional as F


class Residual3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(1, channels), nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1), nn.GroupNorm(1, channels),
            nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    ema_counts: torch.Tensor
    ema_sums: torch.Tensor
    last_indices: torch.Tensor

    def __init__(self, codes=128, channels=16, commitment=.25, ema_decay=None, dead_code_threshold=0.):
        super().__init__()
        if codes < 2 or channels < 1 or commitment < 0 or dead_code_threshold < 0 or (ema_decay is not None and not 0 < ema_decay < 1):
            raise ValueError('Invalid vector quantizer configuration')
        self.embedding = nn.Embedding(codes, channels)
        nn.init.uniform_(self.embedding.weight, -1/codes, 1/codes)
        self.commitment = commitment
        self.ema_decay, self.dead_code_threshold = ema_decay, dead_code_threshold
        if ema_decay is not None:
            self.embedding.requires_grad_(False)
            self.register_buffer('ema_counts', torch.ones(codes))
            self.register_buffer('ema_sums', self.embedding.weight.detach().clone())

    @torch.no_grad()
    def initialize(self, features):
        indices = torch.randperm(len(features), device=features.device)[:self.embedding.num_embeddings]
        if len(indices) != self.embedding.num_embeddings:
            raise ValueError('Not enough training features to initialize the codebook')
        self.embedding.weight.copy_(features[indices])
        if self.ema_decay is not None:
            self.ema_counts.fill_(1.)
            self.ema_sums.copy_(self.embedding.weight)

    def forward(self, z):
        flat = z.movedim(1, -1).reshape(-1, z.shape[1])
        # Chunk distances to bound temporary allocations for 3D grids.
        chunks = []
        e = self.embedding.weight
        for chunk in flat.split(4096):
            distances = chunk.square().sum(1, keepdim=True) + e.square().sum(1) - 2*chunk@e.T
            chunks.append(distances.argmin(1))
        indices = torch.cat(chunks)
        self.last_indices = indices.detach()
        q = self.embedding(indices).view(z.shape[0], *z.shape[2:], z.shape[1]).movedim(-1, 1)
        loss = self.commitment*F.mse_loss(z, q.detach())
        if self.ema_decay is None:
            loss = loss+F.mse_loss(q, z.detach())
        elif self.training:
            with torch.no_grad():
                counts = torch.bincount(indices, minlength=e.shape[0]).to(e.dtype)
                sums = torch.zeros_like(e).index_add_(0, indices, flat.detach().to(e.dtype))
                self.ema_counts.mul_(self.ema_decay).add_(counts, alpha=1-self.ema_decay)
                self.ema_sums.mul_(self.ema_decay).add_(sums, alpha=1-self.ema_decay)
                self.embedding.weight.copy_(self.ema_sums/self.ema_counts.clamp_min(1e-5)[:, None])
                dead = self.ema_counts < self.dead_code_threshold
                if dead.any():
                    replacements = flat.detach()[torch.randint(len(flat), (int(dead.sum()),), device=z.device)]
                    self.embedding.weight[dead] = replacements.to(e.dtype)
                    self.ema_counts[dead] = 1.
                    self.ema_sums[dead] = replacements.to(e.dtype)
        probabilities = torch.bincount(indices, minlength=e.shape[0]).float()/indices.numel()
        perplexity = torch.exp(-(probabilities * (probabilities+1e-10).log()).sum())
        return z + (q-z).detach(), loss, perplexity


class VQVAE3D(nn.Module):
    def __init__(self, base=16, latent_channels=16, codes=128, commitment=.25, ema_decay=None, dead_code_threshold=0.):
        super().__init__()
        self.config = dict(base=base, latent_channels=latent_channels, codes=codes, commitment=commitment,
                           ema_decay=ema_decay, dead_code_threshold=dead_code_threshold)
        self.encoder = nn.Sequential(nn.Conv3d(1, base, 4, 2, 1), nn.SiLU(), Residual3D(base),
            nn.Conv3d(base, base*2, 4, 2, 1), nn.SiLU(), Residual3D(base*2),
            nn.Conv3d(base*2, latent_channels, 1))
        self.quantizer = VectorQuantizer(codes, latent_channels, commitment, ema_decay, dead_code_threshold)
        self.decoder = nn.Sequential(nn.Conv3d(latent_channels, base*2, 3, padding=1), Residual3D(base*2),
            nn.ConvTranspose3d(base*2, base, 4, 2, 1), nn.SiLU(), Residual3D(base),
            nn.ConvTranspose3d(base, 1, 4, 2, 1), nn.Tanh())

    def encode(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(s % 4 for s in x.shape[2:]):
            raise ValueError('VQ input must be [B,1,H,W,D], spatial dimensions divisible by 4')
        return self.quantizer(self.encoder(x))

    def decode(self, latent, quantize=False):
        if quantize:
            latent = self.quantizer(latent)[0]
        return self.decoder(latent)

    def forward(self, x):
        q, loss, perplexity = self.encode(x)
        return self.decode(q), loss, perplexity


class PatchDiscriminator3D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(1, base, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv3d(base, base*2, 4, 2, 1), nn.GroupNorm(1, base*2), nn.LeakyReLU(.2),
            nn.Conv3d(base*2, 1, 3, padding=1))

    def forward(self, x):
        return self.net(x)

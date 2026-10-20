"""Epsilon-prediction DDPM with a time/label-conditioned 3D U-Net."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeBlock(nn.Module):
    def __init__(self, inputs, outputs, embedding):
        super().__init__()
        self.conv1 = nn.Conv3d(inputs, outputs, 3, padding=1)
        self.conv2 = nn.Conv3d(outputs, outputs, 3, padding=1)
        self.norm1, self.norm2 = nn.GroupNorm(1, outputs), nn.GroupNorm(1, outputs)
        self.condition = nn.Linear(embedding, outputs*2)
        self.skip = nn.Conv3d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x, embedding):
        h = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.condition(embedding).chunk(2, 1)
        h = self.norm2(self.conv2(h))*(1+scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return self.skip(x) + F.silu(h)


class SpatialAttention3D(nn.Module):
    """Global spatial/depth interaction, restricted to the U-Net bottleneck."""
    def __init__(self, channels, heads=4):
        super().__init__()
        if heads < 1 or channels % heads:
            raise ValueError('Attention heads must divide bottleneck channels')
        self.heads = heads
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.project = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        batch, channels, *spatial = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(batch, 3, self.heads, channels // self.heads, -1).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        h = h.transpose(-1, -2).reshape(batch, channels, *spatial)
        return x + self.project(h)


class UNet3D(nn.Module):
    def __init__(self, channels=16, base=32, label_dim=0, attention=False, attention_heads=4):
        super().__init__()
        if base < 2:
            raise ValueError('U-Net base width must be at least two')
        self.config = dict(channels=channels, base=base, label_dim=label_dim,
                           attention=attention, attention_heads=attention_heads)
        self.time_dim = base*4
        self.time_mlp = nn.Sequential(nn.Linear(self.time_dim, self.time_dim), nn.SiLU(), nn.Linear(self.time_dim, self.time_dim))
        self.labels = nn.Linear(label_dim, self.time_dim) if label_dim else None
        self.entry = TimeBlock(channels, base, self.time_dim)
        self.down = nn.Conv3d(base, base*2, 4, 2, 1)
        self.middle = TimeBlock(base*2, base*2, self.time_dim)
        self.attention = SpatialAttention3D(base*2, attention_heads) if attention else nn.Identity()
        self.up = TimeBlock(base*3, base, self.time_dim)
        self.output = nn.Conv3d(base, channels, 3, padding=1)

    def forward(self, x, t, labels=None):
        half = self.time_dim//2
        frequencies = torch.exp(-math.log(10000)*torch.arange(half, device=x.device)/(half-1))
        angles = t.float()[:, None]*frequencies
        embedding = self.time_mlp(torch.cat((angles.sin(), angles.cos()), 1))
        if self.labels is not None:
            if labels is None or labels.shape != (x.shape[0], self.config['label_dim']):
                raise ValueError('Conditional diffusion requires one label vector per volume')
            embedding = embedding+self.labels(labels.float())
        h = self.entry(x, embedding)
        middle = self.attention(self.middle(self.down(h), embedding))
        up = F.interpolate(middle, size=h.shape[2:], mode='trilinear', align_corners=False)
        return self.output(self.up(torch.cat((up, h), 1), embedding))


class DDPM(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        if steps < 2:
            raise ValueError('At least two diffusion steps are required')
        # Cosine schedule reaches approximately unit Gaussian noise even in small smoke runs.
        grid = torch.linspace(0, steps, steps+1, dtype=torch.float64)
        curve = torch.cos(((grid/steps+.008)/1.008)*math.pi/2).square()
        curve = curve/curve[0]
        beta = (1-curve[1:]/curve[:-1]).clamp(1e-5, .999).float()
        alpha = 1-beta
        cumulative = torch.cumprod(alpha, 0)
        previous = F.pad(cumulative[:-1], (1, 0), value=1.)
        for name, value in dict(beta=beta, alpha=alpha, cumulative=cumulative,
            posterior_variance=beta*(1-previous)/(1-cumulative),
            posterior_clean_coefficient=beta*previous.sqrt()/(1-cumulative),
            posterior_noisy_coefficient=(1-previous)*alpha.sqrt()/(1-cumulative)).items():
            self.register_buffer(name, value)
        self.steps = steps

    def add_noise(self, clean, timesteps, noise):
        a = self.cumulative[timesteps][:, None, None, None, None]
        return a.sqrt()*clean + (1-a).sqrt()*noise

    @torch.no_grad()
    def sample(self, model, shape, labels=None, seed=42, clip_range=None):
        device = self.beta.device
        rng = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, generator=rng, device=device)
        if clip_range is not None:
            lower, upper = [torch.as_tensor(v, device=device, dtype=x.dtype) for v in clip_range]
            if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or torch.any(lower > upper):
                raise ValueError('Invalid denoised latent bounds')
        model.eval()
        for step in reversed(range(self.steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            epsilon = model(x, t, labels)
            clean = (x-(1-self.cumulative[step]).sqrt()*epsilon)/self.cumulative[step].sqrt()
            if clip_range is not None:
                # A modest denoiser error at near-zero SNR otherwise explodes by 1/sqrt(alpha_bar).
                # Bounds come from the trained codebook, not validation data or target images.
                clean = torch.maximum(torch.minimum(clean, upper), lower)
            mean = self.posterior_clean_coefficient[step]*clean+self.posterior_noisy_coefficient[step]*x
            x = mean if step == 0 else mean+self.posterior_variance[step].sqrt()*torch.randn(shape, generator=rng, device=device)
        return x

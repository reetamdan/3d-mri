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
        assert self.project.bias is not None
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        batch, channels, *spatial = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(batch, 3, self.heads, channels // self.heads, -1).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        h = h.transpose(-1, -2).reshape(batch, channels, *spatial)
        return x + self.project(h)


class UNet3D(nn.Module):
    def __init__(self, channels=16, base=32, label_dim=0, attention=False, attention_heads=4,
                 channel_mult=None, allow_null_labels=False):
        super().__init__()
        if base < 2:
            raise ValueError('U-Net base width must be at least two')
        self.config = dict(channels=channels, base=base, label_dim=label_dim,
                           attention=attention, attention_heads=attention_heads,
                           channel_mult=channel_mult, allow_null_labels=allow_null_labels)
        self.time_dim = base*4
        self.time_mlp = nn.Sequential(nn.Linear(self.time_dim, self.time_dim), nn.SiLU(), nn.Linear(self.time_dim, self.time_dim))
        self.labels = nn.Linear(label_dim, self.time_dim) if label_dim else None
        if label_dim and allow_null_labels:
            self.null_label = nn.Parameter(torch.zeros(self.time_dim))
        if channel_mult is None:
            self.entry = TimeBlock(channels, base, self.time_dim)
            self.down = nn.Conv3d(base, base*2, 4, 2, 1)
            self.middle = TimeBlock(base*2, base*2, self.time_dim)
            self.attention = SpatialAttention3D(base*2, attention_heads) if attention else nn.Identity()
            self.up = TimeBlock(base*3, base, self.time_dim)
            self.output = nn.Conv3d(base, channels, 3, padding=1)
        else:
            if len(channel_mult) < 2 or any(not isinstance(v, int) or v < 1 for v in channel_mult):
                raise ValueError('Use at least two positive integer channel multipliers')
            widths = [base*v for v in channel_mult]
            self.entry = TimeBlock(channels, widths[0], self.time_dim)
            self.downsamples = nn.ModuleList(nn.Conv3d(a, b, 4, 2, 1) for a, b in zip(widths, widths[1:]))
            self.encoder_blocks = nn.ModuleList(TimeBlock(w, w, self.time_dim) for w in widths[1:])
            self.middle = TimeBlock(widths[-1], widths[-1], self.time_dim)
            self.attention = SpatialAttention3D(widths[-1], attention_heads) if attention else nn.Identity()
            self.decoder_blocks = nn.ModuleList(TimeBlock(a+b, a, self.time_dim)
                                                for a, b in reversed(list(zip(widths, widths[1:]))))
            self.output = nn.Conv3d(widths[0], channels, 3, padding=1)
            assert self.output.bias is not None
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, t, labels=None, condition_mask=None):
        half = self.time_dim//2
        frequencies = torch.exp(-math.log(10000)*torch.arange(half, device=x.device)/(half-1))
        angles = t.float()[:, None]*frequencies
        embedding = self.time_mlp(torch.cat((angles.sin(), angles.cos()), 1))
        if self.labels is not None:
            if labels is None and self.config['allow_null_labels']:
                condition = self.null_label[None].expand(x.shape[0], -1)
            elif labels is None or labels.shape != (x.shape[0], self.config['label_dim']):
                raise ValueError('Conditional diffusion requires one label vector per volume')
            else:
                condition = self.labels(labels.float())
                if condition_mask is not None:
                    if not self.config['allow_null_labels'] or condition_mask.shape != (x.shape[0],):
                        raise ValueError('Label dropout requires a trained null label and one mask per input')
                    condition = torch.where(condition_mask[:, None].bool(), condition, self.null_label[None])
            embedding = embedding+condition
        h = self.entry(x, embedding)
        if self.config['channel_mult'] is not None:
            if min(x.shape[2:]) < 2**len(self.downsamples):
                raise ValueError('Latent dimensions are too small for the configured U-Net depth')
            skips = [h]
            for down, block in zip(self.downsamples, self.encoder_blocks):
                h = block(down(h), embedding); skips.append(h)
            h = self.attention(self.middle(h, embedding))
            for skip, block in zip(reversed(skips[:-1]), self.decoder_blocks):
                h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
                h = block(torch.cat((h, skip), 1), embedding)
            return self.output(h)
        middle = self.attention(self.middle(self.down(h), embedding))
        up = F.interpolate(middle, size=h.shape[2:], mode='trilinear', align_corners=False)
        return self.output(self.up(torch.cat((up, h), 1), embedding))


class DDPM(nn.Module):
    beta: torch.Tensor
    alpha: torch.Tensor
    cumulative: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_clean_coefficient: torch.Tensor
    posterior_noisy_coefficient: torch.Tensor

    def __init__(self, steps=1000, prediction_type='epsilon'):
        super().__init__()
        if steps < 2:
            raise ValueError('At least two diffusion steps are required')
        if prediction_type not in ('epsilon', 'v'):
            raise ValueError('prediction_type must be epsilon or v')
        self.prediction_type = prediction_type
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

    def training_target(self, clean, timesteps, noise):
        if self.prediction_type == 'epsilon':
            return noise
        a = self.cumulative[timesteps][:, None, None, None, None]
        return a.sqrt()*noise-(1-a).sqrt()*clean

    def clean_and_noise(self, noisy, step, prediction):
        a = self.cumulative[step]
        if self.prediction_type == 'v':
            return a.sqrt()*noisy-(1-a).sqrt()*prediction, (1-a).sqrt()*noisy+a.sqrt()*prediction
        return (noisy-(1-a).sqrt()*prediction)/a.sqrt(), prediction

    @torch.no_grad()
    def sample(self, model, shape, labels=None, seed=42, clip_range=None, method='ddpm',
               sampling_steps=None, guidance_scale=1., eta=0., return_diagnostics=False):
        if method not in ('ddpm', 'ddim') or not math.isfinite(guidance_scale) or guidance_scale < 0 or not 0 <= eta <= 1:
            raise ValueError('Invalid sampler, guidance scale, or DDIM eta')
        count = self.steps if sampling_steps is None else sampling_steps
        if not isinstance(count, int) or not 1 <= count <= self.steps or (method == 'ddpm' and count != self.steps):
            raise ValueError('DDPM uses every step; DDIM sampling_steps must be between one and training steps')
        if guidance_scale != 1 and (labels is None or not model.config.get('allow_null_labels')):
            raise ValueError('Classifier-free guidance requires a model trained with label dropout')
        device = self.beta.device
        rng = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, generator=rng, device=device)
        if clip_range is not None:
            lower, upper = [torch.as_tensor(v, device=device, dtype=x.dtype) for v in clip_range]
            if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or torch.any(lower > upper):
                raise ValueError('Invalid denoised latent bounds')
        model.eval()
        sequence = list(reversed(range(self.steps))) if method == 'ddpm' else torch.linspace(self.steps-1, 0, count).round().long().tolist()
        diagnostics = []
        for index, step in enumerate(sequence):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            prediction = model(x, t, labels)
            if guidance_scale != 1:
                unconditional = model(x, t, None)
                prediction = unconditional+guidance_scale*(prediction-unconditional)
            clean, epsilon = self.clean_and_noise(x, step, prediction)
            fraction = 0.
            if clip_range is not None:
                # A modest denoiser error at near-zero SNR otherwise explodes by 1/sqrt(alpha_bar).
                # Bounds come from the trained codebook, not validation data or target images.
                if return_diagnostics:
                    fraction = float(((clean < lower) | (clean > upper)).float().mean())
                clean = torch.maximum(torch.minimum(clean, upper), lower)
            if method == 'ddpm':
                mean = self.posterior_clean_coefficient[step]*clean+self.posterior_noisy_coefficient[step]*x
                x = mean if step == 0 else mean+self.posterior_variance[step].sqrt()*torch.randn(shape, generator=rng, device=device)
            else:
                previous = self.cumulative[sequence[index+1]] if index+1 < len(sequence) else torch.ones((), device=device)
                a = self.cumulative[step]
                if clip_range is not None:
                    epsilon = (x-a.sqrt()*clean)/(1-a).sqrt()
                sigma = eta*torch.sqrt(((1-previous)/(1-a)*(1-a/previous)).clamp_min(0))
                x = previous.sqrt()*clean+(1-previous-sigma.square()).clamp_min(0).sqrt()*epsilon
                if eta and index+1 < len(sequence):
                    x = x+sigma*torch.randn(shape, generator=rng, device=device)
            if return_diagnostics:
                diagnostics.append(dict(step=step, clipped_fraction=fraction, latent_std=float(x.std()),
                                        predicted_clean_abs_max=float(clean.abs().max())))
        return (x, diagnostics) if return_diagnostics else x
